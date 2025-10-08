"""
SetFit Few-Shot Training Script
================================
Train SetFit model with few-shot learning on food safety event detection.

Based on:
- Reference: /Users/lowrenhwa/Desktop/Code/contrastive_train.py
- Docs: https://huggingface.co/docs/setfit/en/quickstart
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset, DatasetDict
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


class StratifiedSampler:
    """
    Stratified few-shot sampler for imbalanced datasets.

    Ensures minority classes get more samples relative to their size.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def sample_balanced(
        self,
        df: pd.DataFrame,
        k_shot: int,
        label_col: str = 'label_id',
        boost_minority: bool = True
    ) -> pd.DataFrame:
        """
        Sample k-shot examples with stratified sampling.

        Args:
            df: Training dataframe
            k_shot: Base number of samples per class
            label_col: Label column name
            boost_minority: Give minority classes more samples

        Returns:
            Sampled dataframe
        """
        sampled_dfs = []
        class_counts = df[label_col].value_counts()

        for class_id in sorted(df[label_col].unique()):
            class_df = df[df[label_col] == class_id]

            # Boost minority classes (double samples for classes with <15% of data)
            if boost_minority and class_counts[class_id] < len(df) * 0.15:
                n_samples = min(k_shot * 2, len(class_df))
                logger.info(f"Class {class_id}: Boosting to {n_samples} samples (minority)")
            else:
                n_samples = min(k_shot, len(class_df))
                logger.info(f"Class {class_id}: Sampling {n_samples} samples")

            sampled = class_df.sample(n=n_samples, random_state=self.seed)
            sampled_dfs.append(sampled)

        return pd.concat(sampled_dfs, ignore_index=True)


def load_gold_data(gold_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load preprocessed gold data.

    Args:
        gold_dir: Directory containing gold parquet files

    Returns:
        Dictionary with train/val/test dataframes
    """
    logger.info(f"Loading gold data from {gold_dir}")

    data = {}
    for split in ['train', 'val', 'test']:
        file_path = gold_dir / f"{split}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Gold file not found: {file_path}")

        df = pd.read_parquet(file_path)
        data[split] = df
        logger.info(f"Loaded {split}: {len(df)} samples")

    return data


def df_to_dataset(df: pd.DataFrame, text_col: str = 'text_ranked') -> Dataset:
    """
    Convert pandas DataFrame to HuggingFace Dataset.

    Args:
        df: Pandas dataframe
        text_col: Text column to use ('text_ranked' recommended)

    Returns:
        HuggingFace Dataset
    """
    return Dataset.from_dict({
        'text': df[text_col].tolist(),
        'label': df['label_id'].tolist()
    })


def compute_metrics(y_true, y_pred, label_names: List[str]) -> Dict:
    """
    Compute evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: List of label names

    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Per-class metrics
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'report': report,
        'confusion_matrix': cm
    }

    return metrics


def print_results(metrics: Dict, label_names: List[str], k_shot: int):
    """Print formatted results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"SetFit Few-Shot Results (k={k_shot})")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {metrics['macro_recall']:.4f}")

    logger.info(f"\nPer-Class Performance:")
    for i, label in enumerate(label_names):
        label_metrics = metrics['report'][label]
        logger.info(
            f"  {label:20s} - "
            f"P: {label_metrics['precision']:.4f}, "
            f"R: {label_metrics['recall']:.4f}, "
            f"F1: {label_metrics['f1-score']:.4f}"
        )

    logger.info(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    logger.info(f"{'':20s} " + " ".join([f"{ln:10s}" for ln in label_names]))
    for i, label in enumerate(label_names):
        logger.info(f"{label:20s} " + " ".join([f"{cm[i][j]:10d}" for j in range(len(label_names))]))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set seed
    seed_everything(cfg.seed)

    # Load gold data
    gold_dir = Path(cfg.paths.gold_dir)
    data = load_gold_data(gold_dir)

    df_train = data['train']
    df_val = data['val']
    df_test = data['test']

    # Label mapping
    label_names = ['Food Poisoning', 'Zoonotic Disease', 'Food Recall', 'Negative']

    # Log class distribution
    logger.info(f"\nOriginal class distribution (train):")
    for label_id, count in df_train['label_id'].value_counts().sort_index().items():
        logger.info(f"  {label_names[label_id]:20s}: {count} samples")

    # Initialize sampler
    sampler = StratifiedSampler(seed=cfg.seed)

    # Track results across k-shot experiments
    all_results = []

    # Few-shot experiments
    for k_shot in cfg.model.k_shots:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with k={k_shot} shots per class")
        logger.info(f"{'='*60}")

        # Sample training data
        df_train_sampled = sampler.sample_balanced(
            df_train,
            k_shot=k_shot,
            boost_minority=cfg.training.boost_minority
        )

        logger.info(f"Sampled {len(df_train_sampled)} training examples")
        logger.info(f"Distribution: {df_train_sampled['label_id'].value_counts().sort_index().to_dict()}")

        # Convert to HuggingFace datasets
        train_dataset = df_to_dataset(df_train_sampled, text_col=cfg.model.text_column)
        val_dataset = df_to_dataset(df_val, text_col=cfg.model.text_column)
        test_dataset = df_to_dataset(df_test, text_col=cfg.model.text_column)

        # Initialize model (use CPU to avoid MPS memory issues)
        logger.info(f"Loading SetFit model: {cfg.model.name}")
        model = SetFitModel.from_pretrained(
            cfg.model.name,
            labels=label_names,
            multi_target_strategy=cfg.model.multi_target_strategy,
            device="cpu"  # Use CPU to avoid MPS memory issues on macOS
        )

        # Initialize trainer (SetFit API)
        # Disable WandB to avoid login requirement
        import os
        os.environ["WANDB_DISABLED"] = "true"

        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            metric="accuracy",
            column_mapping={"text": "text", "label": "label"},
            batch_size=cfg.training.batch_size,
            num_epochs=cfg.training.num_epochs,
            num_iterations=cfg.training.num_iterations,
            warmup_proportion=cfg.training.warmup_proportion,
            learning_rate=cfg.training.learning_rate,
            seed=cfg.seed,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_preds = model.predict(df_val[cfg.model.text_column].tolist())

        # Convert string predictions to numeric if needed
        if isinstance(val_preds[0], str):
            label_to_id = {name: i for i, name in enumerate(label_names)}
            val_preds = [label_to_id[p] for p in val_preds]

        val_metrics = compute_metrics(
            df_val['label_id'].tolist(),
            val_preds,
            label_names
        )

        logger.info("\nValidation Results:")
        print_results(val_metrics, label_names, k_shot)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_preds = model.predict(df_test[cfg.model.text_column].tolist())

        # Convert string predictions to numeric if needed
        if isinstance(test_preds[0], str):
            label_to_id = {name: i for i, name in enumerate(label_names)}
            test_preds = [label_to_id[p] for p in test_preds]

        test_metrics = compute_metrics(
            df_test['label_id'].tolist(),
            test_preds,
            label_names
        )

        logger.info("\nTest Results:")
        print_results(test_metrics, label_names, k_shot)

        # Save model
        model_dir = Path(cfg.paths.models_dir) / f"setfit_k{k_shot}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_dir))
        logger.info(f"Model saved to {model_dir}")

        # Store results
        all_results.append({
            'k_shot': k_shot,
            'val_accuracy': val_metrics['accuracy'],
            'val_macro_f1': val_metrics['macro_f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_macro_f1': test_metrics['macro_f1'],
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        })

    # Summary of all experiments
    logger.info(f"\n{'='*60}")
    logger.info("Summary of Few-Shot Experiments")
    logger.info(f"{'='*60}")
    logger.info(f"{'k-shot':>10s} {'Val Acc':>10s} {'Val F1':>10s} {'Test Acc':>10s} {'Test F1':>10s}")
    logger.info("-" * 60)

    for result in all_results:
        logger.info(
            f"{result['k_shot']:10d} "
            f"{result['val_accuracy']:10.4f} "
            f"{result['val_macro_f1']:10.4f} "
            f"{result['test_accuracy']:10.4f} "
            f"{result['test_macro_f1']:10.4f}"
        )

    # Save results
    results_df = pd.DataFrame([{
        'k_shot': r['k_shot'],
        'val_accuracy': r['val_accuracy'],
        'val_macro_f1': r['val_macro_f1'],
        'test_accuracy': r['test_accuracy'],
        'test_macro_f1': r['test_macro_f1']
    } for r in all_results])

    results_path = Path(cfg.paths.results_dir) / "setfit_fewshot_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to {results_path}")

    logger.info("\n✅ SetFit few-shot training complete!")


if __name__ == "__main__":
    main()
