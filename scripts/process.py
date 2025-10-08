"""
Unified Data Preprocessing Pipeline
Medallion Architecture: Bronze → Silver → Gold
All layers consolidated in single script for simplicity
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.seed import seed_everything
from src.utils.logger import get_logger

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

console = Console()
logger = get_logger(__name__)


# =====================================================================
# BRONZE LAYER: Raw Data Ingestion
# =====================================================================

class BronzeLayer:
    """Load and validate raw JSON data."""

    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        self.required_fields = ['id', 'title', 'text', 'type', 'entities']

    def load_json(self, file_path: Path) -> List[Dict]:
        """Load JSON file."""
        logger.info(f"Loading {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples from {file_path.name}")
        return data

    def validate_sample(self, sample: Dict, idx: int) -> bool:
        """Validate sample has required fields."""
        for field in self.required_fields:
            if field not in sample:
                logger.warning(f"Sample {idx} missing field: {field}")
                return False
        return True

    def to_dataframe(self, data: List[Dict], split: str = "train") -> pd.DataFrame:
        """Convert JSON to DataFrame."""
        logger.info(f"Converting {len(data)} samples to DataFrame ({split})")

        records = []
        for idx, item in enumerate(tqdm(data, desc=f"Processing {split}")):
            if not self.validate_sample(item, idx):
                continue

            record = {
                'id': str(item['id']),
                'title': item['title'],
                'text': item['text'],
                'type': item['type'],
                'date': item.get('date', None),
                'url': item.get('url', None),
                'num_sentences': len(item['entities']),
                'split': split
            }
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Bronze: {len(df)} valid samples")
        logger.info(f"Missing dates: {df['date'].isna().sum()}")
        logger.info(f"Missing URLs: {df['url'].isna().sum()}")

        return df

    def load_all_splits(self, train_file: str, test_file: str, val_file: Optional[str] = None) -> pd.DataFrame:
        """Load all data splits."""
        dfs = []

        # Train
        train_data = self.load_json(self.raw_data_dir / train_file)
        df_train = self.to_dataframe(train_data, split="train")
        dfs.append(df_train)

        # Test
        test_data = self.load_json(self.raw_data_dir / test_file)
        df_test = self.to_dataframe(test_data, split="test")
        dfs.append(df_test)

        # Val (optional)
        if val_file and (self.raw_data_dir / val_file).exists():
            val_data = self.load_json(self.raw_data_dir / val_file)
            df_val = self.to_dataframe(val_data, split="val")
            dfs.append(df_val)

        df_bronze = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total bronze samples: {len(df_bronze)}")

        return df_bronze

    def save(self, df: pd.DataFrame, output_dir: Path) -> Path:
        """Save bronze data."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "bronze_data.parquet"
        df.to_parquet(output_path, index=False)

        logger.info(f"Bronze saved: {output_path}")
        return output_path


# =====================================================================
# LABEL HANDLER: Multi-label → Single-label
# =====================================================================

class LabelHandler:
    """Extract primary label from multi-label samples."""

    PRIORITY = ['Food Poisoning', 'Zoonotic Disease', 'Food Recall', 'Negative']
    LABEL_TO_ID = {'Food Poisoning': 0, 'Zoonotic Disease': 1, 'Food Recall': 2, 'Negative': 3}
    ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

    def __init__(self):
        self.stats = {'total_samples': 0, 'multilabel_samples': 0, 'conversions': {}}

    def is_multilabel(self, label: str) -> bool:
        """Check if label is multi-label."""
        return '; ' in label

    def extract_primary_label(self, label: str) -> str:
        """Extract primary label by priority."""
        if not self.is_multilabel(label):
            return label

        labels = [l.strip() for l in label.split(';')]
        for priority_label in self.PRIORITY:
            if priority_label in labels:
                if label not in self.stats['conversions']:
                    self.stats['conversions'][label] = priority_label
                return priority_label

        return labels[0]

    def process_labels(self, df: pd.DataFrame, label_col: str = 'type') -> pd.DataFrame:
        """Process labels in DataFrame."""
        self.stats['total_samples'] = len(df)

        df['is_multilabel'] = df[label_col].apply(self.is_multilabel)
        self.stats['multilabel_samples'] = df['is_multilabel'].sum()

        df['primary_label'] = df[label_col].apply(self.extract_primary_label)
        df['label_id'] = df['primary_label'].map(self.LABEL_TO_ID)

        if df['label_id'].isna().any():
            unknown = df[df['label_id'].isna()]['primary_label'].unique()
            raise ValueError(f"Unknown labels: {unknown}")

        logger.info(f"Labels: {self.stats['multilabel_samples']} multi-label converted")

        for label, count in df['primary_label'].value_counts().items():
            pct = count / len(df) * 100
            logger.info(f"  {label}: {count} ({pct:.1f}%)")

        return df

    def get_class_weights(self, df: pd.DataFrame) -> dict:
        """Calculate class weights for imbalanced data."""
        label_counts = df['label_id'].value_counts().sort_index()
        n_samples = len(df)
        n_classes = len(self.LABEL_TO_ID)

        weights = {}
        for label_id, count in label_counts.items():
            weight = n_samples / (n_classes * count)
            weights[label_id] = weight
            label_name = self.ID_TO_LABEL[label_id]
            logger.info(f"Class weight {label_name} (id={label_id}): {weight:.3f}")

        return weights


# =====================================================================
# SENTENCE RANKER: TF-IDF for long documents
# =====================================================================

class SentenceRanker:
    """Rank and select important sentences using TF-IDF."""

    def __init__(self, keep_ratio: float = 0.5, min_sentences: int = 3, max_sentences: int = 50):
        self.keep_ratio = keep_ratio
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.stats = {
            'total_docs': 0,
            'avg_original_sentences': 0,
            'avg_ranked_sentences': 0,
            'avg_original_chars': 0,
            'avg_ranked_chars': 0
        }

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        return [s for s in sentences if len(s.strip()) > 0]

    def rank_sentences_tfidf(self, sentences: List[str]) -> np.ndarray:
        """Rank sentences by TF-IDF."""
        if len(sentences) <= self.min_sentences:
            return np.arange(len(sentences))

        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            ranked_indices = np.argsort(scores)[::-1]
        except:
            ranked_indices = np.arange(len(sentences))

        return ranked_indices

    def select_top_sentences(self, text: str, preserve_order: bool = True) -> str:
        """Select top sentences from text."""
        sentences = self.split_sentences(text)

        if len(sentences) <= self.min_sentences:
            return text

        ranked_indices = self.rank_sentences_tfidf(sentences)

        n_keep = max(
            self.min_sentences,
            min(self.max_sentences, int(len(sentences) * self.keep_ratio))
        )

        top_indices = ranked_indices[:n_keep]
        if preserve_order:
            top_indices = sorted(top_indices)

        ranked_text = ' '.join([sentences[i] for i in top_indices])
        return ranked_text

    def process_dataframe(self, df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
        """Process entire DataFrame."""
        logger.info(f"Ranking sentences for {len(df)} documents...")

        original_sentences = []
        ranked_sentences = []
        original_chars = []
        ranked_chars = []

        for idx, row in df.iterrows():
            text = row[text_col]

            orig_sents = self.split_sentences(text)
            original_sentences.append(len(orig_sents))
            original_chars.append(len(text))

            ranked_text = self.select_top_sentences(text)

            rank_sents = self.split_sentences(ranked_text)
            ranked_sentences.append(len(rank_sents))
            ranked_chars.append(len(ranked_text))

            df.at[idx, 'text_ranked'] = ranked_text

        self.stats['total_docs'] = len(df)
        self.stats['avg_original_sentences'] = np.mean(original_sentences)
        self.stats['avg_ranked_sentences'] = np.mean(ranked_sentences)
        self.stats['avg_original_chars'] = np.mean(original_chars)
        self.stats['avg_ranked_chars'] = np.mean(ranked_chars)

        logger.info(f"Sentences: {self.stats['avg_original_sentences']:.1f} → {self.stats['avg_ranked_sentences']:.1f}")
        logger.info(f"Chars: {self.stats['avg_original_chars']:.0f} → {self.stats['avg_ranked_chars']:.0f}")

        return df


# =====================================================================
# SILVER LAYER: Cleaning & Validation
# =====================================================================

class SilverLayer:
    """Clean and validate data."""

    def __init__(self, apply_sentence_ranking: bool = True, ranking_keep_ratio: float = 0.5):
        self.apply_sentence_ranking = apply_sentence_ranking
        self.label_handler = LabelHandler()
        self.sentence_ranker = SentenceRanker(keep_ratio=ranking_keep_ratio) if apply_sentence_ranking else None
        self.stats = {'input_samples': 0, 'output_samples': 0, 'cleaned_chars_removed': 0}

    def clean_text(self, text: str) -> str:
        """Clean text content."""
        if not isinstance(text, str):
            return ""

        original_len = len(text)

        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)

        # Normalize quotes and whitespace
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = re.sub(r'\s+', ' ', text).strip()

        self.stats['cleaned_chars_removed'] += (original_len - len(text))
        return text

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality."""
        original_len = len(df)

        # Remove empty texts
        df = df[df['text'].str.len() > 0].copy()
        logger.info(f"Removed {original_len - len(df)} empty texts")

        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['text'], keep='first')
        logger.info(f"Removed {original_len - len(df)} duplicates")

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        missing_text = df['text'].isna().sum()
        if missing_text > 0:
            logger.warning(f"Dropping {missing_text} samples with missing text")
            df = df.dropna(subset=['text'])

        logger.info(f"Missing dates: {df['date'].isna().sum()}")
        logger.info(f"Missing URLs: {df['url'].isna().sum()}")

        return df

    def add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata columns."""
        df['text_length'] = df['text'].str.len()
        df['text_cleaned_length'] = df['text_cleaned'].str.len()

        if self.apply_sentence_ranking:
            df['text_ranked_length'] = df['text_ranked'].str.len()

        df['token_count'] = (df['text_cleaned_length'] / 4).astype(int)

        if self.apply_sentence_ranking:
            df['token_count_ranked'] = (df['text_ranked_length'] / 4).astype(int)
            exceeds_512 = (df['token_count'] > 512).sum()
            exceeds_512_ranked = (df['token_count_ranked'] > 512).sum()
            logger.info(f"Exceeding 512 tokens: {exceeds_512} → {exceeds_512_ranked}")

        return df

    def process(self, df_bronze: pd.DataFrame) -> pd.DataFrame:
        """Process bronze to silver."""
        self.stats['input_samples'] = len(df_bronze)
        logger.info(f"Processing {len(df_bronze)} bronze samples to silver...")

        df = df_bronze.copy()

        # Clean text
        logger.info("Step 1/6: Cleaning text...")
        df['text_cleaned'] = df['text'].apply(self.clean_text)

        # Extract labels
        logger.info("Step 2/6: Extracting primary labels...")
        df = self.label_handler.process_labels(df, label_col='type')

        # Rank sentences
        if self.apply_sentence_ranking:
            logger.info("Step 3/6: Ranking sentences...")
            df = self.sentence_ranker.process_dataframe(df, text_col='text_cleaned')
        else:
            logger.info("Step 3/6: Skipping sentence ranking")
            df['text_ranked'] = df['text_cleaned']

        # Handle missing values
        logger.info("Step 4/6: Handling missing values...")
        df = self.handle_missing_values(df)

        # Validate
        logger.info("Step 5/6: Validating data...")
        df = self.validate_data(df)

        # Add metadata
        logger.info("Step 6/6: Adding metadata...")
        df = self.add_metadata(df)

        self.stats['output_samples'] = len(df)
        logger.info(f"Silver complete: {self.stats['input_samples']} → {self.stats['output_samples']}")

        return df

    def save(self, df: pd.DataFrame, output_dir: Path) -> Path:
        """Save silver data."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "silver_data.parquet"
        df.to_parquet(output_path, index=False)

        logger.info(f"Silver saved: {output_path}")
        return output_path


# =====================================================================
# GOLD LAYER: ML-Ready Splits
# =====================================================================

class GoldLayer:
    """Create ML-ready train/val/test splits."""

    def __init__(self, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 42):
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.stats = {'train_samples': 0, 'val_samples': 0, 'test_samples': 0}

    def create_splits(self, df: pd.DataFrame, use_predefined_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified splits."""
        if use_predefined_split and 'split' in df.columns:
            logger.info("Using predefined train/test split")

            df_test = df[df['split'] == 'test'].copy()
            df_train_full = df[df['split'] == 'train'].copy()

            df_train, df_val = train_test_split(
                df_train_full,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=df_train_full['label_id']
            )
        else:
            logger.info("Creating new stratified splits")

            df_train_val, df_test = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=df['label_id']
            )

            val_ratio = self.val_size / (1 - self.test_size)
            df_train, df_val = train_test_split(
                df_train_val,
                test_size=val_ratio,
                random_state=self.random_state,
                stratify=df_train_val['label_id']
            )

        df_train['split'] = 'train'
        df_val['split'] = 'val'
        df_test['split'] = 'test'

        self.stats['train_samples'] = len(df_train)
        self.stats['val_samples'] = len(df_val)
        self.stats['test_samples'] = len(df_test)

        logger.info(f"Splits: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

        # Log class distribution
        for split_name, split_df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
            logger.info(f"\n{split_name}:")
            for label, count in split_df['primary_label'].value_counts().items():
                pct = count / len(split_df) * 100
                logger.info(f"  {label}: {count} ({pct:.1f}%)")

        return df_train, df_val, df_test

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select feature columns."""
        feature_cols = [
            'id', 'text', 'text_cleaned', 'text_ranked', 'title',
            'primary_label', 'label_id', 'split', 'is_multilabel',
            'text_length', 'text_cleaned_length', 'text_ranked_length',
            'token_count', 'token_count_ranked', 'date', 'url'
        ]

        available_cols = [col for col in feature_cols if col in df.columns]
        return df[available_cols].copy()

    def process(self, df_silver: pd.DataFrame, use_predefined_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process silver to gold."""
        logger.info(f"Processing {len(df_silver)} silver samples to gold...")

        df_train, df_val, df_test = self.create_splits(df_silver, use_predefined_split)

        df_train = self.prepare_features(df_train)
        df_val = self.prepare_features(df_val)
        df_test = self.prepare_features(df_test)

        logger.info("Gold layer complete!")
        return df_train, df_val, df_test

    def save(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
             output_dir: Path, formats: list = ['parquet', 'csv']) -> dict:
        """Save gold data."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for fmt in formats:
            logger.info(f"Saving {fmt.upper()} format...")

            if fmt == 'parquet':
                train_path = output_dir / "train.parquet"
                val_path = output_dir / "val.parquet"
                test_path = output_dir / "test.parquet"

                df_train.to_parquet(train_path, index=False)
                df_val.to_parquet(val_path, index=False)
                df_test.to_parquet(test_path, index=False)

            elif fmt == 'csv':
                train_path = output_dir / "train.csv"
                val_path = output_dir / "val.csv"
                test_path = output_dir / "test.csv"

                df_train.to_csv(train_path, index=False)
                df_val.to_csv(val_path, index=False)
                df_test.to_csv(test_path, index=False)

            saved_files[fmt] = {'train': train_path, 'val': val_path, 'test': test_path}

        logger.info(f"Gold saved to {output_dir}")
        return saved_files

    def save_combined(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                     df_test: pd.DataFrame, output_dir: Path) -> Path:
        """Save combined dataset."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df_combined = pd.concat([df_train, df_val, df_test], ignore_index=True)
        output_path = output_dir / "gold_data_combined.parquet"
        df_combined.to_parquet(output_path, index=False)

        logger.info(f"Combined gold saved: {output_path}")
        return output_path


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def print_layer_header(layer_name: str, emoji: str = "📦"):
    """Print layer header."""
    console.print(f"\n{emoji} [bold cyan]{layer_name} Layer[/bold cyan] {emoji}\n", style="bold")


def print_stats_table(stats: dict, title: str):
    """Print statistics table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for key, value in stats.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                table.add_row(str(key), f"{value:.2f}")
            else:
                table.add_row(str(key), f"{value:,}")
        else:
            table.add_row(str(key), str(value))

    console.print(table)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Run unified preprocessing pipeline."""

    # Print header
    console.print(Panel.fit(
        "[bold]Food Safety Event Detection[/bold]\n"
        "[cyan]Unified Preprocessing Pipeline[/cyan]\n"
        "[dim]Medallion Architecture: Bronze → Silver → Gold[/dim]",
        border_style="blue"
    ))

    console.print("\n[bold]Configuration:[/bold]")
    console.print(OmegaConf.to_yaml(cfg))

    # Set seed
    seed_everything(cfg.seed)
    logger.info("Starting unified preprocessing pipeline...")

    # =====================
    # BRONZE LAYER
    # =====================
    print_layer_header("BRONZE", "🥉")

    bronze_layer = BronzeLayer(raw_data_dir=Path(cfg.paths.raw_data_dir))
    df_bronze = bronze_layer.load_all_splits(
        train_file=cfg.data.bronze.train_file,
        test_file=cfg.data.bronze.test_file,
        val_file=cfg.data.bronze.val_file
    )
    bronze_path = bronze_layer.save(df_bronze, output_dir=Path(cfg.paths.bronze_dir))

    bronze_stats = {
        "Total samples": len(df_bronze),
        "Train samples": (df_bronze['split'] == 'train').sum(),
        "Test samples": (df_bronze['split'] == 'test').sum(),
        "Missing dates": df_bronze['date'].isna().sum(),
        "Missing URLs": df_bronze['url'].isna().sum(),
    }
    print_stats_table(bronze_stats, "Bronze Layer Statistics")

    # =====================
    # SILVER LAYER
    # =====================
    print_layer_header("SILVER", "🥈")

    silver_layer = SilverLayer(
        apply_sentence_ranking=cfg.data.silver.apply_sentence_ranking,
        ranking_keep_ratio=cfg.data.silver.ranking_keep_ratio
    )
    df_silver = silver_layer.process(df_bronze)
    silver_path = silver_layer.save(df_silver, output_dir=Path(cfg.paths.silver_dir))

    print_stats_table(silver_layer.stats, "Silver Layer Statistics")

    # Get class weights
    if cfg.data.imbalance.compute_class_weights:
        class_weights = silver_layer.label_handler.get_class_weights(df_silver)

        console.print("\n[bold]Class Weights:[/bold]")
        for label_id, weight in class_weights.items():
            label_name = silver_layer.label_handler.ID_TO_LABEL[label_id]
            console.print(f"  {label_name} (id={label_id}): [green]{weight:.3f}[/green]")

    # =====================
    # GOLD LAYER
    # =====================
    print_layer_header("GOLD", "🥇")

    gold_layer = GoldLayer(
        val_size=cfg.data.gold.val_size,
        test_size=cfg.data.gold.test_size,
        random_state=cfg.seed
    )
    df_train, df_val, df_test = gold_layer.process(
        df_silver,
        use_predefined_split=cfg.data.gold.use_predefined_split
    )

    saved_files = gold_layer.save(
        df_train, df_val, df_test,
        output_dir=Path(cfg.paths.gold_dir),
        formats=cfg.data.gold.save_formats
    )

    if cfg.data.gold.save_combined:
        combined_path = gold_layer.save_combined(
            df_train, df_val, df_test,
            output_dir=Path(cfg.paths.gold_dir)
        )

    print_stats_table(gold_layer.stats, "Gold Layer Statistics")

    # =====================
    # SUMMARY
    # =====================
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold green]✅ Pipeline Complete![/bold green]\n\n"
        f"[cyan]Bronze:[/cyan] {bronze_path}\n"
        f"[cyan]Silver:[/cyan] {silver_path}\n"
        f"[cyan]Gold:[/cyan] {cfg.paths.gold_dir}\n\n"
        "[dim]Data is now ready for model training![/dim]",
        border_style="green"
    ))

    console.print("\n[bold]Output Files:[/bold]")
    console.print(f"📁 Bronze: [cyan]{bronze_path}[/cyan]")
    console.print(f"📁 Silver: [cyan]{silver_path}[/cyan]")

    for fmt, paths in saved_files.items():
        console.print(f"\n[bold]{fmt.upper()} files:[/bold]")
        for split, path in paths.items():
            console.print(f"  {split}: [cyan]{path}[/cyan]")

    logger.info("Unified preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
