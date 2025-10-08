"""
Data Preprocessing Pipeline
Medallion Architecture: Bronze → Silver → Gold
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data.bronze import BronzeLayer
from src.data.silver import SilverLayer
from src.data.gold import GoldLayer
from src.utils.seed import seed_everything
from src.utils.logger import get_logger

console = Console()


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
    """
    Run data preprocessing pipeline.

    Pipeline:
    1. Bronze: Load raw data
    2. Silver: Clean and validate
    3. Gold: Create ML-ready splits
    """
    # Print configuration
    console.print(Panel.fit(
        "[bold]Food Safety Event Detection[/bold]\n"
        "[cyan]Data Preprocessing Pipeline[/cyan]\n"
        "[dim]Medallion Architecture: Bronze → Silver → Gold[/dim]",
        border_style="blue"
    ))

    console.print("\n[bold]Configuration:[/bold]")
    console.print(OmegaConf.to_yaml(cfg))

    # Set seed
    seed_everything(cfg.seed)
    logger = get_logger(__name__)

    logger.info("Starting data preprocessing pipeline...")

    # =====================
    # BRONZE LAYER
    # =====================
    print_layer_header("BRONZE", "🥉")

    bronze_layer = BronzeLayer(raw_data_dir=Path(cfg.paths.raw_data_dir))

    # Load data
    df_bronze = bronze_layer.load_all_splits(
        train_file=cfg.data.bronze.train_file,
        test_file=cfg.data.bronze.test_file,
        val_file=cfg.data.bronze.val_file
    )

    # Save bronze
    bronze_path = bronze_layer.save(df_bronze, output_dir=Path(cfg.paths.bronze_dir))

    # Print bronze stats
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

    # Process bronze to silver
    df_silver = silver_layer.process(df_bronze)

    # Save silver
    silver_path = silver_layer.save(df_silver, output_dir=Path(cfg.paths.silver_dir))

    # Print silver stats
    silver_stats = silver_layer.get_stats()
    print_stats_table(silver_stats, "Silver Layer Statistics")

    # Get class weights
    if cfg.data.imbalance.compute_class_weights:
        from src.data.label_handler import LabelHandler
        label_handler = LabelHandler()
        class_weights = label_handler.get_class_weights(df_silver)

        console.print("\n[bold]Class Weights (for handling imbalance):[/bold]")
        for label_id, weight in class_weights.items():
            label_name = label_handler.ID_TO_LABEL[label_id]
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

    # Process silver to gold
    df_train, df_val, df_test = gold_layer.process(
        df_silver,
        use_predefined_split=cfg.data.gold.use_predefined_split
    )

    # Save gold
    saved_files = gold_layer.save(
        df_train, df_val, df_test,
        output_dir=Path(cfg.paths.gold_dir),
        formats=cfg.data.gold.save_formats
    )

    # Save combined dataset
    if cfg.data.gold.save_combined:
        combined_path = gold_layer.save_combined(
            df_train, df_val, df_test,
            output_dir=Path(cfg.paths.gold_dir)
        )

    # Print gold stats
    gold_stats = gold_layer.get_stats()
    print_stats_table(gold_stats, "Gold Layer Statistics")

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

    # Print file paths
    console.print("\n[bold]Output Files:[/bold]")
    console.print(f"📁 Bronze: [cyan]{bronze_path}[/cyan]")
    console.print(f"📁 Silver: [cyan]{silver_path}[/cyan]")

    for fmt, paths in saved_files.items():
        console.print(f"\n[bold]{fmt.upper()} files:[/bold]")
        for split, path in paths.items():
            console.print(f"  {split}: [cyan]{path}[/cyan]")

    logger.info("Data preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
