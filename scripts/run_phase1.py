"""
run_phase1.py — Phase 1 Orchestration Script

Executes the complete Phase 1 pipeline:
  1. Load and validate configuration
  2. Audit dataset structure and integrity
  3. Build split manifests (or validate existing splits)
  4. Verify tf.data pipeline produces correct batches
  5. Run full EDA visualization suite

Usage:
    # From neurovision-ai/ project root:
    python scripts/run_phase1.py

    # With custom .env:
    python scripts/run_phase1.py --env /path/to/.env

    # Skip slow EDA plots:
    python scripts/run_phase1.py --skip-eda

Environment:
    DATASET_ROOT  — absolute path to brain tumor dataset
    PROJECT_ROOT  — absolute path to neurovision-ai/ directory
    (see .env.example)
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.utils.logger import logger, setup_logger
from src.utils.config import load_config
from src.data.dataset_organizer import DatasetOrganizer
from src.data.splitter import DatasetSplitter
from src.data.data_loader import BrainTumorDataLoader
from src.visualization.eda_visualizer import EDAVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroVision AI — Phase 1: Dataset Organization & Preprocessing"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=str(project_root / ".env"),
        help="Path to .env file (default: <project_root>/.env)",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        default=False,
        help="Skip EDA visualization (faster run for CI/quick checks)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    return parser.parse_args()


def verify_pipeline(config, loader: BrainTumorDataLoader) -> None:
    """Run one batch through the tf.data pipeline and verify shapes."""
    import tensorflow as tf
    logger.info("Verifying tf.data pipeline...")

    train_ds = loader.get_train_dataset_from_dir()
    images, labels = next(iter(train_ds))

    expected_img_shape = (
        config.training.batch_size,
        config.preprocessing.image_size[0],
        config.preprocessing.image_size[1],
        config.preprocessing.channels,
    )
    expected_lbl_shape = (config.training.batch_size, config.classes.num_classes)

    assert images.shape == expected_img_shape, (
        f"Image shape mismatch: got {images.shape}, expected {expected_img_shape}"
    )
    assert labels.shape == expected_lbl_shape, (
        f"Label shape mismatch: got {labels.shape}, expected {expected_lbl_shape}"
    )

    img_min = float(tf.reduce_min(images).numpy())
    img_max = float(tf.reduce_max(images).numpy())
    logger.info(
        f"Pipeline OK | "
        f"image batch: {images.shape} | "
        f"label batch: {labels.shape} | "
        f"pixel range: [{img_min:.2f}, {img_max:.2f}]"
    )


def main() -> None:
    args = parse_args()

    # Load .env before any config resolution
    load_dotenv(args.env)

    # -----------------------------------------------------------------
    # Step 0: Logging
    # -----------------------------------------------------------------
    # Config not yet loaded — use a temp log dir
    temp_log_dir = project_root / "outputs" / "logs"
    temp_log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=temp_log_dir, level="INFO")

    logger.info("=" * 65)
    logger.info("  NeuroVision AI — Phase 1: Dataset Organization & Preprocessing")
    logger.info("=" * 65)
    t_start = time.time()

    # -----------------------------------------------------------------
    # Step 1: Load configuration
    # -----------------------------------------------------------------
    logger.info("\n[STEP 1] Loading configuration...")
    config = load_config(args.config)

    # Re-initialize logger with configured log directory
    setup_logger(
        log_dir=config.paths.logs_dir,
        level="INFO",
    )

    # -----------------------------------------------------------------
    # Step 2: Dataset audit
    # -----------------------------------------------------------------
    logger.info("\n[STEP 2] Auditing dataset...")
    organizer = DatasetOrganizer(config)
    stats = organizer.audit()

    if stats.total_corrupted > 0:
        logger.warning(
            f"Found {stats.total_corrupted} corrupted images. "
            "They will be excluded from all splits automatically."
        )

    # -----------------------------------------------------------------
    # Step 3: Build split manifests
    # -----------------------------------------------------------------
    logger.info("\n[STEP 3] Building split manifests...")
    splitter = DatasetSplitter(config)
    manifests = splitter.validate_existing_splits(stats)

    for manifest in manifests:
        logger.info(
            f"  [{manifest.name.upper()}] {manifest.num_samples} samples | "
            f"{manifest.class_distribution}"
        )

    # -----------------------------------------------------------------
    # Step 4: Verify tf.data pipeline
    # -----------------------------------------------------------------
    logger.info("\n[STEP 4] Verifying tf.data pipeline...")
    try:
        loader = BrainTumorDataLoader(config)
        verify_pipeline(config, loader)
        logger.success("tf.data pipeline verified successfully.")
    except Exception as e:
        logger.error(f"Pipeline verification failed: {e}")
        logger.warning("Continuing — install TensorFlow to enable pipeline check.")

    # -----------------------------------------------------------------
    # Step 5: EDA Visualizations
    # -----------------------------------------------------------------
    if args.skip_eda:
        logger.info("\n[STEP 5] EDA skipped (--skip-eda flag).")
    else:
        logger.info("\n[STEP 5] Running EDA visualization pipeline...")
        visualizer = EDAVisualizer(config)
        saved_plots = visualizer.run_full_eda(stats)
        logger.info(f"  {len(saved_plots)} plots saved to: {config.paths.plots_dir}")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 65)
    logger.info("  PHASE 1 COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Total images   : {stats.total_images}")
    logger.info(f"  Classes        : {config.classes.names}")
    logger.info(f"  Train samples  : {manifests[0].num_samples}")
    logger.info(f"  Val samples    : {manifests[1].num_samples}")
    logger.info(f"  Test samples   : {manifests[2].num_samples}")
    logger.info(f"  Corrupted      : {stats.total_corrupted}")
    logger.info(f"  Output dir     : {config.paths.outputs_dir}")
    logger.info(f"  Elapsed time   : {elapsed:.1f}s")
    logger.info("=" * 65)
    logger.info("  Next -> Phase 2: EfficientNetB3 Training Pipeline")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
