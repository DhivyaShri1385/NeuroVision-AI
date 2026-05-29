"""
run_phase2.py — Phase 2 Orchestration Script

Runs the complete training and evaluation pipeline:

  1. Load config + dataset manifests
  2. Build tf.data pipelines
  3. Train selected backbone(s) — 2-stage transfer learning
  4. Evaluate on test set — full metrics suite
  5. Generate evaluation plots
  6. (Optional) Train ensemble members and compare

Usage:
    # Train EfficientNetB3 (default)
    python scripts/run_phase2.py

    # Train a specific backbone
    python scripts/run_phase2.py --backbone ResNet50

    # Train all 3 backbones for ensemble
    python scripts/run_phase2.py --ensemble

    # Quick smoke-test (2 epochs)
    python scripts/run_phase2.py --quick-test

    # Evaluate existing saved model (skip training)
    python scripts/run_phase2.py --eval-only --backbone EfficientNetB3

Colab GPU tip:
    On CPU this will be slow (~2-3 min/epoch for EfficientNetB3).
    For full training, upload the project to Google Colab and run:
        !python scripts/run_phase2.py --backbone EfficientNetB3
    with a GPU T4/A100 runtime for 10x speed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.utils.logger import logger, setup_logger
from src.utils.config import load_config
from src.data.data_loader import BrainTumorDataLoader
from src.models.classifier import ModelFactory
from src.models.ensemble import EnsemblePredictor
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import ClassificationEvaluator
from src.evaluation.visualizer import EvaluationVisualizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroVision AI — Phase 2: Model Training & Evaluation"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="EfficientNetB3",
        choices=["EfficientNetB3", "ResNet50", "DenseNet121"],
        help="Backbone to train (default: EfficientNetB3)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help="Train all 3 backbones for ensemble comparison",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Skip training; evaluate existing saved model",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        default=False,
        help="Run only 2 epochs for smoke-testing (no real training)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=str(project_root / ".env"),
        help="Path to .env file",
    )
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=30,
        help="Number of backbone layers to unfreeze in Stage 2 (default: 30)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: override epochs for quick test
# ---------------------------------------------------------------------------

def patch_config_for_quick_test(config):
    """Monkeypatch config training epochs to 2 for smoke-testing."""
    import dataclasses
    patched_training = dataclasses.replace(
        config.training,
        epochs=4,
        freeze_base_epochs=2,
        early_stopping_patience=2,
        reduce_lr_patience=1,
    )
    return dataclasses.replace(config, training=patched_training)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def train_and_evaluate(
    backbone_name: str,
    config,
    loader: BrainTumorDataLoader,
    trainer: ModelTrainer,
    evaluator: ClassificationEvaluator,
    visualizer: EvaluationVisualizer,
    args: argparse.Namespace,
):
    """Train one backbone, evaluate, and generate all plots."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  BACKBONE: {backbone_name}")
    logger.info(f"{'='*60}")

    # Build datasets
    train_ds = loader.get_train_dataset_from_dir()
    val_ds   = loader.get_val_dataset_from_dir()
    test_ds  = loader.get_test_dataset_from_dir()

    class_weights = loader.get_class_weights()

    # Check if a saved model already exists (for --eval-only)
    model_path = config.paths.models_dir / f"{backbone_name}_best.keras"

    if args.eval_only:
        if not model_path.exists():
            logger.error(
                f"No saved model found at {model_path}. "
                "Run without --eval-only first."
            )
            return None, None

        import tensorflow as tf
        logger.info(f"Loading saved model: {model_path}")
        model = tf.keras.models.load_model(str(model_path))
        training_result = None
    else:
        # Full two-stage training
        training_result = trainer.train(
            backbone_name=backbone_name,
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            n_unfreeze_layers=args.unfreeze_layers,
        )

        # Load the best checkpoint for evaluation
        import tensorflow as tf
        if model_path.exists():
            model = tf.keras.models.load_model(str(model_path))
            logger.info(f"Loaded best checkpoint: {model_path}")
        else:
            logger.warning("Best checkpoint not saved — using in-memory model")
            model = None

    # Evaluate
    if model is not None:
        eval_result = evaluator.evaluate_model(model, test_ds, backbone_name)

        # Save metrics JSON
        metrics_path = config.paths.outputs_dir / f"metrics_{backbone_name}.json"
        with open(metrics_path, "w") as f:
            json.dump(eval_result.summary_dict(), f, indent=2)
        logger.info(f"Metrics saved: {metrics_path}")

        # Generate plots
        logger.info("Generating evaluation plots...")
        if training_result:
            plots = visualizer.run_all(training_result, eval_result)
        else:
            plots = [
                visualizer.plot_confusion_matrix(eval_result),
                visualizer.plot_roc_curves(eval_result),
                visualizer.plot_per_class_metrics(eval_result),
            ]
        logger.info(f"{len(plots)} plots saved to {config.paths.plots_dir}")
        return training_result, eval_result

    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    load_dotenv(args.env)

    # Setup
    config = load_config()
    setup_logger(log_dir=config.paths.logs_dir, level="INFO")

    logger.info("=" * 65)
    logger.info("  NeuroVision AI — Phase 2: Model Training & Evaluation")
    logger.info("=" * 65)

    if args.quick_test:
        logger.warning(
            "QUICK-TEST mode: only 4 epochs, results not meaningful. "
            "Use for smoke-testing the pipeline only."
        )
        config = patch_config_for_quick_test(config)

    # Check config has freeze_base_epochs
    if not hasattr(config.training, "freeze_base_epochs"):
        logger.error(
            "config.training.freeze_base_epochs not found. "
            "Add 'freeze_base_epochs: 10' under 'training:' in config.yaml"
        )
        sys.exit(1)

    # Build components
    loader     = BrainTumorDataLoader(config)
    trainer    = ModelTrainer(config)
    evaluator  = ClassificationEvaluator(config)
    visualizer = EvaluationVisualizer(config)

    t_start = time.time()

    # Determine backbones to train
    if args.ensemble:
        backbones = ["EfficientNetB3", "ResNet50", "DenseNet121"]
        logger.info(f"Ensemble mode: training {backbones}")
    else:
        backbones = [args.backbone]

    # Train + evaluate each backbone
    all_eval_results = []
    for backbone in backbones:
        _, eval_result = train_and_evaluate(
            backbone_name=backbone,
            config=config,
            loader=loader,
            trainer=trainer,
            evaluator=evaluator,
            visualizer=visualizer,
            args=args,
        )
        if eval_result:
            all_eval_results.append(eval_result)

    # Multi-model comparison plot
    if len(all_eval_results) > 1:
        logger.info("Generating ensemble comparison plot...")
        evaluator.compare_models(all_eval_results)
        visualizer.plot_model_comparison(all_eval_results)

    # Ensemble inference (if all 3 models trained)
    ensemble_model_paths = {
        b: config.paths.models_dir / f"{b}_best.keras"
        for b in ["EfficientNetB3", "ResNet50", "DenseNet121"]
        if (config.paths.models_dir / f"{b}_best.keras").exists()
    }
    if len(ensemble_model_paths) >= 2:
        logger.info(f"\nRunning ensemble inference with: {list(ensemble_model_paths.keys())}")
        import tensorflow as tf
        import numpy as np

        predictor = EnsemblePredictor(config)
        predictor.load_models(ensemble_model_paths)

        # Collect test set
        test_ds = loader.get_test_dataset_from_dir()
        all_true, all_proba = [], []
        for images, labels in test_ds:
            result = predictor.predict(images.numpy())
            all_proba.append(result.probabilities)
            all_true.append(np.argmax(labels.numpy(), axis=1))

        y_true  = np.concatenate(all_true,  axis=0)
        y_proba = np.concatenate(all_proba, axis=0)

        ensemble_eval = evaluator.compute_metrics(y_true, y_proba, "Ensemble")
        all_eval_results.append(ensemble_eval)

        # Save ensemble metrics
        metrics_path = config.paths.outputs_dir / "metrics_Ensemble.json"
        with open(metrics_path, "w") as f:
            json.dump(ensemble_eval.summary_dict(), f, indent=2)

        # Ensemble plots
        visualizer.plot_confusion_matrix(ensemble_eval)
        visualizer.plot_roc_curves(ensemble_eval)
        visualizer.plot_per_class_metrics(ensemble_eval)
        if len(all_eval_results) > 1:
            visualizer.plot_model_comparison(all_eval_results)

    # Final summary
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 65)
    logger.info("  PHASE 2 COMPLETE")
    logger.info("=" * 65)
    for r in all_eval_results:
        logger.info(
            f"  {r.backbone_name:<20} "
            f"Acc={r.accuracy:.4f}  "
            f"F1={r.macro_f1:.4f}  "
            f"AUC={r.roc_auc_macro:.4f}"
        )
    logger.info(f"  Total time : {elapsed / 60:.1f} min")
    logger.info(f"  Models dir : {config.paths.models_dir}")
    logger.info(f"  Plots dir  : {config.paths.plots_dir}")
    logger.info("=" * 65)
    logger.info("  Next -> Phase 3: Attention U-Net Segmentation")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
