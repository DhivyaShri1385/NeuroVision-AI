"""
Two-Stage Model Trainer — Phase 2

Stage 1: Frozen backbone — train only the custom head.
  - Fast convergence (few epochs)
  - High LR safe because backbone is frozen
  - Head learns to use backbone features

Stage 2: Partial fine-tuning — unfreeze top N backbone layers.
  - Very low LR to avoid destroying ImageNet features
  - Backbone fine-tunes to MRI domain
  - BN layers always kept frozen

This two-stage strategy is standard practice in medical imaging transfer
learning because:
  1. Random head weights → large gradients → if backbone is unfrozen
     immediately, those gradients corrupt the pretrained weights.
  2. Fine-tuning from a stable checkpoint (end of Stage 1) leads to better
     final accuracy than training end-to-end from random initialization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.models.classifier import build_classifier, unfreeze_top_layers
from src.training.callbacks import get_stage1_callbacks, get_stage2_callbacks
from src.utils.config import AppConfig
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Complete record of one model's training run."""
    backbone_name: str
    stage1_history: dict = field(default_factory=dict)
    stage2_history: dict = field(default_factory=dict)
    best_val_accuracy: float = 0.0
    best_val_loss: float = float("inf")
    total_epochs_run: int = 0
    training_time_seconds: float = 0.0
    model_path: Path | None = None

    @property
    def combined_history(self) -> dict:
        """Merge stage 1 + stage 2 histories for plotting."""
        combined = {}
        for key in self.stage1_history:
            combined[key] = (
                self.stage1_history.get(key, []) +
                self.stage2_history.get(key, [])
            )
        return combined


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Orchestrates two-stage training for a single backbone model."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.models_dir = config.paths.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._setup_mixed_precision()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        backbone_name: str,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weights: dict[int, float] | None = None,
        n_unfreeze_layers: int = 30,
    ) -> TrainingResult:
        """Run the complete two-stage training pipeline.

        Args:
            backbone_name: 'EfficientNetB3', 'ResNet50', or 'DenseNet121'.
            train_ds: Training tf.data.Dataset (batched, prefetched).
            val_ds:   Validation tf.data.Dataset (batched, prefetched).
            class_weights: Optional dict {class_idx: weight} from DataLoader.
            n_unfreeze_layers: How many backbone layers to unfreeze in Stage 2.

        Returns:
            TrainingResult with full history and best metrics.
        """
        logger.info("=" * 60)
        logger.info(f"TRAINING: {backbone_name}")
        logger.info("=" * 60)
        t_start = time.time()

        # Build model
        model = build_classifier(backbone_name, self.config)
        result = TrainingResult(backbone_name=backbone_name)

        # Stage 1: Frozen backbone
        logger.info("\n[STAGE 1] Frozen backbone — training head only")
        s1_history = self._stage1_frozen(model, train_ds, val_ds, class_weights)
        result.stage1_history = s1_history.history

        # Stage 2: Fine-tune top layers
        logger.info("\n[STAGE 2] Fine-tuning top layers")
        s2_history = self._stage2_finetune(
            model, train_ds, val_ds, class_weights, n_unfreeze_layers
        )
        result.stage2_history = s2_history.history

        # Collect final metrics
        all_val_acc = (
            result.stage1_history.get("val_accuracy", []) +
            result.stage2_history.get("val_accuracy", [])
        )
        all_val_loss = (
            result.stage1_history.get("val_loss", []) +
            result.stage2_history.get("val_loss", [])
        )
        result.best_val_accuracy = max(all_val_acc) if all_val_acc else 0.0
        result.best_val_loss = min(all_val_loss) if all_val_loss else float("inf")
        result.total_epochs_run = len(all_val_acc)
        result.training_time_seconds = time.time() - t_start
        result.model_path = self.models_dir / f"{backbone_name}_best.keras"

        logger.info("\n" + "=" * 60)
        logger.info(f"TRAINING COMPLETE: {backbone_name}")
        logger.info(f"  Best val accuracy : {result.best_val_accuracy:.4f}")
        logger.info(f"  Best val loss     : {result.best_val_loss:.4f}")
        logger.info(f"  Total epochs      : {result.total_epochs_run}")
        logger.info(f"  Time              : {result.training_time_seconds / 60:.1f} min")
        logger.info(f"  Saved to          : {result.model_path}")
        logger.info("=" * 60)
        return result

    # ------------------------------------------------------------------
    # Private: Stage 1
    # ------------------------------------------------------------------

    def _stage1_frozen(
        self,
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weights: dict | None,
    ) -> tf.keras.callbacks.History:
        """Compile and fit with frozen backbone."""
        cfg = self.config.training

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=cfg.initial_learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=cfg.label_smoothing
            ),
            metrics=self._get_metrics(),
        )

        callbacks = get_stage1_callbacks(
            self.config, model.name, self.models_dir
        )

        history = model.fit(
            train_ds,
            epochs=cfg.freeze_base_epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
            # Note: class_weight not passed — dataset is balanced (equal class counts).
            # Keras 3 also has a known issue with class_weight + one-hot labels.
        )
        return history

    # ------------------------------------------------------------------
    # Private: Stage 2
    # ------------------------------------------------------------------

    def _stage2_finetune(
        self,
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weights: dict | None,
        n_unfreeze_layers: int,
    ) -> tf.keras.callbacks.History:
        """Unfreeze top layers, re-compile at lower LR, and fine-tune."""
        cfg = self.config.training

        unfreeze_top_layers(model, n_layers=n_unfreeze_layers)

        # Re-compile with a much lower LR to avoid catastrophic forgetting
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=cfg.fine_tune_learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=cfg.label_smoothing
            ),
            metrics=self._get_metrics(),
        )

        callbacks = get_stage2_callbacks(
            self.config, model.name, self.models_dir
        )

        # Remaining epochs after stage 1
        remaining_epochs = cfg.epochs - cfg.freeze_base_epochs

        history = model.fit(
            train_ds,
            epochs=remaining_epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    def _get_metrics(self) -> list:
        """Metrics tracked during training.

        Keeping only CategoricalAccuracy + TopK during training for Keras 3
        compatibility with one-hot labels.  Precision / Recall / F1 / AUC are
        computed post-training by ClassificationEvaluator using scikit-learn,
        which is more reliable and gives per-class breakdowns.
        """
        return [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ]

    def _setup_mixed_precision(self) -> None:
        """Enable float16 mixed precision only if GPU is available.

        On CPU, mixed precision provides no speedup and can cause NaN
        gradients on some operations. We disable it automatically.
        """
        gpus = tf.config.list_physical_devices("GPU")
        if gpus and self.config.training.mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision enabled (float16/float32)")
        else:
            tf.keras.mixed_precision.set_global_policy("float32")
            if not gpus:
                logger.warning(
                    "No GPU detected — running on CPU. "
                    "Training will be slow. Consider using Google Colab (GPU runtime). "
                    "Mixed precision disabled."
                )
