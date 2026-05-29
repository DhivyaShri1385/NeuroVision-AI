"""
Segmentation Trainer — Phase 3

Handles compilation, callbacks, and training of U-Net / Attention U-Net.

Features:
  - Compiles model with the loss function selected in config (dice/bce_dice/focal_dice/tversky)
  - Adam optimizer with initial LR from config
  - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (best val_dice)
  - Training history returned as dict for downstream visualisation
  - Mixed-precision is intentionally OFF for segmentation (config.training.mixed_precision
    applies only to Phase 2 classification; segmentation is already memory-bound)

Usage:
    trainer = SegmentationTrainer(config, model, train_ds, val_ds)
    history = trainer.train()
    results = trainer.evaluate(test_ds)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.utils.config import AppConfig
from src.utils.logger import logger
from src.segmentation.losses import get_loss
from src.segmentation.metrics import get_metrics, evaluate_masks


class SegmentationTrainer:
    """Compiles and trains a segmentation model.

    Args:
        config:    AppConfig with segmentation sub-config.
        model:     Uncompiled tf.keras.Model (UNet or AttentionUNet).
        train_ds:  tf.data.Dataset yielding (image, mask) pairs, batched.
        val_ds:    Validation tf.data.Dataset.
        model_dir: Where to save checkpoints.  Defaults to config.paths.models_dir.
    """

    def __init__(
        self,
        config: AppConfig,
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        val_ds:   tf.data.Dataset,
        model_dir: Path | str | None = None,
    ):
        self.config   = config
        self.model    = model
        self.train_ds = train_ds
        self.val_ds   = val_ds

        seg = config.segmentation
        self.epochs      = seg.epochs
        self.lr          = seg.learning_rate
        self.min_lr      = seg.min_lr
        self.patience_es = seg.early_stopping_patience
        self.patience_lr = seg.reduce_lr_patience
        self.lr_factor   = seg.reduce_lr_factor
        self.loss_name   = seg.loss
        self.arch_name   = seg.architecture

        self.model_dir = Path(model_dir) if model_dir else config.paths.models_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._compile()

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def _compile(self) -> None:
        """Compile model with Adam + selected loss + segmentation metrics."""
        loss_fn = get_loss(
            self.loss_name,
            tversky_alpha=self.config.segmentation.tversky_alpha,
            tversky_beta=self.config.segmentation.tversky_beta,
        )
        metrics = get_metrics(threshold=0.5)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=loss_fn,
            metrics=metrics,
        )
        logger.info(
            f"Compiled {self.arch_name} | loss={self.loss_name} | lr={self.lr}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _build_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        ckpt_path = str(
            self.model_dir / f"{self.arch_name.lower()}_best.keras"
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_dice",
                mode="max",
                patience=self.patience_es,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_dice",
                mode="max",
                factor=self.lr_factor,
                patience=self.patience_lr,
                min_lr=self.min_lr,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor="val_dice",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            _LRLogger(),
        ]
        logger.info(f"Callbacks built | checkpoint -> {ckpt_path}")
        return callbacks

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run training loop.

        Returns:
            History dict with keys matching metric names.
        """
        callbacks = self._build_callbacks()
        logger.info(
            f"Starting {self.arch_name} training | "
            f"epochs={self.epochs} | loss={self.loss_name}"
        )

        history = self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            validation_data=self.val_ds,
            callbacks=callbacks,
            verbose=1,
        )

        best_dice = max(history.history.get("val_dice", [0.0]))
        best_iou  = max(history.history.get("val_iou",  [0.0]))
        logger.info(
            f"{self.arch_name} training complete | "
            f"best val_dice={best_dice:.4f} | best val_iou={best_iou:.4f}"
        )
        return history.history

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_ds: tf.data.Dataset,
    ) -> dict[str, float]:
        """Run evaluation on a test dataset.

        Collects all predictions, computes comprehensive metrics via NumPy.

        Args:
            test_ds: tf.data.Dataset yielding (image, mask) pairs, batched.

        Returns:
            Metrics dict: dice, iou, pixel_acc, precision, recall, f1.
        """
        logger.info(f"Evaluating {self.arch_name} on test set ...")
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []

        for imgs, masks in test_ds:
            preds = self.model(imgs, training=False).numpy()
            y_true_all.append(masks.numpy())
            y_pred_all.append(preds)

        y_true = np.concatenate(y_true_all, axis=0)
        y_pred = np.concatenate(y_pred_all, axis=0)

        results = evaluate_masks(y_true, y_pred, threshold=0.5)

        logger.info(
            f"Test results | "
            f"Dice={results['dice']:.4f} | "
            f"IoU={results['iou']:.4f} | "
            f"PixAcc={results['pixel_acc']:.4f} | "
            f"Precision={results['precision']:.4f} | "
            f"Recall={results['recall']:.4f}"
        )
        return results

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_final_model(self) -> Path:
        """Save the final (post-training) model weights."""
        out = self.model_dir / f"{self.arch_name.lower()}_final.keras"
        self.model.save(str(out))
        logger.info(f"Saved final model -> {out}")
        return out


# ---------------------------------------------------------------------------
# Helper callback: log LR each epoch
# ---------------------------------------------------------------------------

class _LRLogger(tf.keras.callbacks.Callback):
    """Log current learning rate to the loguru logger each epoch."""

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        lr = float(self.model.optimizer.learning_rate)
        logger.debug(f"Epoch {epoch + 1} | lr={lr:.2e}")
        if logs is not None:
            logs["lr"] = lr
