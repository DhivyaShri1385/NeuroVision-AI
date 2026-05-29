"""
Custom Training Callbacks — Phase 2

Provides:
  - WarmupLearningRateScheduler  : Linear warmup for N epochs then steady
  - get_stage1_callbacks()       : Callbacks for frozen-backbone phase
  - get_stage2_callbacks()       : Callbacks for fine-tuning phase

Why warmup?
  On the first few epochs the random head weights produce large gradients.
  Without warmup, these can corrupt the pretrained backbone weights even
  when frozen. Warmup ramps the LR from near-zero to the target value over
  5 epochs, stabilizing the gradient signal.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.utils.config import AppConfig
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Warmup LR Scheduler
# ---------------------------------------------------------------------------

class WarmupLearningRateScheduler(tf.keras.callbacks.Callback):
    """Linear warmup from start_lr to target_lr over warmup_epochs.

    After warmup, learning rate stays at target_lr (ReduceLROnPlateau
    handles further decay).
    """

    def __init__(
        self,
        target_lr: float,
        warmup_epochs: int = 5,
        start_lr_fraction: float = 0.1,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
        self.start_lr = target_lr * start_lr_fraction
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        if epoch < self.warmup_epochs:
            # Linear interpolation from start_lr to target_lr
            lr = self.start_lr + (self.target_lr - self.start_lr) * (epoch / self.warmup_epochs)
            # Keras 3 compatible: assign directly instead of backend.set_value
            self.model.optimizer.learning_rate = lr
            if self.verbose:
                logger.debug(f"Warmup epoch {epoch + 1}/{self.warmup_epochs} | LR = {lr:.2e}")

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        # Keras 3 compatible: read lr as float directly
        current_lr = float(self.model.optimizer.learning_rate)
        if logs is not None:
            logs["lr"] = current_lr


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------

def get_stage1_callbacks(
    config: AppConfig,
    model_name: str,
    checkpoint_dir: Path,
) -> list[tf.keras.callbacks.Callback]:
    """Return callbacks for Stage 1 (frozen backbone training).

    Stage 1 is short (10 epochs) — we use early stopping with low patience
    to avoid wasting time if the head converges quickly.

    Args:
        config: AppConfig.
        model_name: Backbone name, used for checkpoint filename.
        checkpoint_dir: Where to save the best model.

    Returns:
        List of compiled Keras callbacks.
    """
    cfg = config.training
    checkpoint_path = checkpoint_dir / f"{model_name}_stage1_best.keras"

    callbacks = [
        # Warmup LR
        WarmupLearningRateScheduler(
            target_lr=cfg.initial_learning_rate,
            warmup_epochs=cfg.warmup_epochs,
        ),

        # Save best model by val_accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        ),

        # Early stopping — patience=5 for stage 1 (short phase)
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),

        # Reduce LR on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
            mode="min",
        ),

        # CSV logger — all metrics per epoch
        tf.keras.callbacks.CSVLogger(
            str(checkpoint_dir / f"{model_name}_stage1_history.csv"),
            append=False,
        ),
    ]

    logger.info(
        f"Stage 1 callbacks initialized for {model_name} | "
        f"checkpoint: {checkpoint_path}"
    )
    return callbacks


def get_stage2_callbacks(
    config: AppConfig,
    model_name: str,
    checkpoint_dir: Path,
) -> list[tf.keras.callbacks.Callback]:
    """Return callbacks for Stage 2 (fine-tuning).

    Stage 2 uses the full patience settings from config and saves the
    final best model (this is the production checkpoint).

    Args:
        config: AppConfig.
        model_name: Backbone name.
        checkpoint_dir: Where to save the best model.

    Returns:
        List of compiled Keras callbacks.
    """
    cfg = config.training
    # Stage 2 checkpoint is the production model
    checkpoint_path = checkpoint_dir / f"{model_name}_best.keras"

    callbacks = [
        # Save best by val_accuracy — this is the final production model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        ),

        # Full early stopping patience
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),

        # Reduce LR on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
            mode="min",
        ),

        # CSV logger — stage 2 history
        tf.keras.callbacks.CSVLogger(
            str(checkpoint_dir / f"{model_name}_stage2_history.csv"),
            append=False,
        ),

        # TensorBoard (optional — runs on localhost:6006)
        tf.keras.callbacks.TensorBoard(
            log_dir=str(checkpoint_dir / "tensorboard" / model_name),
            histogram_freq=0,
            write_graph=False,
            update_freq="epoch",
        ),
    ]

    logger.info(
        f"Stage 2 callbacks initialized for {model_name} | "
        f"production checkpoint: {checkpoint_path}"
    )
    return callbacks
