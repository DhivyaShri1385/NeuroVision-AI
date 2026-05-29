"""
Segmentation Loss Functions — Phase 3

Implements:
  - dice_loss           : 1 - Dice coefficient  (smooth=1 to avoid / 0)
  - bce_dice_loss       : Binary cross-entropy + Dice  (most practical default)
  - focal_dice_loss     : Focal + Dice  (handles severe class imbalance)
  - tversky_loss        : Tversky index loss  (alpha/beta control FP vs FN weight)

All losses operate on float32 logits/predictions and float32 binary masks.
All are returned as scalar tensors compatible with model.compile(loss=...).

Usage:
    from src.segmentation.losses import get_loss
    loss_fn = get_loss("bce_dice")
    model.compile(optimizer=..., loss=loss_fn, metrics=[...])
"""

from __future__ import annotations

import tensorflow as tf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMOOTH = 1.0   # Laplace smoothing to avoid division by zero


def _flatten(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Flatten spatial dims to (batch, N) for loss computation."""
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [tf.shape(y_pred)[0], -1])
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Soft Dice loss = 1 - DSC.

    DSC = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)

    Args:
        y_true: Ground-truth binary mask, shape (B, H, W, 1), float32.
        y_pred: Predicted probability map, shape (B, H, W, 1), float32.

    Returns:
        Scalar loss (mean over batch).
    """
    y_true, y_pred = _flatten(y_true, y_pred)
    intersection = tf.reduce_sum(y_true * y_pred, axis=1)
    dsc = (2.0 * intersection + _SMOOTH) / (
        tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1) + _SMOOTH
    )
    return tf.reduce_mean(1.0 - dsc)


# ---------------------------------------------------------------------------
# BCE + Dice Loss
# ---------------------------------------------------------------------------

def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Weighted sum: BCE + Dice.

    BCE handles per-pixel supervision; Dice handles region-level overlap.
    Default weights: 0.5 * BCE + 0.5 * Dice.

    Args:
        y_true: (B, H, W, 1) float32 binary mask.
        y_pred: (B, H, W, 1) float32 probability map.

    Returns:
        Scalar combined loss.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    dl  = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dl


# ---------------------------------------------------------------------------
# Focal Loss (binary)
# ---------------------------------------------------------------------------

def _focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> tf.Tensor:
    """Binary focal loss — down-weights easy examples."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Clip for numerical stability
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pt     = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
    at     = tf.where(tf.equal(y_true, 1.0), alpha,  1.0 - alpha)
    fl     = -at * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
    return tf.reduce_mean(fl)


def focal_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """0.5 * Focal + 0.5 * Dice.

    Good for highly imbalanced foreground (small tumors).
    """
    return 0.5 * _focal_loss(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)


# ---------------------------------------------------------------------------
# Tversky Loss
# ---------------------------------------------------------------------------

def make_tversky_loss(alpha: float = 0.3, beta: float = 0.7):
    """Factory that returns a Tversky loss function with fixed alpha/beta.

    Tversky index = TP / (TP + alpha*FP + beta*FN)
    * alpha < beta  → penalises false negatives more → higher recall  → catches more tumors.
    * alpha=0.5, beta=0.5 → reduces to Dice.

    Args:
        alpha: Weight for false positives (default 0.3).
        beta:  Weight for false negatives (default 0.7).

    Returns:
        Callable loss function compatible with model.compile(loss=...).
    """
    def tversky_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true, y_pred = _flatten(y_true, y_pred)
        tp = tf.reduce_sum(y_true * y_pred, axis=1)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred, axis=1)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred), axis=1)
        tversky_idx = (tp + _SMOOTH) / (tp + alpha * fp + beta * fn + _SMOOTH)
        return tf.reduce_mean(1.0 - tversky_idx)

    tversky_loss.__name__ = f"tversky_a{alpha}_b{beta}"
    return tversky_loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_loss(name: str, tversky_alpha: float = 0.3, tversky_beta: float = 0.7):
    """Return the loss function for the given name string.

    Args:
        name:          One of: "dice", "bce_dice", "focal_dice", "tversky".
        tversky_alpha: Used only when name="tversky".
        tversky_beta:  Used only when name="tversky".

    Returns:
        A callable loss function.
    """
    _registry = {
        "dice":        dice_loss,
        "bce_dice":    bce_dice_loss,
        "focal_dice":  focal_dice_loss,
    }
    if name in _registry:
        return _registry[name]
    if name == "tversky":
        return make_tversky_loss(tversky_alpha, tversky_beta)
    raise ValueError(
        f"Unknown loss: '{name}'. Valid options: {list(_registry.keys()) + ['tversky']}"
    )
