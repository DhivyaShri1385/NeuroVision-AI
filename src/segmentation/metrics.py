"""
Segmentation Metrics — Phase 3

Keras-compatible metric classes and standalone NumPy evaluation functions.

Keras metrics (for model.compile / training loop):
  - DiceCoefficient   : 2|X∩Y| / (|X|+|Y|)
  - IoUScore          : |X∩Y| / |X∪Y|  (Jaccard index)
  - PixelAccuracy     : correct pixels / total pixels

NumPy evaluation (for post-training analysis):
  - evaluate_masks()  : returns dict with dice, iou, pixel_acc, precision, recall

All Keras metrics inherit from tf.keras.metrics.Metric and are stateful
(accumulate over batches within an epoch, reset between epochs).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMOOTH = 1.0


def _binarize(y_pred: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    return tf.cast(y_pred >= threshold, tf.float32)


# ---------------------------------------------------------------------------
# Keras Metric: Dice Coefficient
# ---------------------------------------------------------------------------

class DiceCoefficient(tf.keras.metrics.Metric):
    """Soft Dice coefficient accumulated over batches.

    Soft version (uses probabilities, not hard predictions) is used during
    training so gradients flow; hard threshold applied in evaluation.
    """

    def __init__(self, threshold: float = 0.5, name: str = "dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self._sum   = self.add_weight(name="sum",   initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight=None,
    ) -> None:
        y_true = tf.cast(tf.reshape(y_true, [tf.shape(y_true)[0], -1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), tf.float32)
        y_pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=1)
        dsc = (2.0 * intersection + _SMOOTH) / (
            tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred_bin, axis=1) + _SMOOTH
        )
        self._sum.assign_add(tf.reduce_sum(dsc))
        self._count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self._sum, self._count)

    def reset_state(self) -> None:
        self._sum.assign(0.0)
        self._count.assign(0.0)

    def get_config(self) -> dict:
        return {"threshold": self.threshold, **super().get_config()}


# ---------------------------------------------------------------------------
# Keras Metric: IoU / Jaccard
# ---------------------------------------------------------------------------

class IoUScore(tf.keras.metrics.Metric):
    """Intersection over Union (Jaccard index), accumulated over batches."""

    def __init__(self, threshold: float = 0.5, name: str = "iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self._sum   = self.add_weight(name="sum",   initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight=None,
    ) -> None:
        y_true = tf.cast(tf.reshape(y_true, [tf.shape(y_true)[0], -1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]), tf.float32)
        y_pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=1)
        union        = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred_bin, axis=1) - intersection
        iou = (intersection + _SMOOTH) / (union + _SMOOTH)

        self._sum.assign_add(tf.reduce_sum(iou))
        self._count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self._sum, self._count)

    def reset_state(self) -> None:
        self._sum.assign(0.0)
        self._count.assign(0.0)

    def get_config(self) -> dict:
        return {"threshold": self.threshold, **super().get_config()}


# ---------------------------------------------------------------------------
# Keras Metric: Pixel Accuracy
# ---------------------------------------------------------------------------

class PixelAccuracy(tf.keras.metrics.Metric):
    """Fraction of correctly classified pixels, accumulated over batches."""

    def __init__(self, threshold: float = 0.5, name: str = "pixel_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self._correct = self.add_weight(name="correct", initializer="zeros")
        self._total   = self.add_weight(name="total",   initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight=None,
    ) -> None:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        total   = tf.cast(tf.size(y_true), tf.float32)
        self._correct.assign_add(correct)
        self._total.assign_add(total)

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self._correct, self._total)

    def reset_state(self) -> None:
        self._correct.assign(0.0)
        self._total.assign(0.0)

    def get_config(self) -> dict:
        return {"threshold": self.threshold, **super().get_config()}


# ---------------------------------------------------------------------------
# NumPy evaluation helper
# ---------------------------------------------------------------------------

def evaluate_masks(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute segmentation metrics on NumPy arrays.

    Args:
        y_true:    Ground-truth masks, shape (N, H, W) or (N, H, W, 1), binary float.
        y_pred:    Predicted probability maps, same shape.
        threshold: Binarisation threshold.

    Returns:
        Dict with keys: dice, iou, pixel_acc, precision, recall, f1.
    """
    y_true = y_true.squeeze().astype(np.float32)
    y_pred = (y_pred.squeeze() >= threshold).astype(np.float32)

    # Flatten spatial dims: (N, H*W)
    n = y_true.shape[0]
    yt = y_true.reshape(n, -1)
    yp = y_pred.reshape(n, -1)

    tp = np.sum(yt * yp, axis=1)
    fp = np.sum((1 - yt) * yp, axis=1)
    fn = np.sum(yt * (1 - yp), axis=1)
    tn = np.sum((1 - yt) * (1 - yp), axis=1)

    dice     = np.mean((2 * tp + _SMOOTH) / (2 * tp + fp + fn + _SMOOTH))
    iou      = np.mean((tp + _SMOOTH) / (tp + fp + fn + _SMOOTH))
    prec     = np.mean((tp + _SMOOTH) / (tp + fp + _SMOOTH))
    recall   = np.mean((tp + _SMOOTH) / (tp + fn + _SMOOTH))
    f1       = 2 * prec * recall / (prec + recall + 1e-8)
    pixel_acc = np.mean((tp + tn) / (tp + fp + fn + tn + _SMOOTH))

    return {
        "dice":       float(dice),
        "iou":        float(iou),
        "precision":  float(prec),
        "recall":     float(recall),
        "f1":         float(f1),
        "pixel_acc":  float(pixel_acc),
    }


# ---------------------------------------------------------------------------
# Convenience getter
# ---------------------------------------------------------------------------

def get_metrics(threshold: float = 0.5) -> list:
    """Return list of Keras metric instances for model.compile."""
    return [
        DiceCoefficient(threshold=threshold, name="dice"),
        IoUScore(threshold=threshold,        name="iou"),
        PixelAccuracy(threshold=threshold,   name="pixel_acc"),
    ]
