"""
Saliency Maps — Phase 4

Implements two pixel-attribution methods:

  VanillaSaliency   — gradient of class score w.r.t. input image pixels.
                      Fast, but noisy (saturated gradients problem).

  SmoothGrad        — average saliency over N copies of the input with
                      independent Gaussian noise added.  Sharper, more
                      interpretable maps.  Smilkov et al. 2017.
                      https://arxiv.org/abs/1706.03825

Both methods return an (H, W) float32 heatmap in [0, 1] that highlights
which pixels most influenced the predicted class.

Usage:
    from src.xai.saliency import VanillaSaliency, SmoothGrad
    vsm  = VanillaSaliency(classifier_model, config)
    hm_v, cls, probs = vsm.compute(img_array)

    sgm  = SmoothGrad(classifier_model, config, n_samples=50, noise_std=0.15)
    hm_s, cls, probs = sgm.compute(img_array)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.utils.config import AppConfig
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseSaliency:
    """Common helper methods for saliency-based explainers."""

    def __init__(self, model: tf.keras.Model, config: AppConfig):
        self.model       = model
        self.config      = config
        self.class_names = config.classes.names

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _predict(self, img: tf.Tensor) -> tuple[int, np.ndarray]:
        """Get predicted class index and probability vector."""
        preds     = self.model(img, training=False)
        class_idx = int(tf.argmax(preds[0]).numpy())
        return class_idx, preds.numpy()[0]

    @staticmethod
    def _prepare(img: np.ndarray) -> tf.Tensor:
        """Ensure (1, H, W, C) float32 tensor."""
        if isinstance(img, tf.Tensor):
            img = img.numpy()
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        return tf.constant(img, dtype=tf.float32)

    @staticmethod
    def _aggregate_channels(grads: np.ndarray) -> np.ndarray:
        """Aggregate (H, W, C) gradient volume to (H, W) by max absolute value."""
        return np.max(np.abs(grads), axis=-1)

    @staticmethod
    def _normalise(heatmap: np.ndarray) -> np.ndarray:
        """Min-max normalise to [0, 1]."""
        mn, mx = heatmap.min(), heatmap.max()
        if mx - mn < 1e-8:
            return np.zeros_like(heatmap, dtype=np.float32)
        return ((heatmap - mn) / (mx - mn)).astype(np.float32)


# ---------------------------------------------------------------------------
# Vanilla Saliency
# ---------------------------------------------------------------------------

class VanillaSaliency(_BaseSaliency):
    """Gradient of class score w.r.t. input pixels.

    Args:
        model:  Compiled or uncompiled classifier tf.keras.Model.
        config: AppConfig.
    """

    def compute(
        self,
        img_array: np.ndarray,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Compute vanilla saliency map.

        Args:
            img_array:  (H, W, C) or (1, H, W, C) float32 image.
            class_idx:  Target class.  None → argmax.

        Returns:
            saliency:  (H, W) float32 in [0, 1].
            class_idx: Target class index used.
            probs:     (num_classes,) softmax probabilities.
        """
        img = self._prepare(img_array)

        with tf.GradientTape() as tape:
            tape.watch(img)
            preds = self.model(img, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]).numpy())
            target_score = preds[:, class_idx]

        grads = tape.gradient(target_score, img)   # (1, H, W, C)
        if grads is None:
            logger.warning("VanillaSaliency: gradient is None — returning zeros")
            h, w = img.shape[1], img.shape[2]
            return np.zeros((h, w), dtype=np.float32), class_idx, preds.numpy()[0]

        saliency = self._aggregate_channels(grads[0].numpy())  # (H, W)
        saliency = self._normalise(saliency)
        return saliency, int(class_idx), preds.numpy()[0]

    def compute_batch(
        self,
        images:     np.ndarray,
        class_idxs: list[int | None] | None = None,
    ) -> list[tuple[np.ndarray, int, np.ndarray]]:
        if class_idxs is None:
            class_idxs = [None] * len(images)
        return [self.compute(images[i], class_idxs[i]) for i in range(len(images))]


# ---------------------------------------------------------------------------
# SmoothGrad
# ---------------------------------------------------------------------------

class SmoothGrad(_BaseSaliency):
    """Average gradient saliency over N noisy copies of the input.

    Smilkov et al. (2017): adds Gaussian noise σ ∝ (x_max - x_min) to the
    input N times and averages the resulting saliency maps.  This reduces
    noise and sharpens the relevant features.

    Args:
        model:       Classifier tf.keras.Model.
        config:      AppConfig.
        n_samples:   Number of noisy copies (default 50).  Higher = smoother
                     but slower.
        noise_std:   Noise standard deviation as fraction of input range
                     (default 0.15 → σ = 0.15 * (max - min)).
    """

    def __init__(
        self,
        model:     tf.keras.Model,
        config:    AppConfig,
        n_samples: int   = 50,
        noise_std: float = 0.15,
    ):
        super().__init__(model, config)
        self.n_samples = n_samples
        self.noise_std = noise_std

    def compute(
        self,
        img_array:  np.ndarray,
        class_idx:  int | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Compute SmoothGrad saliency map.

        Args:
            img_array:  (H, W, C) or (1, H, W, C) float32 image.
            class_idx:  Target class.  None → argmax on clean image.

        Returns:
            saliency:  (H, W) float32 in [0, 1].
            class_idx: Target class used.
            probs:     Probabilities from the CLEAN (un-noised) image.
        """
        img = self._prepare(img_array)

        # Determine class from clean prediction first
        clean_preds = self.model(img, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(clean_preds[0]).numpy())

        # Compute noise scale: σ = noise_std * range(img)
        img_np   = img.numpy()
        sigma    = self.noise_std * float(img_np.max() - img_np.min())
        if sigma < 1e-8:
            sigma = self.noise_std  # fallback for constant images

        # Accumulate gradients over noisy samples
        h, w = img.shape[1], img.shape[2]
        acc_grads = np.zeros((h, w, img.shape[3]), dtype=np.float64)

        for _ in range(self.n_samples):
            noise     = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
            noisy_img = tf.constant(img_np + noise, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(noisy_img)
                preds        = self.model(noisy_img, training=False)
                target_score = preds[:, class_idx]

            grads = tape.gradient(target_score, noisy_img)
            if grads is not None:
                acc_grads += grads[0].numpy()

        # Average over samples
        avg_grads = (acc_grads / self.n_samples).astype(np.float32)
        saliency  = self._aggregate_channels(avg_grads)   # (H, W)
        saliency  = self._normalise(saliency)

        return saliency, int(class_idx), clean_preds.numpy()[0]

    def compute_batch(
        self,
        images:     np.ndarray,
        class_idxs: list[int | None] | None = None,
    ) -> list[tuple[np.ndarray, int, np.ndarray]]:
        if class_idxs is None:
            class_idxs = [None] * len(images)
        return [self.compute(images[i], class_idxs[i]) for i in range(len(images))]


# ---------------------------------------------------------------------------
# Integrated Gradients (optional, lightweight)
# ---------------------------------------------------------------------------

class IntegratedGradients(_BaseSaliency):
    """Integrated Gradients (Sundararajan et al. 2017).

    Integrates gradient along a straight-line path from a baseline
    (black image) to the input.  Satisfies the 'completeness' axiom —
    attribution scores sum to the predicted class score difference from
    the baseline.

    Args:
        model:    Classifier tf.keras.Model.
        config:   AppConfig.
        steps:    Number of integration steps (default 50; more = more accurate).
        baseline: Baseline image value (default 0.0 = black image).
    """

    def __init__(
        self,
        model:    tf.keras.Model,
        config:   AppConfig,
        steps:    int   = 50,
        baseline: float = 0.0,
    ):
        super().__init__(model, config)
        self.steps    = steps
        self.baseline = baseline

    def compute(
        self,
        img_array:  np.ndarray,
        class_idx:  int | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Compute Integrated Gradients attribution map.

        Returns:
            ig_map:    (H, W) float32 in [0, 1] — absolute attribution.
            class_idx: Target class used.
            probs:     Predictions on the ORIGINAL (non-interpolated) image.
        """
        img = self._prepare(img_array)
        img_np = img.numpy()

        # Baseline (black image of same shape)
        baseline_np = np.full_like(img_np, self.baseline, dtype=np.float32)

        # Predict on original image to get class
        preds = self.model(img, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]).numpy())

        # Interpolated inputs: baseline → img  (steps+1 images)
        alphas = np.linspace(0, 1, self.steps + 1, dtype=np.float32)
        interpolated = np.stack([baseline_np[0] + a * (img_np[0] - baseline_np[0])
                                  for a in alphas], axis=0)  # (steps+1, H, W, C)

        # Compute gradients at each interpolation step
        interp_tf = tf.constant(interpolated, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(interp_tf)
            preds_interp = self.model(interp_tf, training=False)
            scores       = preds_interp[:, class_idx]
        grads = tape.gradient(scores, interp_tf)   # (steps+1, H, W, C)

        if grads is None:
            h, w = img.shape[1], img.shape[2]
            return np.zeros((h, w), dtype=np.float32), class_idx, preds.numpy()[0]

        # Trapezoidal integration
        grads_np   = grads.numpy()                                  # (steps+1, H, W, C)
        avg_grads  = (grads_np[:-1] + grads_np[1:]) / 2.0          # (steps, H, W, C)
        avg_grads  = np.mean(avg_grads, axis=0)                     # (H, W, C)

        # Element-wise product with (input - baseline)
        ig_attrs   = (img_np[0] - baseline_np[0]) * avg_grads       # (H, W, C)
        ig_map     = self._aggregate_channels(ig_attrs)              # (H, W)
        ig_map     = self._normalise(ig_map)

        return ig_map, int(class_idx), preds.numpy()[0]
