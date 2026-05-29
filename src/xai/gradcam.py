"""
Grad-CAM — Phase 4

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
https://arxiv.org/abs/1610.02391

Algorithm:
  1. Forward pass through a two-output model that returns
     (last_conv_activations, final_class_probabilities).
  2. Compute gradient of the target class score w.r.t. the conv activations.
  3. Global-average-pool the gradients over H×W → importance weights α_k.
  4. Weighted sum of feature maps → linear combination L.
  5. ReLU(L) and resize to input resolution → heatmap ∈ [0, 1].

Key design decisions:
  - GradCAM watches the INPUT tensor (tape.watch(img)) so that every
    intermediate tensor — including conv_out — is in the tape's recorded
    computation, enabling gradient computation w.r.t. conv_out.
  - The two-output "grad model" is built once at __init__ time; computation
    per image is cheap (one forward + one backward pass).
  - If the target layer cannot be found (e.g. wrong name, Keras 3 graph
    tracing issue), a fallback "input saliency" heatmap is returned.

Usage:
    from src.xai.gradcam import GradCAM
    gcam = GradCAM(classifier_model, config)
    heatmap, pred_class, probs = gcam.compute(img_array)
    # heatmap: np.ndarray (H, W) float32 in [0, 1]
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from src.utils.config import AppConfig
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Default last-conv layer names per backbone
# ---------------------------------------------------------------------------
_LAST_CONV_DEFAULTS: dict[str, str] = {
    "EfficientNetB0":  "top_conv",
    "EfficientNetB3":  "top_conv",
    "EfficientNetB4":  "top_conv",
    "EfficientNetB7":  "top_conv",
    "ResNet50":        "conv5_block3_3_conv",
    "ResNet101":       "conv5_block3_3_conv",
    "DenseNet121":     "conv5_block16_1_conv",
    "DenseNet201":     "conv5_block32_1_conv",
}


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Computes Grad-CAM heatmaps for a classification model.

    Args:
        classifier: A compiled or uncompiled tf.keras.Model whose structure is:
                    Input → backbone (sub-model) → head layers → softmax output.
                    (This matches the build_classifier() output from Phase 2.)
        config:     AppConfig — used for gradcam_layer name and class names.
    """

    def __init__(self, classifier: Model, config: AppConfig):
        self.model      = classifier
        self.config     = config
        self.class_names = config.classes.names

        # Resolve the target layer name
        xai_cfg         = getattr(config, "xai", None)
        layer_name_hint = getattr(xai_cfg, "gradcam_layer", None) if xai_cfg else None
        self.layer_name = self._resolve_layer_name(layer_name_hint)

        # Build the two-output gradient model
        self.grad_model = self._build_grad_model()

    # ------------------------------------------------------------------
    # Compute heatmap for a single image
    # ------------------------------------------------------------------

    def compute(
        self,
        img_array: np.ndarray,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Compute Grad-CAM heatmap.

        Args:
            img_array:  Input image, shape (1, H, W, C) or (H, W, C), float32 [0,1] or [-∞,∞].
            class_idx:  Target class index.  If None, uses argmax(predictions).

        Returns:
            heatmap:   (H, W) float32 ndarray in [0, 1].
            class_idx: The target class index used.
            probs:     (num_classes,) softmax probabilities.
        """
        img = self._prepare_input(img_array)   # (1, H, W, C)
        h, w = img.shape[1], img.shape[2]

        if self.grad_model is None:
            logger.warning("grad_model unavailable — returning input saliency fallback")
            return self._saliency_fallback(img, class_idx, h, w)

        try:
            heatmap, class_idx, probs = self._compute_gradcam(img, class_idx, h, w)
        except Exception as exc:
            logger.warning(f"Grad-CAM failed ({exc}) — falling back to input saliency")
            heatmap, class_idx, probs = self._saliency_fallback(img, class_idx, h, w)

        return heatmap, int(class_idx), probs

    def compute_batch(
        self,
        images:    np.ndarray,
        class_idxs: list[int | None] | None = None,
    ) -> list[tuple[np.ndarray, int, np.ndarray]]:
        """Compute Grad-CAM for a batch of images.

        Args:
            images:     (N, H, W, C) float32.
            class_idxs: List of target class indices (length N); None per entry → argmax.

        Returns:
            List of (heatmap, class_idx, probs) tuples.
        """
        if class_idxs is None:
            class_idxs = [None] * len(images)
        return [
            self.compute(images[i : i + 1], class_idxs[i])
            for i in range(len(images))
        ]

    # ------------------------------------------------------------------
    # Internal: core GradCAM
    # ------------------------------------------------------------------

    def _compute_gradcam(
        self,
        img: tf.Tensor,
        class_idx: int | None,
        orig_h: int,
        orig_w: int,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Core Grad-CAM computation."""
        img_tf = tf.cast(img, tf.float32)

        with tf.GradientTape() as tape:
            # Watching img ensures that every intermediate tensor
            # in the forward pass (including conv_out) is in the tape.
            tape.watch(img_tf)
            conv_out, preds = self.grad_model(img_tf, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]).numpy())
            class_score = preds[:, class_idx]

        # Gradient of target class score w.r.t. conv feature maps
        grads = tape.gradient(class_score, conv_out)  # (1, H', W', C)

        if grads is None:
            raise RuntimeError("tape.gradient returned None — cannot compute Grad-CAM")

        # Global-average-pool gradients: importance weights per channel (C,)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

        # Weighted combination of feature maps
        conv_np    = conv_out[0].numpy()          # (H', W', C)
        weights_np = pooled_grads.numpy()          # (C,)
        heatmap    = np.dot(conv_np, weights_np)   # (H', W')

        # ReLU: retain only activations that push towards the target class
        heatmap = np.maximum(heatmap, 0)

        # Resize to input resolution
        heatmap = self._resize_heatmap(heatmap, orig_h, orig_w)

        # Normalise to [0, 1]
        heat_max = heatmap.max()
        if heat_max > 0:
            heatmap = heatmap / heat_max

        return heatmap.astype(np.float32), class_idx, preds.numpy()[0]

    # ------------------------------------------------------------------
    # Internal: fallback (input saliency via gradient w.r.t. image)
    # ------------------------------------------------------------------

    def _saliency_fallback(
        self,
        img: tf.Tensor,
        class_idx: int | None,
        orig_h: int,
        orig_w: int,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Gradient × Input saliency map (Grad-CAM spirit, no nested-model issues).

        Computes gradient of the class score w.r.t. the input image, multiplies
        element-wise by the input (Gradient × Input method), aggregates across
        channels, and normalises.  Produces class-discriminative spatial maps
        similar to Grad-CAM but at input resolution.
        """
        img_tf = tf.cast(img, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tf)
            preds = self.model(img_tf, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]).numpy())
            class_score = preds[:, class_idx]

        grads = tape.gradient(class_score, img_tf)  # (1, H, W, C)
        if grads is None:
            heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
            return heatmap, class_idx, preds.numpy()[0]

        # Gradient × Input: highlight where large gradient meets large activation
        grad_input = grads[0].numpy() * img_tf[0].numpy()  # (H, W, C)
        # Aggregate over channels by max absolute value
        saliency = np.max(np.abs(grad_input), axis=-1)     # (H, W)
        # ReLU — keep only positive contributions
        saliency = np.maximum(saliency, 0)
        mx = saliency.max()
        saliency = (saliency / mx) if mx > 1e-8 else saliency
        return saliency.astype(np.float32), class_idx, preds.numpy()[0]

    # ------------------------------------------------------------------
    # Internal: model building
    # ------------------------------------------------------------------

    def _build_grad_model(self) -> Model | None:
        """Build two-output model: input → [conv_activations, predictions]."""
        try:
            backbone   = self.model.layers[1]   # Sub-model (EfficientNetB3, etc.)
            target_lyr = backbone.get_layer(self.layer_name)

            grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[target_lyr.output, self.model.output],
                name="gradcam_model",
            )
            # output.shape works in Keras 3 (output_shape removed in TF 2.20)
            try:
                conv_shape = target_lyr.output.shape
            except Exception:
                conv_shape = "unknown"
            logger.info(
                f"GradCAM model built | backbone layer='{self.layer_name}' | "
                f"conv shape={conv_shape}"
            )
            return grad_model

        except Exception as exc:
            logger.warning(
                f"Could not build GradCAM grad_model "
                f"(layer='{self.layer_name}'): {exc}\n"
                "Will use input-saliency fallback."
            )
            return None

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _resolve_layer_name(self, hint: str | None) -> str:
        """Pick the best conv layer name for the loaded backbone."""
        if hint:
            return hint  # Config takes precedence

        # Try to infer from backbone model name
        backbone = self.model.layers[1] if len(self.model.layers) > 1 else None
        if backbone is not None:
            bname = backbone.name.lower()
            for key, val in _LAST_CONV_DEFAULTS.items():
                if key.lower() in bname:
                    logger.info(f"Auto-selected Grad-CAM layer: '{val}' for {backbone.name}")
                    return val

            # Auto-detect: last Conv2D in backbone
            for layer in reversed(backbone.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    logger.info(f"Auto-detected last Conv2D: '{layer.name}'")
                    return layer.name

        raise ValueError(
            "Cannot determine Grad-CAM target layer. "
            "Set 'xai.gradcam_layer' in config.yaml."
        )

    @staticmethod
    def _prepare_input(img: np.ndarray) -> tf.Tensor:
        """Ensure input is (1, H, W, C) float32 tensor."""
        if isinstance(img, tf.Tensor):
            img = img.numpy()
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        return tf.constant(img, dtype=tf.float32)

    @staticmethod
    def _resize_heatmap(heatmap: np.ndarray, h: int, w: int) -> np.ndarray:
        """Bilinearly resize heatmap to (h, w)."""
        hm = tf.constant(heatmap[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        hm = tf.image.resize(hm, [h, w], method="bilinear")
        return hm[0, :, :, 0].numpy()
