"""
API Image Preprocessing — Phase 5

Converts uploaded file bytes into the correct tensor format for each model:

  prepare_for_classifier()   → (1, 224, 224, 3) float32, ImageNet-normalised
  prepare_for_segmentation() → (1, 256, 256, 1) float32, [0, 1]

Also provides helpers to convert NumPy arrays back to base64-encoded PNG
strings for JSON responses.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from src.utils.config import AppConfig


# ImageNet normalisation constants (must match Phase 2 training)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Byte → tensor
# ---------------------------------------------------------------------------

def prepare_for_classifier(
    image_bytes: bytes,
    config: AppConfig,
) -> np.ndarray:
    """Decode bytes, resize, convert to RGB, ImageNet-normalise.

    Returns:
        (1, H, W, 3) float32 array ready for the classifier.
    """
    pp  = config.preprocessing
    h, w = pp.image_size

    img = _decode_image(image_bytes)                   # (H', W', 3) uint8
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0               # [0, 1]
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD       # ImageNet normalise
    return img[np.newaxis]                              # (1, H, W, 3)


def prepare_for_segmentation(
    image_bytes: bytes,
    config: AppConfig,
) -> np.ndarray:
    """Decode bytes, resize to segmentation input, convert to grayscale.

    Returns:
        (1, H, W, 1) float32 array in [0, 1].
    """
    seg  = config.segmentation
    h, w = seg.input_size

    img = _decode_image(image_bytes)                   # (H', W', 3) uint8
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)       # (H', W')
    gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0             # [0, 1]
    return gray[np.newaxis, :, :, np.newaxis]          # (1, H, W, 1)


# ---------------------------------------------------------------------------
# Array → base64 PNG
# ---------------------------------------------------------------------------

def array_to_b64(image: np.ndarray) -> str:
    """Convert (H, W) or (H, W, C) float/uint8 array to base64 PNG string."""
    img = _to_uint8(image)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def mask_to_b64(mask: np.ndarray) -> str:
    """Convert binary mask (H, W) float [0,1] to base64 PNG."""
    binary = (mask.squeeze() * 255).clip(0, 255).astype(np.uint8)
    return array_to_b64(binary)


def overlay_mask_on_image(
    image_bytes: bytes,
    mask: np.ndarray,
    alpha: float = 0.45,
    color: tuple[int, int, int] = (255, 0, 0),
) -> str:
    """Blend red tumour mask over the original MRI → base64 PNG.

    Args:
        image_bytes: Original uploaded image bytes.
        mask:        (H, W) or (H, W, 1) float32 binary mask in [0, 1].
        alpha:       Mask opacity.
        color:       RGB colour for the mask overlay.

    Returns:
        Base64-encoded PNG string.
    """
    img   = _decode_image(image_bytes)                 # (H', W', 3) uint8
    h, w  = img.shape[:2]
    msk   = mask.squeeze()
    msk   = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
    msk   = (msk >= 0.5).astype(np.float32)

    overlay = img.astype(np.float32).copy()
    for c, val in enumerate(color):
        overlay[:, :, c] = (
            overlay[:, :, c] * (1 - alpha * msk) + val * alpha * msk
        )
    overlay = overlay.clip(0, 255).astype(np.uint8)
    return array_to_b64(overlay)


def overlay_heatmap_on_image(
    image_bytes: bytes,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> str:
    """Blend Grad-CAM heatmap over the original MRI → base64 PNG.

    Args:
        image_bytes: Original uploaded image bytes.
        heatmap:     (H, W) float32 heatmap in [0, 1].
        alpha:       Heatmap opacity.
        colormap:    OpenCV colormap constant.

    Returns:
        Base64-encoded PNG string.
    """
    img  = _decode_image(image_bytes)                  # (H', W', 3) uint8
    h, w = img.shape[:2]

    hm   = (heatmap.squeeze() * 255).clip(0, 255).astype(np.uint8)
    hm   = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(hm, colormap)             # (H, W, 3) BGR uint8
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    blend = (
        (1 - alpha) * img.astype(np.float32)
        + alpha * heat.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return array_to_b64(blend)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to (H, W, 3) uint8 RGB array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Supported formats: JPEG, PNG, BMP, TIFF.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert any numeric array to uint8 for PIL/PNG encoding."""
    img = image.squeeze()
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        img = (img * 255)
    return img.clip(0, 255).astype(np.uint8)
