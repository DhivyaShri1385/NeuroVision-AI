"""
Image Preprocessor — Phase 1

Applies deterministic transformations to every image before it enters
the model. This module handles:

  - Grayscale → RGB conversion (MRI scans are often single-channel)
  - Pixel-range clipping to remove scanner artifact outliers
  - Resize to model input resolution (224×224 for EfficientNet)
  - Normalization with ImageNet statistics (for pretrained backbones)

Design note: keep preprocessing DETERMINISTIC. Stochastic augmentation
lives in augmentation.py and is applied ONLY during training.
"""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path

from src.utils.config import AppConfig, PreprocessingConfig
from src.utils.logger import logger


class ImagePreprocessor:
    """Stateless image preprocessor — safe to use across threads."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg: PreprocessingConfig = config.preprocessing
        self.target_h, self.target_w = self.cfg.image_size
        self.mean = np.array(self.cfg.mean, dtype=np.float32)
        self.std  = np.array(self.cfg.std,  dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_path(self, image_path: str | Path) -> np.ndarray:
        """Load an image from disk and return a preprocessed float32 array.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            np.ndarray of shape (H, W, 3), dtype float32, values ≈ [-2, 2].

        Raises:
            ValueError: If the image cannot be loaded.
        """
        image_path = Path(image_path)
        image = self._load(image_path)
        return self.process_array(image)

    def process_array(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a numpy array that was already loaded.

        Pipeline:
          1. Convert to uint8 if needed
          2. Grayscale → RGB if single-channel
          3. Clip outlier pixel values
          4. Resize to target resolution
          5. Normalize with ImageNet statistics

        Args:
            image: uint8 or float numpy array, shape (H, W) or (H, W, C).

        Returns:
            np.ndarray of shape (target_H, target_W, 3), dtype float32.
        """
        image = self._ensure_uint8(image)
        image = self._to_rgb(image)
        image = self._clip_outliers(image)
        image = self._resize(image)
        image = self._normalize(image)
        return image

    def decode_normalized(self, image: np.ndarray) -> np.ndarray:
        """Reverse normalization for visualization purposes only.

        Returns a uint8 RGB image suitable for matplotlib/cv2 display.
        """
        img = (image * self.std + self.mean) * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> np.ndarray:
        """Load image via OpenCV (handles JPEG, PNG, BMP, TIFF)."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        # OpenCV loads BGR — convert to RGB
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            # RGBA → RGB (drop alpha)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img

    def _ensure_uint8(self, image: np.ndarray) -> np.ndarray:
        """Normalize any float image to uint8 range [0, 255]."""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255.0)
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert grayscale or single-channel MRI to 3-channel RGB.

        Grayscale MRI scans lack color information, so we replicate the
        channel three times. This lets us use ImageNet pretrained weights
        without modification.
        """
        if image.ndim == 2:
            return np.stack([image, image, image], axis=-1)
        if image.ndim == 3 and image.shape[2] == 1:
            return np.concatenate([image, image, image], axis=-1)
        return image  # Already RGB

    def _clip_outliers(self, image: np.ndarray) -> np.ndarray:
        """Clip pixels outside [p_low, p_high] percentile range.

        MRI scanners can produce extreme pixel values (bright spots from
        shimming artifacts, etc.). Clipping to the 1st–99th percentile
        brings the intensity distribution into a consistent range.
        """
        low, high = self.cfg.clip_percentile
        p_low  = np.percentile(image, low)
        p_high = np.percentile(image, high)
        return np.clip(image, p_low, p_high).astype(np.uint8)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize to target resolution using Lanczos interpolation.

        Lanczos is preferred over bilinear/bicubic for medical images
        because it preserves high-frequency edge detail better, which
        matters for detecting tumor boundaries.
        """
        return cv2.resize(
            image,
            (self.target_w, self.target_h),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply ImageNet mean/std normalization.

        Input:  uint8 (H, W, 3) in [0, 255]
        Output: float32 (H, W, 3) approx. in [-2.1, 2.6]

        Using ImageNet statistics is correct because EfficientNet /
        ResNet / DenseNet were all pretrained on ImageNet with these
        exact normalization parameters.
        """
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image.astype(np.float32)
