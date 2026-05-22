"""
Medical Image Augmentation Pipeline — Phase 1

Uses Albumentations, which is 10–50× faster than TF augmentation layers
for CPU preprocessing and produces identical results on GPU via numpy.

Medical imaging constraints respected:
  - NO vertical flips  (brain anatomy has a fixed superior/inferior axis)
  - Small rotation limits  (±15° max — more causes anatomically invalid images)
  - Conservative elastic deformation  (simulates patient positioning variance)
  - NO random crops  (could remove diagnostically critical regions)
  - Brightness/contrast variations  (simulate scanner calibration differences)

Pipeline is composed at init time for performance (Albumentations compiles
the transform graph once). The same pipeline instance is reused per epoch.
"""

from __future__ import annotations

import numpy as np
import albumentations as A
from albumentations.core.composition import Compose

from src.utils.config import AppConfig, AugmentationConfig
from src.utils.logger import logger


class MedicalAugmentationPipeline:
    """Builds and applies an Albumentations augmentation pipeline.

    Usage:
        pipeline = MedicalAugmentationPipeline(config)

        # During training
        augmented = pipeline.apply_train(image)   # Returns augmented uint8 array

        # During val/test — passthrough (no augmentation)
        result = pipeline.apply_eval(image)
    """

    def __init__(self, config: AppConfig) -> None:
        self.aug_cfg: AugmentationConfig = config.augmentation
        self._train_transform: Compose = self._build_train_transform()
        self._eval_transform:  Compose = self._build_eval_transform()
        logger.info(
            f"Augmentation pipeline initialized | enabled={self.aug_cfg.enabled}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_train(self, image: np.ndarray) -> np.ndarray:
        """Apply stochastic augmentation for training.

        Args:
            image: uint8 RGB numpy array (H, W, 3).

        Returns:
            Augmented uint8 RGB numpy array (H, W, 3).
        """
        if not self.aug_cfg.enabled:
            return image
        result = self._train_transform(image=image)
        return result["image"]

    def apply_eval(self, image: np.ndarray) -> np.ndarray:
        """Apply eval transform (currently identity — no augmentation).

        Kept as a method for future TTA (test-time augmentation) support.

        Args:
            image: uint8 RGB numpy array (H, W, 3).

        Returns:
            Unchanged image.
        """
        result = self._eval_transform(image=image)
        return result["image"]

    def get_train_transform(self) -> Compose:
        """Expose underlying Albumentations compose for use in tf.data."""
        return self._train_transform

    def get_eval_transform(self) -> Compose:
        """Expose underlying Albumentations compose for use in tf.data."""
        return self._eval_transform

    # ------------------------------------------------------------------
    # Pipeline builders
    # ------------------------------------------------------------------

    def _build_train_transform(self) -> Compose:
        """Construct the full training augmentation pipeline."""
        cfg = self.aug_cfg
        p = cfg.augmentation_probability

        transforms = []

        # Spatial transforms
        if cfg.horizontal_flip:
            transforms.append(A.HorizontalFlip(p=p))

        # Albumentations 2.x uses Affine instead of ShiftScaleRotate
        transforms.append(
            A.Affine(
                translate_percent={"x": (-cfg.shift_limit, cfg.shift_limit),
                                   "y": (-cfg.shift_limit, cfg.shift_limit)},
                scale=(1 - cfg.scale_limit, 1 + cfg.scale_limit),
                rotate=(-cfg.rotation_limit, cfg.rotation_limit),
                interpolation=4,     # Lanczos
                cval=0,              # Black padding value — anatomically safe
                p=p,
            )
        )

        # Elastic deformation — simulates patient positioning variance
        if cfg.elastic_transform:
            transforms.append(
                A.ElasticTransform(
                    alpha=cfg.elastic_alpha,
                    sigma=cfg.elastic_sigma,
                    p=p * 0.5,
                )
            )

        # Pixel-level transforms — simulate scanner variability
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=p,
            )
        )

        transforms.append(
            A.GaussianBlur(
                blur_limit=cfg.blur_limit,
                p=p * 0.3,
            )
        )

        # GaussNoise in albumentations 2.x uses std_range
        noise_std_low  = (cfg.gaussian_noise_variance[0] ** 0.5) / 255.0
        noise_std_high = (cfg.gaussian_noise_variance[1] ** 0.5) / 255.0
        transforms.append(
            A.GaussNoise(
                std_range=(noise_std_low, noise_std_high),
                p=p * 0.4,
            )
        )

        # CLAHE — improves local contrast, especially useful for MRI
        transforms.append(
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=p * 0.3,
            )
        )

        # CoarseDropout in albumentations 2.x uses range-based params
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                fill=0,
                p=p * 0.2,
            )
        )

        return A.Compose(transforms)

    def _build_eval_transform(self) -> Compose:
        """No-op pipeline — return image unchanged."""
        return A.Compose([])  # Identity transform


# ---------------------------------------------------------------------------
# Standalone helper — useful in tf.data py_function wrappers
# ---------------------------------------------------------------------------

def build_train_augmenter(config: AppConfig):
    """Return a callable that augments a single uint8 numpy image.

    Designed for use inside tf.data.Dataset.map() via py_function.

    Example:
        augment_fn = build_train_augmenter(config)
        ds = ds.map(lambda img, lbl: (
            tf.numpy_function(augment_fn, [img], tf.uint8), lbl
        ))
    """
    pipeline = MedicalAugmentationPipeline(config)

    def _augment(image: np.ndarray) -> np.ndarray:
        return pipeline.apply_train(image)

    return _augment
