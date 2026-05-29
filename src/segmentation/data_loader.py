"""
Segmentation Data Loader — Phase 3

Builds tf.data pipelines that yield (image, mask) pairs for training the
U-Net / Attention U-Net models.

Expected directory structure on disk:
    <split_dir>/
        <class>/
            image1.jpg
            image2.jpg

    <masks_root>/<split>/
        <class>/
            image1.png    ← pseudo-mask (uint8, 0 or 255)
            image2.png

Each image is:
  - Read as grayscale (1 channel)
  - Resized to (H, W) from config.segmentation.input_size
  - Normalised to [0, 1]

Each mask is:
  - Read as grayscale
  - Resized to (H, W) with nearest-neighbour (no blurring of binary values)
  - Thresholded at 0.5 → binary {0, 1} float32

Augmentation (training only):
  - Random horizontal flip
  - Random vertical flip
  - Random rotation ±15° (via tf.keras.preprocessing.image)
  Both image and mask receive IDENTICAL transforms.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.utils.config import AppConfig
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# SegmentationDataLoader
# ---------------------------------------------------------------------------

class SegmentationDataLoader:
    """tf.data pipeline for (image, mask) segmentation pairs.

    Args:
        config:      AppConfig with segmentation + pipeline sub-configs.
        masks_root:  Root of the pseudo-mask directory tree.  Must contain
                     <masks_root>/train/, <masks_root>/val/, <masks_root>/test/
                     sub-trees that mirror the image directory structure.
    """

    def __init__(self, config: AppConfig, masks_root: Path | str):
        self.config     = config
        self.masks_root = Path(masks_root)
        seg             = config.segmentation
        self.h, self.w  = seg.input_size
        self.channels   = seg.channels           # 1 (grayscale)
        self.batch_size = seg.batch_size
        pipe            = config.pipeline
        self.prefetch   = pipe.prefetch_buffer   # -1 → AUTOTUNE
        self.n_parallel = pipe.num_parallel_calls

    # ------------------------------------------------------------------
    # Public: dataset getters
    # ------------------------------------------------------------------

    def get_train_dataset(self) -> tf.data.Dataset:
        """Training dataset with augmentation, shuffle, repeat."""
        img_paths, msk_paths = self._collect_pairs("train")
        ds = self._build_base_dataset(img_paths, msk_paths)
        ds = ds.shuffle(min(len(img_paths), 1000), reshuffle_each_iteration=True)
        ds = ds.map(self._augment_pair, num_parallel_calls=self._autotune())
        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(self._autotune())
        logger.info(
            f"Train seg dataset | pairs={len(img_paths)} | "
            f"batch={self.batch_size}"
        )
        return ds

    def get_val_dataset(self) -> tf.data.Dataset:
        """Validation dataset — no augmentation, no shuffle."""
        img_paths, msk_paths = self._collect_pairs("val")
        ds = self._build_base_dataset(img_paths, msk_paths)
        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(self._autotune())
        logger.info(
            f"Val seg dataset | pairs={len(img_paths)} | batch={self.batch_size}"
        )
        return ds

    def get_test_dataset(self) -> tf.data.Dataset:
        """Test dataset — no augmentation, no shuffle."""
        img_paths, msk_paths = self._collect_pairs("test")
        ds = self._build_base_dataset(img_paths, msk_paths)
        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(self._autotune())
        logger.info(
            f"Test seg dataset | pairs={len(img_paths)} | batch={self.batch_size}"
        )
        return ds

    def num_train_pairs(self) -> int:
        """Return count of training image-mask pairs."""
        imgs, _ = self._collect_pairs("train")
        return len(imgs)

    def num_val_pairs(self) -> int:
        imgs, _ = self._collect_pairs("val")
        return len(imgs)

    # ------------------------------------------------------------------
    # Private: pair collection
    # ------------------------------------------------------------------

    def _collect_pairs(self, split: str) -> tuple[list[str], list[str]]:
        """Return parallel lists of (image_path, mask_path) strings."""
        mapping = {
            "train": self.config.paths.train_dir,
            "val":   self.config.paths.val_dir,
            "test":  self.config.paths.test_dir,
        }
        img_root  = Path(mapping[split])
        mask_root = self.masks_root / split

        img_exts  = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        img_paths: list[str] = []
        msk_paths: list[str] = []

        for img_path in sorted(img_root.rglob("*")):
            if img_path.suffix.lower() not in img_exts:
                continue
            rel  = img_path.relative_to(img_root)
            mask = (mask_root / rel).with_suffix(".png")
            if mask.exists():
                img_paths.append(str(img_path))
                msk_paths.append(str(mask))

        if not img_paths:
            raise RuntimeError(
                f"No image-mask pairs found for split='{split}'.\n"
                f"  Images searched: {img_root}\n"
                f"  Masks  searched: {mask_root}\n"
                "Run PseudoMaskGenerator.generate_for_split() first."
            )

        logger.debug(f"Collected {len(img_paths)} pairs | split={split}")
        return img_paths, msk_paths

    # ------------------------------------------------------------------
    # Private: tf.data helpers
    # ------------------------------------------------------------------

    def _build_base_dataset(
        self,
        img_paths: list[str],
        msk_paths: list[str],
    ) -> tf.data.Dataset:
        """Create a dataset that loads + preprocesses each pair."""
        ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
        ds = ds.map(self._load_and_preprocess, num_parallel_calls=self._autotune())
        return ds

    def _load_and_preprocess(
        self, img_path: tf.Tensor, msk_path: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Load one image+mask pair and normalise."""
        h, w = self.h, self.w

        # Load image
        img_raw  = tf.io.read_file(img_path)
        # Decode — try as JPEG/PNG; force 1 channel (grayscale)
        img = tf.image.decode_image(img_raw, channels=1, expand_animations=False)
        img = tf.image.resize(img, [h, w], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0          # [0, 1]

        # Load mask
        msk_raw = tf.io.read_file(msk_path)
        msk = tf.image.decode_image(msk_raw, channels=1, expand_animations=False)
        msk = tf.image.resize(msk, [h, w], method="nearest")
        msk = tf.cast(msk, tf.float32) / 255.0          # {0, 1}
        msk = tf.cast(msk >= 0.5, tf.float32)           # hard binary

        return img, msk

    def _augment_pair(
        self, img: tf.Tensor, msk: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply identical random spatial augmentations to image and mask."""
        # Concat on channel axis so both receive the same random op
        combined = tf.concat([img, msk], axis=-1)  # (H, W, 2)

        # Random horizontal flip
        combined = tf.image.random_flip_left_right(combined)

        # Random vertical flip
        combined = tf.image.random_flip_up_down(combined)

        # Split back
        img = combined[:, :, :1]
        msk = combined[:, :, 1:]
        msk = tf.cast(msk >= 0.5, tf.float32)  # re-binarise after any interpolation

        return img, msk

    def _autotune(self) -> int | tf.data.AUTOTUNE:
        v = self.n_parallel
        return tf.data.AUTOTUNE if v == -1 else v
