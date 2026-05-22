"""
tf.data Pipeline — Phase 1

Builds highly optimized TensorFlow Dataset pipelines for train/val/test.

Why tf.data over keras.preprocessing.image.ImageDataGenerator?
  - Parallel file reads (num_parallel_calls=AUTOTUNE)
  - Prefetching — GPU never waits for data
  - Cache after preprocessing — repeated epochs are fast
  - Native integration with tf.distribute for multi-GPU training
  - Composable with @tf.function for full graph optimization

Pipeline stages per split:
  Train:  read → decode → preprocess → augment → batch → prefetch
  Val:    read → decode → preprocess → batch → prefetch
  Test:   read → decode → preprocess → batch (no shuffle)

The Albumentations augmentation is applied via tf.numpy_function which
wraps Python/NumPy code safely inside the tf.data graph.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from src.data.augmentation import MedicalAugmentationPipeline
from src.data.preprocessor import ImagePreprocessor
from src.utils.config import AppConfig
from src.utils.logger import logger

AUTOTUNE = tf.data.AUTOTUNE


class BrainTumorDataLoader:
    """Produces tf.data.Dataset objects for all three splits.

    Usage:
        loader = BrainTumorDataLoader(config)
        train_ds = loader.get_train_dataset()
        val_ds   = loader.get_val_dataset()
        test_ds  = loader.get_test_dataset()
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.preprocessor = ImagePreprocessor(config)
        self.augmenter = MedicalAugmentationPipeline(config)
        self.num_classes = config.classes.num_classes
        self.batch_size = config.training.batch_size
        self.img_h, self.img_w = config.preprocessing.image_size
        self.channels = config.preprocessing.channels
        self.pl_cfg = config.pipeline

        logger.info(
            f"DataLoader initialized | "
            f"batch={self.batch_size} | "
            f"img_size={config.preprocessing.image_size}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_train_dataset(
        self,
        manifest_path: str | Path | None = None,
        split_dir: Path | None = None,
    ) -> tf.data.Dataset:
        """Build the training dataset with augmentation and shuffling."""
        df = self._load_df(manifest_path, "train", split_dir)
        paths, labels = self._extract_paths_labels(df)

        ds = self._build_base_dataset(paths, labels)
        ds = ds.shuffle(
            buffer_size=self.pl_cfg.shuffle_buffer,
            seed=self.config.split.random_seed,
            reshuffle_each_iteration=True,
        )
        ds = ds.map(self._load_and_preprocess_train, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTOTUNE)

        logger.info(f"Train dataset: {len(paths)} images, {len(paths) // self.batch_size} steps/epoch")
        return ds

    def get_val_dataset(
        self,
        manifest_path: str | Path | None = None,
        split_dir: Path | None = None,
    ) -> tf.data.Dataset:
        """Build the validation dataset (no augmentation, no shuffle)."""
        df = self._load_df(manifest_path, "val", split_dir)
        paths, labels = self._extract_paths_labels(df)

        ds = self._build_base_dataset(paths, labels)
        ds = ds.map(self._load_and_preprocess_eval, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)

        logger.info(f"Val dataset: {len(paths)} images")
        return ds

    def get_test_dataset(
        self,
        manifest_path: str | Path | None = None,
        split_dir: Path | None = None,
        return_paths: bool = False,
    ) -> tf.data.Dataset:
        """Build the test dataset (no augmentation, no shuffle, ordered).

        Args:
            return_paths: If True, dataset yields (image, label, path) tuples.
                          Useful for error analysis.
        """
        df = self._load_df(manifest_path, "test", split_dir)
        paths, labels = self._extract_paths_labels(df)

        if return_paths:
            path_tensors = tf.constant(paths)
            label_tensors = tf.constant(labels, dtype=tf.int32)
            ds = tf.data.Dataset.from_tensor_slices((path_tensors, label_tensors, path_tensors))
            ds = ds.map(self._load_preprocess_with_path, num_parallel_calls=AUTOTUNE)
        else:
            ds = self._build_base_dataset(paths, labels)
            ds = ds.map(self._load_and_preprocess_eval, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)

        logger.info(f"Test dataset: {len(paths)} images")
        return ds

    def get_class_weights(self, manifest_path: str | Path | None = None) -> dict[int, float]:
        """Compute inverse-frequency class weights for imbalanced datasets.

        Uses sklearn convention: weight_i = N / (num_classes * count_i)
        so that all classes contribute equally to the loss.
        """
        df = self._load_df(manifest_path, "train", None)
        counts = df["label"].value_counts().sort_index()
        n_total = len(df)
        weights = {
            label: n_total / (self.num_classes * count)
            for label, count in counts.items()
        }
        logger.info(f"Class weights: {weights}")
        return weights

    def get_steps_per_epoch(
        self,
        manifest_path: str | Path | None = None,
        split: str = "train",
    ) -> int:
        """Return the number of batches per epoch for a given split."""
        df = self._load_df(manifest_path, split, None)
        return len(df) // self.batch_size

    # ------------------------------------------------------------------
    # Fallback: directory-based loading (when no manifest CSV exists)
    # ------------------------------------------------------------------

    def get_train_dataset_from_dir(self) -> tf.data.Dataset:
        """Build train dataset directly from train_dir (no manifest needed)."""
        train_dir = self.config.paths.train_dir
        df = self._scan_directory(train_dir)
        paths, labels = self._extract_paths_labels(df)
        ds = self._build_base_dataset(paths, labels)
        ds = ds.shuffle(self.pl_cfg.shuffle_buffer, seed=self.config.split.random_seed)
        ds = ds.map(self._load_and_preprocess_train, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        return ds.prefetch(AUTOTUNE)

    def get_val_dataset_from_dir(self) -> tf.data.Dataset:
        """Build val dataset directly from val_dir."""
        val_dir = self.config.paths.val_dir
        df = self._scan_directory(val_dir)
        paths, labels = self._extract_paths_labels(df)
        ds = self._build_base_dataset(paths, labels)
        ds = ds.map(self._load_and_preprocess_eval, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size, drop_remainder=False)
        return ds.prefetch(AUTOTUNE)

    def get_test_dataset_from_dir(self) -> tf.data.Dataset:
        """Build test dataset directly from test_dir."""
        test_dir = self.config.paths.test_dir
        df = self._scan_directory(test_dir)
        paths, labels = self._extract_paths_labels(df)
        ds = self._build_base_dataset(paths, labels)
        ds = ds.map(self._load_and_preprocess_eval, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size, drop_remainder=False)
        return ds.prefetch(AUTOTUNE)

    # ------------------------------------------------------------------
    # tf.data map functions
    # ------------------------------------------------------------------

    def _load_and_preprocess_train(
        self, path: tf.Tensor, label: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Load, preprocess, and augment one training image."""
        image = self._decode_image(path)

        # Apply Albumentations augmentation via py_function
        image = tf.numpy_function(
            func=self._augment_numpy,
            inp=[image],
            Tout=tf.float32,
        )
        image.set_shape([self.img_h, self.img_w, self.channels])

        label_onehot = tf.one_hot(label, depth=self.num_classes)
        return image, label_onehot

    def _load_and_preprocess_eval(
        self, path: tf.Tensor, label: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Load and preprocess one eval image (no augmentation)."""
        image = self._decode_image(path)
        label_onehot = tf.one_hot(label, depth=self.num_classes)
        return image, label_onehot

    def _load_preprocess_with_path(
        self, path: tf.Tensor, label: tf.Tensor, path_copy: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Load, preprocess, and return (image, label, path) for test analysis."""
        image = self._decode_image(path)
        label_onehot = tf.one_hot(label, depth=self.num_classes)
        return image, label_onehot, path_copy

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _decode_image(self, path: tf.Tensor) -> tf.Tensor:
        """Read → decode JPEG/PNG → convert channels → resize → normalize."""
        raw = tf.io.read_file(path)

        # Attempt JPEG decode first, fall back to PNG
        image = tf.io.decode_image(
            raw,
            channels=self.channels,
            expand_animations=False,
            dtype=tf.uint8,
        )
        image.set_shape([None, None, self.channels])

        # Resize with bilinear interpolation (fast, used inside tf graph)
        image = tf.image.resize(
            image,
            [self.img_h, self.img_w],
            method=tf.image.ResizeMethod.BILINEAR,
        )
        image = tf.cast(image, tf.float32) / 255.0

        # ImageNet normalization
        mean = tf.constant(self.config.preprocessing.mean, dtype=tf.float32)
        std  = tf.constant(self.config.preprocessing.std,  dtype=tf.float32)
        image = (image - mean) / std
        return image

    def _augment_numpy(self, image: np.ndarray) -> np.ndarray:
        """Numpy-side augmentation called inside tf.numpy_function."""
        # Denormalize back to uint8 for Albumentations
        mean = np.array(self.config.preprocessing.mean, dtype=np.float32)
        std  = np.array(self.config.preprocessing.std,  dtype=np.float32)
        img_uint8 = np.clip((image * std + mean) * 255.0, 0, 255).astype(np.uint8)
        img_aug = self.augmenter.apply_train(img_uint8)
        # Re-normalize
        result = img_aug.astype(np.float32) / 255.0
        result = (result - mean) / std
        return result.astype(np.float32)

    def _build_base_dataset(
        self, paths: list[str], labels: list[int]
    ) -> tf.data.Dataset:
        """Build the initial path/label tensor dataset."""
        path_tensor  = tf.constant(paths,  dtype=tf.string)
        label_tensor = tf.constant(labels, dtype=tf.int32)
        return tf.data.Dataset.from_tensor_slices((path_tensor, label_tensor))

    def _extract_paths_labels(
        self, df: pd.DataFrame
    ) -> tuple[list[str], list[int]]:
        return df["filepath"].tolist(), df["label"].tolist()

    def _load_df(
        self,
        manifest_path: str | Path | None,
        split: str,
        split_dir: Path | None,
    ) -> pd.DataFrame:
        """Load manifest CSV or scan directory as fallback."""
        if manifest_path is not None:
            return pd.read_csv(manifest_path)

        default_csv = self.config.paths.outputs_dir / f"manifest_{split}.csv"
        if default_csv.exists():
            return pd.read_csv(default_csv)

        # Last resort: scan the split directory directly
        if split_dir is not None:
            return self._scan_directory(split_dir)

        dir_map = {"train": self.config.paths.train_dir,
                   "val":   self.config.paths.val_dir,
                   "test":  self.config.paths.test_dir}
        return self._scan_directory(dir_map[split])

    def _scan_directory(self, directory: Path) -> pd.DataFrame:
        """Build a DataFrame by scanning a class-per-folder directory."""
        label_map = self.config.classes.label_map
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        rows = []
        for class_name in self.config.classes.names:
            class_dir = directory / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in valid_ext:
                    rows.append({
                        "filepath":   str(img_path),
                        "label":      label_map[class_name],
                        "class_name": class_name,
                    })
        return pd.DataFrame(rows)
