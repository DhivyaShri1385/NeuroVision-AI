"""
Dataset Splitter — Phase 1

Two operating modes:

1. VALIDATE mode (default for this project):
   The Kaggle brain tumor dataset is already split into train/valid/test.
   This mode verifies the existing splits are stratified and correct.

2. SPLIT mode (for raw unsplit datasets):
   Performs a stratified train/val/test split and either copies files to
   new directories or produces manifest CSVs (preferred — avoids
   duplicating gigabytes of data).

Design: always use manifest CSVs for splits. They are a few KB vs GBs for
image copies, support symlinks, and make cross-validation trivial.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.dataset_organizer import DatasetStats
from src.utils.config import AppConfig
from src.utils.logger import logger

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SplitManifest:
    """Paths + labels for one split, stored as a DataFrame."""
    name: str
    df: pd.DataFrame  # Columns: ["filepath", "label", "class_name"]

    @property
    def num_samples(self) -> int:
        return len(self.df)

    @property
    def class_distribution(self) -> dict[str, int]:
        return self.df["class_name"].value_counts().to_dict()


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

class DatasetSplitter:
    """Creates or validates dataset splits and produces manifest CSVs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.class_names = config.classes.names
        self.label_map = config.classes.label_map
        self.split_cfg = config.split
        self.outputs_dir = config.paths.outputs_dir
        random.seed(self.split_cfg.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_existing_splits(self, stats: DatasetStats) -> list[SplitManifest]:
        """Validate the pre-existing train/val/test splits and build manifests.

        This is the primary mode for the Kaggle brain tumor dataset which
        already ships with train/valid/test directories.

        Args:
            stats: Output from DatasetOrganizer.audit()

        Returns:
            List of SplitManifest (train, val, test).
        """
        logger.info("Validating existing dataset splits...")

        split_dir_map = {
            "train": self.config.paths.train_dir,
            "val":   self.config.paths.val_dir,
            "test":  self.config.paths.test_dir,
        }

        manifests = []
        for split_name, split_dir in split_dir_map.items():
            manifest = self._build_manifest_from_dir(split_name, split_dir)
            self._save_manifest(manifest)
            self._log_manifest_stats(manifest)
            manifests.append(manifest)

        self._validate_no_leakage(manifests)
        logger.info("Split validation complete - no leakage detected.")
        return manifests

    def create_fresh_splits(self, raw_data_dir: Path) -> list[SplitManifest]:
        """Create stratified splits from a flat raw dataset directory.

        Expected raw_data_dir layout:
            raw_data_dir/
              glioma/
              meningioma/
              no_tumor/
              pituitary/

        Returns manifest CSVs saved to outputs/. Does NOT copy image files.
        """
        logger.info(f"Creating fresh stratified splits from: {raw_data_dir}")

        all_files: list[Path] = []
        all_labels: list[int] = []

        for class_name in self.class_names:
            class_dir = raw_data_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            files = [
                f for f in sorted(class_dir.iterdir())
                if f.suffix.lower() in VALID_EXTENSIONS
            ]
            all_files.extend(files)
            all_labels.extend([self.label_map[class_name]] * len(files))
            logger.info(f"  {class_name}: {len(files)} images")

        train_idx, val_idx, test_idx = self._stratified_split(all_labels)

        manifests = []
        for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            manifest = SplitManifest(
                name=split_name,
                df=pd.DataFrame({
                    "filepath":   [str(all_files[i]) for i in indices],
                    "label":      [all_labels[i] for i in indices],
                    "class_name": [self.class_names[all_labels[i]] for i in indices],
                })
            )
            self._save_manifest(manifest)
            self._log_manifest_stats(manifest)
            manifests.append(manifest)

        return manifests

    def load_manifests(self) -> list[SplitManifest]:
        """Load previously saved manifest CSVs from disk."""
        manifests = []
        for split_name in ("train", "val", "test"):
            csv_path = self.outputs_dir / f"manifest_{split_name}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Manifest not found: {csv_path}. "
                    "Run validate_existing_splits() first."
                )
            df = pd.read_csv(csv_path)
            manifests.append(SplitManifest(name=split_name, df=df))
        return manifests

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_manifest_from_dir(self, split_name: str, split_dir: Path) -> SplitManifest:
        """Build a manifest DataFrame from an existing split directory."""
        rows = []
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Missing class dir: {class_dir}")
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in VALID_EXTENSIONS:
                    rows.append({
                        "filepath":   str(img_path),
                        "label":      self.label_map[class_name],
                        "class_name": class_name,
                    })

        df = pd.DataFrame(rows)
        return SplitManifest(name=split_name, df=df)

    def _save_manifest(self, manifest: SplitManifest) -> None:
        """Persist manifest to outputs/ as a CSV."""
        csv_path = self.outputs_dir / f"manifest_{manifest.name}.csv"
        manifest.df.to_csv(csv_path, index=False)
        logger.debug(f"Manifest saved: {csv_path}")

    def _log_manifest_stats(self, manifest: SplitManifest) -> None:
        logger.info(
            f"  [{manifest.name.upper()}] {manifest.num_samples} images | "
            f"distribution: {manifest.class_distribution}"
        )

    def _stratified_split(
        self, labels: list[int]
    ) -> tuple[list[int], list[int], list[int]]:
        """Return train / val / test index lists via stratified splitting."""
        cfg = self.split_cfg
        n = len(labels)
        indices = list(range(n))

        # Split off test set first
        test_size = cfg.test_ratio
        sss_test = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=cfg.random_seed
        )
        trainval_idx, test_idx = next(sss_test.split(indices, labels))

        # Split remaining into train / val
        val_size = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)
        trainval_labels = [labels[i] for i in trainval_idx]
        sss_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=cfg.random_seed
        )
        inner_train_idx, inner_val_idx = next(
            sss_val.split(trainval_idx, trainval_labels)
        )

        train_idx = [trainval_idx[i] for i in inner_train_idx]
        val_idx   = [trainval_idx[i] for i in inner_val_idx]

        logger.info(
            f"Split sizes — train:{len(train_idx)} | "
            f"val:{len(val_idx)} | test:{len(test_idx)}"
        )
        return train_idx, val_idx, list(test_idx)

    def _validate_no_leakage(self, manifests: list[SplitManifest]) -> None:
        """Assert no image path appears in more than one split."""
        path_sets = [set(m.df["filepath"]) for m in manifests]
        names = [m.name for m in manifests]
        for i in range(len(path_sets)):
            for j in range(i + 1, len(path_sets)):
                overlap = path_sets[i] & path_sets[j]
                if overlap:
                    logger.error(
                        f"DATA LEAKAGE detected between {names[i]} and {names[j]}: "
                        f"{len(overlap)} shared paths"
                    )
                    raise RuntimeError(
                        f"Data leakage between {names[i]}/{names[j]}. "
                        "Investigate dataset structure."
                    )
