"""
Dataset Organizer — Phase 1

Responsibilities:
  - Validate that all expected class folders exist in each split
  - Audit image counts, file extensions, and detect corrupted files
  - Compute and return a structured DatasetStats summary
  - Log all findings in human-readable format

This module does NOT move or modify any files. It only reads and reports.
The stats it returns are consumed by the splitter and EDA visualizer.
"""

from __future__ import annotations

import imghdr
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src.utils.config import AppConfig
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SplitStats:
    name: str
    counts_per_class: dict[str, int] = field(default_factory=dict)
    corrupted_files: list[Path] = field(default_factory=list)
    total: int = 0

    def __post_init__(self) -> None:
        self.total = sum(self.counts_per_class.values())


@dataclass
class DatasetStats:
    splits: dict[str, SplitStats] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        return sum(s.total for s in self.splits.values())

    @property
    def class_names(self) -> list[str]:
        for split in self.splits.values():
            return list(split.counts_per_class.keys())
        return []

    @property
    def is_balanced(self) -> bool:
        """True if every class has the same count within each split."""
        for split in self.splits.values():
            counts = list(split.counts_per_class.values())
            if len(set(counts)) > 1:
                return False
        return True

    @property
    def total_corrupted(self) -> int:
        return sum(len(s.corrupted_files) for s in self.splits.values())


# ---------------------------------------------------------------------------
# Supported Image Extensions
# ---------------------------------------------------------------------------

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Core Organizer
# ---------------------------------------------------------------------------

class DatasetOrganizer:
    """Audits the dataset directory structure and image integrity."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.dataset_root = config.paths.dataset_root
        self.class_names = config.classes.names

    def audit(self) -> DatasetStats:
        """Run a full audit of train / val / test splits.

        Returns:
            DatasetStats: Structured summary of all splits.
        """
        logger.info("=" * 60)
        logger.info("DATASET AUDIT — NeuroVision AI Phase 1")
        logger.info("=" * 60)
        logger.info(f"Dataset root : {self.dataset_root}")
        logger.info(f"Expected classes : {self.class_names}")

        # Map canonical split names to their folder names on disk
        split_dir_map = {
            "train": self.config.paths.train_dir,
            "val":   self.config.paths.val_dir,
            "test":  self.config.paths.test_dir,
        }

        stats = DatasetStats()
        for split_name, split_path in split_dir_map.items():
            split_stats = self._audit_split(split_name, split_path)
            stats.splits[split_name] = split_stats

        self._log_summary(stats)
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _audit_split(self, split_name: str, split_path: Path) -> SplitStats:
        """Audit one split directory."""
        logger.info(f"\n[{split_name.upper()} SPLIT] -> {split_path}")

        if not split_path.exists():
            logger.error(f"  Split directory not found: {split_path}")
            return SplitStats(name=split_name)

        split_stats = SplitStats(name=split_name)

        for class_name in self.class_names:
            class_dir = split_path / class_name
            if not class_dir.exists():
                logger.warning(f"  Missing class folder: {class_dir}")
                split_stats.counts_per_class[class_name] = 0
                continue

            image_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in VALID_EXTENSIONS
            ]

            corrupted = self._find_corrupted(image_files)
            split_stats.corrupted_files.extend(corrupted)

            valid_count = len(image_files) - len(corrupted)
            split_stats.counts_per_class[class_name] = valid_count

            status = "OK" if not corrupted else f"WARN ({len(corrupted)} corrupted)"
            logger.info(
                f"  {class_name:<14} {valid_count:>5} images   [{status}]"
            )

        split_stats.total = sum(split_stats.counts_per_class.values())
        logger.info(f"  {'TOTAL':<14} {split_stats.total:>5} images")
        return split_stats

    def _find_corrupted(self, image_files: list[Path]) -> list[Path]:
        """Identify files that cannot be decoded by OpenCV."""
        corrupted = []
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError("cv2.imread returned None")
                # Confirm file header matches a real image format
                if imghdr.what(str(img_path)) is None and img_path.suffix.lower() not in {".bmp"}:
                    raise ValueError("imghdr could not identify format")
            except Exception as e:
                logger.warning(f"  Corrupted file: {img_path.name} — {e}")
                corrupted.append(img_path)
        return corrupted

    def _log_summary(self, stats: DatasetStats) -> None:
        """Print a concise dataset summary table."""
        logger.info("\n" + "=" * 60)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info(f"{'Split':<10} {'Total':>8} {'Balance':>10}")
        logger.info("-" * 60)

        for split_name, split_stats in stats.splits.items():
            counts = list(split_stats.counts_per_class.values())
            balance = "balanced" if len(set(counts)) == 1 else "IMBALANCED"
            logger.info(
                f"{split_name:<10} {split_stats.total:>8}  {balance:>10}"
            )

        logger.info("-" * 60)
        logger.info(f"{'TOTAL':<10} {stats.total_images:>8}")
        logger.info(f"Corrupted files : {stats.total_corrupted}")
        logger.info(
            f"Dataset balance : {'BALANCED [OK]' if stats.is_balanced else 'IMBALANCED [!!]'}"
        )
        logger.info("=" * 60)

        if stats.total_corrupted > 0:
            logger.warning(
                f"{stats.total_corrupted} corrupted file(s) detected. "
                "They will be excluded from the pipeline automatically."
            )
