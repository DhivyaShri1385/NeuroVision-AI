"""
Pseudo-Mask Generator — Phase 3

Generates approximate binary segmentation masks for MRI images that lack
ground-truth pixel annotations (our classification dataset falls in this category).

Strategy:
  1. Load image as grayscale.
  2. Gaussian blur to suppress noise.
  3. Otsu thresholding — automatically finds optimal foreground/background threshold.
  4. Morphological closing — fills small holes inside the lesion region.
  5. Keep the largest connected component — removes spurious noise blobs.
  6. Save as single-channel PNG at the same resolution as the input.

These masks are NOT ground truth. They exist purely so the segmentation
pipeline (data loader, trainer, metrics) can be exercised end-to-end.
When real BraTS-style masks become available, drop in the real files and
retrain — the rest of the pipeline is identical.

Usage:
    from src.segmentation.mask_generator import PseudoMaskGenerator
    gen = PseudoMaskGenerator(config)
    gen.generate_for_split("train")       # saves to masks/train/
    gen.generate_for_split("val")
    gen.generate_for_split("test")
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.config import AppConfig
from src.utils.logger import logger


class PseudoMaskGenerator:
    """Generates pseudo-binary masks using Otsu thresholding + morphology.

    Args:
        config:       AppConfig with segmentation and paths sub-configs.
        masks_root:   Root directory to write masks.  Default: <project_root>/outputs/pseudo_masks/
    """

    def __init__(self, config: AppConfig, masks_root: Path | str | None = None):
        self.config = config
        seg = config.segmentation
        self.target_size = tuple(seg.input_size)   # (H, W)
        self.blur_k      = seg.pseudo_mask_blur     # e.g. 5
        self.morph_k     = seg.pseudo_mask_morph_size  # e.g. 7

        if masks_root is None:
            masks_root = config.paths.outputs_dir / "pseudo_masks"
        self.masks_root = Path(masks_root)
        self.masks_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"PseudoMaskGenerator | masks_root={self.masks_root}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_for_split(self, split: str) -> Path:
        """Generate masks for all images in a split.

        Args:
            split: "train", "val", or "test".

        Returns:
            Path to the mask directory for this split.
        """
        split_dir = self._get_split_dir(split)
        mask_dir  = self.masks_root / split
        mask_dir.mkdir(parents=True, exist_ok=True)

        image_paths = self._collect_images(split_dir)
        logger.info(
            f"Generating pseudo-masks | split={split} | "
            f"images={len(image_paths)} | dest={mask_dir}"
        )

        generated = 0
        skipped   = 0
        for img_path in tqdm(image_paths, desc=f"Masks [{split}]", unit="img"):
            # Preserve class subdirectory in mask output
            rel = img_path.relative_to(split_dir)
            dst = mask_dir / rel.with_suffix(".png")
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists():
                skipped += 1
                continue

            mask = self._generate_single(img_path)
            cv2.imwrite(str(dst), mask)
            generated += 1

        logger.info(
            f"Masks [{split}] done | generated={generated} | "
            f"skipped(existing)={skipped}"
        )
        return mask_dir

    def generate_single_from_array(self, image: np.ndarray) -> np.ndarray:
        """Generate a pseudo-mask from a NumPy image array (uint8 or float32).

        Args:
            image: HxW or HxWxC NumPy array.

        Returns:
            Binary mask (H×W), dtype uint8, values {0, 255}.
        """
        return self._process(image)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_split_dir(self, split: str) -> Path:
        mapping = {
            "train": self.config.paths.train_dir,
            "val":   self.config.paths.val_dir,
            "test":  self.config.paths.test_dir,
        }
        if split not in mapping:
            raise ValueError(f"Unknown split '{split}'. Choose from: train, val, test.")
        return Path(mapping[split])

    def _collect_images(self, root: Path) -> list[Path]:
        """Recursively collect all image files under root."""
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        paths: list[Path] = []
        for dirpath, _, files in os.walk(root):
            for fname in sorted(files):
                if Path(fname).suffix.lower() in exts:
                    paths.append(Path(dirpath) / fname)
        return paths

    def _generate_single(self, img_path: Path) -> np.ndarray:
        """Load image from disk, run pipeline, return binary mask."""
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            h, w = self.target_size
            return np.zeros((h, w), dtype=np.uint8)
        return self._process(img)

    def _process(self, img: np.ndarray) -> np.ndarray:
        """Core processing pipeline: resize → grayscale → blur → Otsu → morph → largest CC."""
        h, w = self.target_size

        # Convert to grayscale
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Resize to target resolution
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

        # Ensure uint8 for OpenCV threshold
        if gray.dtype != np.uint8:
            gray = (gray / gray.max() * 255).clip(0, 255).astype(np.uint8)

        # Gaussian blur
        k = self.blur_k if self.blur_k % 2 == 1 else self.blur_k + 1  # must be odd
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # Otsu thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing to fill holes
        m = self.morph_k if self.morph_k % 2 == 1 else self.morph_k + 1
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))
        closed   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Keep largest connected component only
        mask = self._largest_cc(closed)
        return mask.astype(np.uint8)

    @staticmethod
    def _largest_cc(binary: np.ndarray) -> np.ndarray:
        """Return a binary mask containing only the largest connected component."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels <= 1:
            return binary  # No foreground found

        # stats[:,4] = area; skip label 0 (background)
        largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        out = np.zeros_like(binary)
        out[labels == largest] = 255
        return out
