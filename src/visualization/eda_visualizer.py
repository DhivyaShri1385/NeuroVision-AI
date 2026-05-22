"""
EDA Visualizer — Phase 1

Generates publication-ready plots for exploratory data analysis:

  1. Class distribution bar chart (all splits)
  2. Sample image grid (random samples per class)
  3. Pixel intensity histograms (per class)
  4. Augmentation preview (before/after grid)
  5. Dataset statistics summary table (saved as PNG)

All plots saved to outputs/plots/ as high-DPI PNGs.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from src.data.augmentation import MedicalAugmentationPipeline
from src.data.dataset_organizer import DatasetStats
from src.utils.config import AppConfig
from src.utils.logger import logger

# Use non-interactive backend for server/script environments
matplotlib.use("Agg")

# NeuroVision color palette — consistent across all plots
CLASS_COLORS = {
    "glioma":      "#E74C3C",   # Red
    "meningioma":  "#3498DB",   # Blue
    "no_tumor":    "#2ECC71",   # Green
    "pituitary":   "#F39C12",   # Orange
}
BACKGROUND_COLOR = "#1A1A2E"   # Dark navy — matches frontend dark theme
TEXT_COLOR = "#ECF0F1"
GRID_COLOR = "#2C3E50"
DPI = 150


def _apply_dark_style() -> None:
    """Apply the NeuroVision dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor":  BACKGROUND_COLOR,
        "axes.facecolor":    "#16213E",
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "grid.color":        GRID_COLOR,
        "grid.alpha":        0.4,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "legend.facecolor":  "#16213E",
        "legend.edgecolor":  GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
    })


class EDAVisualizer:
    """Generates and saves all Phase 1 EDA plots."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.plots_dir = config.paths.plots_dir
        self.class_names = config.classes.names
        self.label_map = config.classes.label_map
        self.aug_pipeline = MedicalAugmentationPipeline(config)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        _apply_dark_style()
        logger.info(f"EDAVisualizer initialized | output: {self.plots_dir}")

    # ------------------------------------------------------------------
    # Public API — call these from run_phase1.py
    # ------------------------------------------------------------------

    def plot_class_distribution(self, stats: DatasetStats) -> Path:
        """Grouped bar chart showing image counts per class per split."""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(BACKGROUND_COLOR)

        split_names = list(stats.splits.keys())
        x = np.arange(len(self.class_names))
        width = 0.25
        split_colors = ["#9B59B6", "#1ABC9C", "#E67E22"]

        for idx, (split_name, color) in enumerate(zip(split_names, split_colors)):
            split_stats = stats.splits[split_name]
            counts = [split_stats.counts_per_class.get(cls, 0) for cls in self.class_names]
            bars = ax.bar(
                x + idx * width,
                counts,
                width,
                label=split_name.capitalize(),
                color=color,
                alpha=0.85,
                edgecolor=BACKGROUND_COLOR,
                linewidth=0.8,
            )
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3,
                    str(count),
                    ha="center", va="bottom",
                    fontsize=9, color=TEXT_COLOR,
                )

        ax.set_title("Class Distribution Across Splits", pad=15, fontweight="bold")
        ax.set_xlabel("Tumor Class")
        ax.set_ylabel("Image Count")
        ax.set_xticks(x + width)
        ax.set_xticklabels([c.replace("_", " ").title() for c in self.class_names])
        ax.legend(loc="upper right")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Annotate total per class
        for i, cls in enumerate(self.class_names):
            total = sum(
                stats.splits[s].counts_per_class.get(cls, 0) for s in split_names
            )
            ax.text(
                x[i] + width, -28, f"Total: {total}",
                ha="center", va="top", fontsize=8.5, color="#BDC3C7", style="italic",
            )

        plt.tight_layout()
        out = self.plots_dir / "class_distribution.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        plt.close()
        logger.info(f"Saved: {out}")
        return out

    def plot_sample_grid(
        self,
        split_dir: Path,
        split_name: str = "train",
        samples_per_class: int = 4,
    ) -> Path:
        """Grid of random sample MRI images per class."""
        fig = plt.figure(figsize=(samples_per_class * 3, len(self.class_names) * 3 + 0.8))
        fig.patch.set_facecolor(BACKGROUND_COLOR)

        gs = gridspec.GridSpec(
            len(self.class_names), samples_per_class,
            hspace=0.35, wspace=0.05,
        )
        fig.suptitle(
            f"Sample MRI Images — {split_name.capitalize()} Split",
            fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.01,
        )

        for row, class_name in enumerate(self.class_names):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            image_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            sampled = random.sample(image_files, min(samples_per_class, len(image_files)))

            for col, img_path in enumerate(sampled):
                ax = fig.add_subplot(gs[row, col])
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(
                        class_name.replace("_", " ").title(),
                        fontsize=11, color=CLASS_COLORS.get(class_name, TEXT_COLOR),
                        rotation=90, labelpad=5,
                    )
                    ax.yaxis.set_label_position("left")
                if row == 0:
                    ax.set_title(f"Sample {col + 1}", fontsize=9, color=TEXT_COLOR, pad=4)

        plt.tight_layout()
        out = self.plots_dir / f"sample_grid_{split_name}.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        plt.close()
        logger.info(f"Saved: {out}")
        return out

    def plot_intensity_distributions(self, split_dir: Path, n_samples: int = 50) -> Path:
        """Per-class pixel intensity histogram overlay."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(BACKGROUND_COLOR)

        for ax, channel_idx, channel_name in zip(
            axes, [0, 1], ["Red Channel (Grayscale proxy)", "All Channels"]
        ):
            ax.set_facecolor("#16213E")
            ax.set_title(channel_name, fontweight="bold")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Density")

            for class_name in self.class_names:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    continue

                image_files = [
                    f for f in class_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
                sampled = random.sample(image_files, min(n_samples, len(image_files)))

                pixels = []
                for img_path in sampled:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if channel_idx == 0:
                        pixels.extend(img[:, :, 0].flatten().tolist())
                    else:
                        pixels.extend(img.flatten().tolist())

                if pixels:
                    color = CLASS_COLORS.get(class_name, TEXT_COLOR)
                    sns.kdeplot(
                        pixels,
                        ax=ax,
                        label=class_name.replace("_", " ").title(),
                        color=color,
                        linewidth=2,
                        fill=True,
                        alpha=0.15,
                    )

            ax.legend(fontsize=9)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            ax.set_xlim(0, 255)

        plt.suptitle(
            "Pixel Intensity Distribution by Tumor Class",
            fontsize=14, fontweight="bold", color=TEXT_COLOR,
        )
        plt.tight_layout()
        out = self.plots_dir / "intensity_distributions.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        plt.close()
        logger.info(f"Saved: {out}")
        return out

    def plot_augmentation_preview(self, split_dir: Path, n_augmented: int = 5) -> Path:
        """Side-by-side grid showing original vs augmented images."""
        # Pick one image per class
        class_examples = {}
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if files:
                class_examples[class_name] = random.choice(files)

        num_classes = len(class_examples)
        cols = n_augmented + 1  # original + augmented versions
        fig, axes = plt.subplots(
            num_classes, cols,
            figsize=(cols * 2.2, num_classes * 2.5),
        )
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        fig.suptitle(
            "Augmentation Preview (Original → Augmented Variants)",
            fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01,
        )

        if num_classes == 1:
            axes = [axes]

        for row, (class_name, img_path) in enumerate(class_examples.items()):
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Original
            ax_orig = axes[row][0]
            ax_orig.imshow(img_rgb)
            ax_orig.axis("off")
            ax_orig.set_title("Original", fontsize=8, color=TEXT_COLOR, pad=3)
            if row == 0 or True:
                ax_orig.set_ylabel(
                    class_name.replace("_", " ").title(),
                    fontsize=9, color=CLASS_COLORS.get(class_name, TEXT_COLOR),
                    rotation=90, labelpad=4,
                )
                ax_orig.yaxis.set_label_position("left")

            # Augmented versions
            for col in range(1, cols):
                aug_img = self.aug_pipeline.apply_train(img_rgb.copy())
                ax = axes[row][col]
                ax.imshow(aug_img)
                ax.axis("off")
                if row == 0:
                    ax.set_title(f"Aug {col}", fontsize=8, color=TEXT_COLOR, pad=3)

        plt.tight_layout()
        out = self.plots_dir / "augmentation_preview.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        plt.close()
        logger.info(f"Saved: {out}")
        return out

    def plot_dataset_stats_table(self, stats: DatasetStats) -> Path:
        """Render dataset statistics as a styled table image."""
        rows = []
        for split_name, split_stats in stats.splits.items():
            for class_name in self.class_names:
                rows.append({
                    "Split":      split_name.capitalize(),
                    "Class":      class_name.replace("_", " ").title(),
                    "Count":      split_stats.counts_per_class.get(class_name, 0),
                    "Corrupted":  0,  # Populated from audit in full run
                })
        df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis("off")

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style header
        for col_idx in range(len(df.columns)):
            table[0, col_idx].set_facecolor("#0F3460")
            table[0, col_idx].set_text_props(color=TEXT_COLOR, fontweight="bold")

        # Style data rows
        split_row_colors = {
            "Train":   "#1A1A4E",
            "Val":     "#1A3A2E",
            "Test":    "#3A1A1A",
        }
        for row_idx in range(1, len(df) + 1):
            split_val = df.iloc[row_idx - 1]["Split"]
            row_color = split_row_colors.get(split_val, "#1A1A2E")
            for col_idx in range(len(df.columns)):
                table[row_idx, col_idx].set_facecolor(row_color)
                table[row_idx, col_idx].set_text_props(color=TEXT_COLOR)
                table[row_idx, col_idx].set_edgecolor(GRID_COLOR)

        ax.set_title(
            "NeuroVision AI — Dataset Statistics",
            fontsize=14, fontweight="bold", color=TEXT_COLOR,
            pad=20, y=0.98,
        )

        out = self.plots_dir / "dataset_stats_table.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        plt.close()
        logger.info(f"Saved: {out}")
        return out

    def plot_image_size_distribution(self, split_dir: Path, n_samples: int = 100) -> Path:
        """Scatter plot of image dimensions to check for size inconsistency."""
        widths, heights, labels = [], [], []

        for class_name in self.class_names:
            class_dir = split_dir / class_name
            files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            sampled = random.sample(files, min(n_samples, len(files)))
            for f in sampled:
                img = cv2.imread(str(f))
                if img is not None:
                    heights.append(img.shape[0])
                    widths.append(img.shape[1])
                    labels.append(class_name)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(BACKGROUND_COLOR)

        # Scatter: width vs height
        ax = axes[0]
        ax.set_facecolor("#16213E")
        for class_name in self.class_names:
            idxs = [i for i, l in enumerate(labels) if l == class_name]
            ax.scatter(
                [widths[i] for i in idxs],
                [heights[i] for i in idxs],
                label=class_name.replace("_", " ").title(),
                color=CLASS_COLORS.get(class_name, TEXT_COLOR),
                alpha=0.6, s=20,
            )
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.set_title("Image Dimensions Scatter")
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.3)

        # Histogram: unique sizes
        ax2 = axes[1]
        ax2.set_facecolor("#16213E")
        sizes = [f"{w}×{h}" for w, h in zip(widths, heights)]
        from collections import Counter
        size_counts = Counter(sizes).most_common(10)
        size_labels, size_vals = zip(*size_counts) if size_counts else ([], [])
        ax2.barh(
            range(len(size_labels)), size_vals,
            color="#9B59B6", alpha=0.85, edgecolor=BACKGROUND_COLOR,
        )
        ax2.set_yticks(range(len(size_labels)))
        ax2.set_yticklabels(size_labels, fontsize=9)
        ax2.set_xlabel("Count")
        ax2.set_title("Top 10 Image Size Variants")
        ax2.xaxis.grid(True, alpha=0.3)

        plt.suptitle(
            "Image Size Analysis — Training Data",
            fontsize=13, fontweight="bold", color=TEXT_COLOR,
        )
        plt.tight_layout()
        out = self.plots_dir / "image_size_distribution.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        plt.close()
        logger.info(f"Saved: {out}")
        return out

    def run_full_eda(self, stats: DatasetStats) -> list[Path]:
        """Run all EDA plots in sequence and return saved paths."""
        logger.info("Running full EDA visualization pipeline...")
        saved = []
        train_dir = self.config.paths.train_dir

        saved.append(self.plot_class_distribution(stats))
        saved.append(self.plot_sample_grid(train_dir, "train", samples_per_class=4))
        saved.append(self.plot_intensity_distributions(train_dir))
        saved.append(self.plot_augmentation_preview(train_dir))
        saved.append(self.plot_dataset_stats_table(stats))
        saved.append(self.plot_image_size_distribution(train_dir))

        logger.info(f"EDA complete — {len(saved)} plots saved to {self.plots_dir}")
        return saved
