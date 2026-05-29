"""
Segmentation Visualizer — Phase 3

Produces publication-quality plots for the segmentation pipeline:

  plot_training_history()    — Dice, IoU, loss curves per epoch
  plot_mask_overlays()       — side-by-side: image | pseudo-mask | prediction
  plot_metrics_bar()         — bar chart comparing Dice/IoU/PixAcc at test time
  plot_sample_predictions()  — grid of N random test samples

All plots use a dark theme matching Phase 2 visualizer for visual consistency.
All figures are saved to config.paths.plots_dir / "segmentation/".
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf

from src.utils.config import AppConfig
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_DARK_BG    = "#0d1117"
_CARD_BG    = "#161b22"
_ACCENT     = "#58a6ff"
_GREEN      = "#3fb950"
_ORANGE     = "#d29922"
_RED        = "#f85149"
_PURPLE     = "#bc8cff"
_TEXT       = "#c9d1d9"
_GRID       = "#21262d"

_CMAP_MASK  = "plasma"   # colourmap for predicted probability masks


def _apply_dark_theme(fig: plt.Figure, ax_or_axes) -> None:
    """Apply consistent dark theme to figure and axes."""
    fig.patch.set_facecolor(_DARK_BG)
    axes = ax_or_axes if hasattr(ax_or_axes, "__iter__") else [ax_or_axes]
    for ax in np.array(axes).ravel():
        ax.set_facecolor(_CARD_BG)
        ax.tick_params(colors=_TEXT)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.title.set_color(_TEXT)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)


# ---------------------------------------------------------------------------
# Segmentation Visualizer
# ---------------------------------------------------------------------------

class SegmentationVisualizer:
    """Generates and saves segmentation plots.

    Args:
        config: AppConfig for output paths.
    """

    def __init__(self, config: AppConfig):
        self.config   = config
        self.plots_dir = config.paths.plots_dir / "segmentation"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SegmentationVisualizer | output -> {self.plots_dir}")

    # ------------------------------------------------------------------
    # 1. Training History
    # ------------------------------------------------------------------

    def plot_training_history(
        self,
        history: dict[str, list[float]],
        arch_name: str = "AttentionUNet",
    ) -> Path:
        """Plot loss + Dice + IoU + pixel accuracy over epochs.

        Args:
            history:   Dict from trainer.train() — keys like 'loss', 'val_loss',
                       'dice', 'val_dice', 'iou', 'val_iou', 'pixel_acc',
                       'val_pixel_acc', optionally 'lr'.
            arch_name: Model name for plot title and filename.

        Returns:
            Path to saved figure.
        """
        epochs = list(range(1, len(history.get("loss", [])) + 1))
        has_lr = "lr" in history

        n_rows = 3 if not has_lr else 4
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows))
        _apply_dark_theme(fig, axes)

        def _plot(ax, train_key, val_key, ylabel, color_t, color_v):
            if train_key in history:
                ax.plot(epochs, history[train_key], color=color_t,
                        linewidth=1.8, label=f"Train {ylabel}")
            if val_key in history:
                ax.plot(epochs, history[val_key], color=color_v,
                        linewidth=1.8, linestyle="--", label=f"Val {ylabel}")
            ax.set_ylabel(ylabel, color=_TEXT)
            ax.legend(framealpha=0.3, labelcolor=_TEXT, facecolor=_CARD_BG)

        _plot(axes[0], "loss",      "val_loss",      "Loss",       _RED,    _ORANGE)
        _plot(axes[1], "dice",      "val_dice",      "Dice",       _GREEN,  _ACCENT)
        _plot(axes[2], "iou",       "val_iou",       "IoU",        _PURPLE, _ACCENT)

        if has_lr:
            axes[3].plot(epochs, history["lr"], color=_ORANGE, linewidth=1.5)
            axes[3].set_ylabel("LR", color=_TEXT)
            axes[3].set_yscale("log")

        axes[-1].set_xlabel("Epoch", color=_TEXT)
        fig.suptitle(f"{arch_name} — Training History", color=_TEXT, fontsize=14)
        plt.tight_layout()

        out = self.plots_dir / f"{arch_name.lower()}_history.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
        plt.close(fig)
        logger.info(f"Saved history plot -> {out}")
        return out

    # ------------------------------------------------------------------
    # 2. Mask Overlays
    # ------------------------------------------------------------------

    def plot_mask_overlays(
        self,
        images:    np.ndarray,
        true_masks: np.ndarray,
        pred_masks: np.ndarray,
        n_samples:  int = 6,
        arch_name:  str = "AttentionUNet",
    ) -> Path:
        """Side-by-side: original MRI | ground-truth mask | predicted mask.

        Args:
            images:     (N, H, W, 1) float32 [0,1].
            true_masks: (N, H, W, 1) float32 binary {0,1}.
            pred_masks: (N, H, W, 1) float32 probabilities.
            n_samples:  Number of rows in the plot.
            arch_name:  For title / filename.

        Returns:
            Path to saved figure.
        """
        n = min(n_samples, len(images))
        indices = random.sample(range(len(images)), n)

        fig, axes = plt.subplots(n, 3, figsize=(9, 3.5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        _apply_dark_theme(fig, axes)

        col_titles = ["MRI Input", "Pseudo-Mask (GT)", "Prediction"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, color=_TEXT, fontsize=11)

        for row, idx in enumerate(indices):
            img  = images[idx].squeeze()
            gt   = true_masks[idx].squeeze()
            pred = pred_masks[idx].squeeze()

            axes[row, 0].imshow(img,  cmap="gray", vmin=0, vmax=1)
            axes[row, 1].imshow(gt,   cmap="gray", vmin=0, vmax=1)
            axes[row, 2].imshow(pred, cmap=_CMAP_MASK, vmin=0, vmax=1)

            for col in range(3):
                axes[row, col].axis("off")

        fig.suptitle(f"{arch_name} — Mask Predictions", color=_TEXT, fontsize=13)
        plt.tight_layout()

        out = self.plots_dir / f"{arch_name.lower()}_mask_overlays.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
        plt.close(fig)
        logger.info(f"Saved mask overlay plot -> {out}")
        return out

    # ------------------------------------------------------------------
    # 3. Metrics Bar Chart
    # ------------------------------------------------------------------

    def plot_metrics_bar(
        self,
        results:   dict[str, float],
        arch_name: str = "AttentionUNet",
    ) -> Path:
        """Horizontal bar chart of evaluation metrics.

        Args:
            results:   Dict from SegmentationTrainer.evaluate() or evaluate_masks().
            arch_name: For title and filename.

        Returns:
            Path to saved figure.
        """
        keys   = ["dice", "iou", "precision", "recall", "f1", "pixel_acc"]
        labels = ["Dice", "IoU", "Precision", "Recall", "F1", "Pixel Acc"]
        values = [results.get(k, 0.0) for k in keys]
        colors = [_GREEN, _ACCENT, _PURPLE, _ORANGE, _RED, _TEXT[:7]]

        fig, ax = plt.subplots(figsize=(8, 4))
        _apply_dark_theme(fig, ax)

        bars = ax.barh(labels, values, color=colors, height=0.55)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Score", color=_TEXT)
        ax.set_title(f"{arch_name} — Test Metrics", color=_TEXT, fontsize=13)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", color=_TEXT, fontsize=9,
            )

        plt.tight_layout()
        out = self.plots_dir / f"{arch_name.lower()}_metrics_bar.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
        plt.close(fig)
        logger.info(f"Saved metrics bar -> {out}")
        return out

    # ------------------------------------------------------------------
    # 4. Sample Predictions Grid
    # ------------------------------------------------------------------

    def plot_sample_predictions(
        self,
        model:     tf.keras.Model,
        dataset:   tf.data.Dataset,
        n_samples: int = 8,
        arch_name: str = "AttentionUNet",
    ) -> Path:
        """Pull one batch from dataset and plot a grid of predictions.

        Columns: Input | Ground Truth | Prediction | Overlay

        Args:
            model:     Trained tf.keras.Model.
            dataset:   tf.data.Dataset (batched, unbounded).
            n_samples: Max number of samples to show.
            arch_name: For title and filename.

        Returns:
            Path to saved figure.
        """
        # Collect enough samples
        imgs_list:  list[np.ndarray] = []
        masks_list: list[np.ndarray] = []

        for imgs, masks in dataset:
            imgs_list.append(imgs.numpy())
            masks_list.append(masks.numpy())
            if sum(x.shape[0] for x in imgs_list) >= n_samples:
                break

        imgs_all  = np.concatenate(imgs_list, axis=0)[:n_samples]
        masks_all = np.concatenate(masks_list, axis=0)[:n_samples]
        preds_all = model.predict(imgs_all, verbose=0)

        n   = len(imgs_all)
        fig = plt.figure(figsize=(14, 3.5 * n))
        gs  = gridspec.GridSpec(n, 4, figure=fig, hspace=0.05, wspace=0.05)
        fig.patch.set_facecolor(_DARK_BG)

        col_titles = ["MRI Input", "Ground Truth", "Predicted Prob.", "Overlay"]

        for row in range(n):
            img  = imgs_all[row].squeeze()
            gt   = masks_all[row].squeeze()
            pred = preds_all[row].squeeze()
            # Overlay: color prediction on grayscale input
            overlay = _make_overlay(img, pred)

            for col, (data, cmap, vmin, vmax) in enumerate([
                (img,     "gray",    0, 1),
                (gt,      "gray",    0, 1),
                (pred,    _CMAP_MASK, 0, 1),
                (overlay, None,      None, None),
            ]):
                ax = fig.add_subplot(gs[row, col])
                ax.set_facecolor(_CARD_BG)
                if cmap:
                    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                else:
                    ax.imshow(data)
                ax.axis("off")
                if row == 0:
                    ax.set_title(col_titles[col], color=_TEXT, fontsize=10)

        fig.suptitle(f"{arch_name} — Sample Predictions", color=_TEXT,
                     fontsize=13, y=1.01)

        out = self.plots_dir / f"{arch_name.lower()}_sample_predictions.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
        plt.close(fig)
        logger.info(f"Saved sample predictions -> {out}")
        return out


# ---------------------------------------------------------------------------
# Helper: colourful overlay
# ---------------------------------------------------------------------------

def _make_overlay(
    img: np.ndarray,
    pred: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a predicted mask probability over a grayscale image (RGB output)."""
    # Convert grayscale to RGB
    rgb = np.stack([img, img, img], axis=-1)
    # Apply jet colormap to prediction
    import matplotlib.cm as cm
    heat = cm.jet(pred)[..., :3]
    overlay = (1 - alpha) * rgb + alpha * heat
    return np.clip(overlay, 0, 1)
