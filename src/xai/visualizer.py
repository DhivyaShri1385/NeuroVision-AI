"""
XAI Visualizer — Phase 4

Produces publication-quality explainability plots:

  plot_gradcam_overlay()       — Single image: MRI | Grad-CAM | blended overlay
  plot_saliency_overlay()      — Single image: MRI | Saliency | blended overlay
  plot_xai_comparison()        — Side-by-side: original | Grad-CAM | SmoothGrad | IG
  plot_class_activation_grid() — Grid of Grad-CAM maps for each predicted class
  plot_xai_summary()           — Multi-row grid for a batch of test images

All plots:
  - Use the dark theme (matches Phase 2 / Phase 3 visualisers).
  - Are saved to <plots_dir>/xai/.
  - Return the Path to the saved PNG.

Colourmap:
  Heatmaps rendered with 'jet' (classic saliency) or 'plasma' (Grad-CAM).
  Overlay alpha controlled via config.xai.overlay_alpha (default 0.45).
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2

from src.utils.config import AppConfig
from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Dark-theme constants (identical to segmentation/visualizer.py)
# ---------------------------------------------------------------------------
_DARK_BG = "#0d1117"
_CARD_BG = "#161b22"
_TEXT    = "#c9d1d9"
_GRID    = "#21262d"
_ACCENT  = "#58a6ff"
_GREEN   = "#3fb950"
_ORANGE  = "#d29922"
_RED     = "#f85149"

_CMAP_GRADCAM  = "jet"
_CMAP_SALIENCY = "hot"


def _apply_dark(fig: plt.Figure, axes) -> None:
    fig.patch.set_facecolor(_DARK_BG)
    for ax in np.array(axes).ravel():
        ax.set_facecolor(_CARD_BG)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_TEXT)
        ax.title.set_color(_TEXT)


# ---------------------------------------------------------------------------
# XAIVisualizer
# ---------------------------------------------------------------------------

class XAIVisualizer:
    """Generates and saves XAI explanation plots.

    Args:
        config: AppConfig — uses xai.overlay_alpha and paths.plots_dir.
    """

    def __init__(self, config: AppConfig):
        self.config     = config
        self.class_names = config.classes.names

        xai_cfg         = getattr(config, "xai", None)
        self.alpha      = getattr(xai_cfg, "overlay_alpha", 0.45)
        self.cmap       = getattr(xai_cfg, "colormap", "jet")

        self.plots_dir  = config.paths.plots_dir / "xai"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"XAIVisualizer | output -> {self.plots_dir}")

    # ------------------------------------------------------------------
    # 1. Single-image Grad-CAM overlay
    # ------------------------------------------------------------------

    def plot_gradcam_overlay(
        self,
        image:     np.ndarray,
        heatmap:   np.ndarray,
        true_label: str | None,
        pred_label: str,
        pred_prob:  float,
        filename:  str = "gradcam_overlay.png",
    ) -> Path:
        """Three-panel: original MRI | Grad-CAM | blended overlay.

        Args:
            image:      (H, W, C) float32 [0, 1] or [0, 255].
            heatmap:    (H, W) float32 Grad-CAM heatmap in [0, 1].
            true_label: Ground-truth class name (or None if unknown).
            pred_label: Predicted class name.
            pred_prob:  Predicted class probability.
            filename:   Output file name.

        Returns:
            Path to saved PNG.
        """
        img_disp  = self._to_display(image)
        overlay   = self._blend_heatmap(img_disp, heatmap, alpha=self.alpha)
        hmap_rgb  = self._heatmap_to_rgb(heatmap, self.cmap)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        _apply_dark(fig, axes)

        axes[0].imshow(img_disp, cmap="gray" if img_disp.ndim == 2 else None)
        axes[0].set_title("MRI Input", color=_TEXT)

        axes[1].imshow(hmap_rgb)
        axes[1].set_title("Grad-CAM", color=_TEXT)

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", color=_TEXT)

        for ax in axes:
            ax.axis("off")

        label_line = f"Pred: {pred_label} ({pred_prob:.1%})"
        if true_label:
            correct = true_label == pred_label
            color   = _GREEN if correct else _RED
            label_line += f"  |  GT: {true_label}"
        else:
            color = _TEXT

        fig.suptitle(label_line, color=color, fontsize=12)
        plt.tight_layout()

        out = self.plots_dir / filename
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
        plt.close(fig)
        logger.info(f"Saved Grad-CAM overlay -> {out}")
        return out

    # ------------------------------------------------------------------
    # 2. XAI comparison: Grad-CAM | SmoothGrad | IG
    # ------------------------------------------------------------------

    def plot_xai_comparison(
        self,
        image:       np.ndarray,
        gradcam_map: np.ndarray,
        saliency_map: np.ndarray,
        ig_map:      np.ndarray | None,
        pred_label:  str,
        pred_prob:   float,
        true_label:  str | None = None,
        filename:    str = "xai_comparison.png",
    ) -> Path:
        """Side-by-side comparison of all three XAI methods.

        Columns: Original | Grad-CAM overlay | SmoothGrad overlay | IG overlay (optional)
        """
        n_cols = 4 if ig_map is not None else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        _apply_dark(fig, axes)

        img_disp = self._to_display(image)

        axes[0].imshow(img_disp, cmap="gray" if img_disp.ndim == 2 else None)
        axes[0].set_title("Original MRI", color=_TEXT)

        gcam_overlay = self._blend_heatmap(img_disp, gradcam_map, alpha=self.alpha)
        axes[1].imshow(gcam_overlay)
        axes[1].set_title("Grad-CAM", color=_TEXT)

        sal_overlay = self._blend_heatmap(img_disp, saliency_map, alpha=self.alpha,
                                          cmap=_CMAP_SALIENCY)
        axes[2].imshow(sal_overlay)
        axes[2].set_title("SmoothGrad", color=_TEXT)

        if ig_map is not None:
            ig_overlay = self._blend_heatmap(img_disp, ig_map, alpha=self.alpha,
                                              cmap="plasma")
            axes[3].imshow(ig_overlay)
            axes[3].set_title("Integ. Grads", color=_TEXT)

        for ax in axes:
            ax.axis("off")

        label_line = f"Pred: {pred_label} ({pred_prob:.1%})"
        if true_label:
            correct = true_label == pred_label
            color   = _GREEN if correct else _RED
            label_line += f"  |  GT: {true_label}"
        else:
            color = _TEXT

        fig.suptitle(label_line, color=color, fontsize=12)
        plt.tight_layout()

        out = self.plots_dir / filename
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
        plt.close(fig)
        logger.info(f"Saved XAI comparison -> {out}")
        return out

    # ------------------------------------------------------------------
    # 3. Class activation grid (Grad-CAM for each class)
    # ------------------------------------------------------------------

    def plot_class_activation_grid(
        self,
        image:    np.ndarray,
        heatmaps: dict[str, np.ndarray],
        pred_label: str,
        filename: str = "class_activation_grid.png",
    ) -> Path:
        """Grid of Grad-CAM maps — one column per class.

        Args:
            image:     (H, W, C) input MRI.
            heatmaps:  Dict mapping class_name -> heatmap (H, W).
            pred_label: Predicted class name.
            filename:  Output filename.

        Returns:
            Path to saved PNG.
        """
        classes   = list(heatmaps.keys())
        n         = len(classes)
        fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
        if n == 1:
            axes = axes[:, np.newaxis]
        _apply_dark(fig, axes)

        img_disp = self._to_display(image)

        for col, cls in enumerate(classes):
            hm       = heatmaps[cls]
            overlay  = self._blend_heatmap(img_disp, hm, alpha=self.alpha)
            hm_rgb   = self._heatmap_to_rgb(hm, self.cmap)

            color    = _GREEN if cls == pred_label else _TEXT
            axes[0, col].imshow(hm_rgb)
            axes[0, col].set_title(cls, color=color, fontsize=10)
            axes[0, col].axis("off")

            axes[1, col].imshow(overlay)
            axes[1, col].axis("off")

        axes[0, 0].set_ylabel("Heatmap", color=_TEXT)
        axes[1, 0].set_ylabel("Overlay", color=_TEXT)

        fig.suptitle(
            f"Class Activation Maps — predicted: {pred_label}",
            color=_TEXT, fontsize=13,
        )
        plt.tight_layout()

        out = self.plots_dir / filename
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
        plt.close(fig)
        logger.info(f"Saved class activation grid -> {out}")
        return out

    # ------------------------------------------------------------------
    # 4. Multi-image XAI summary grid
    # ------------------------------------------------------------------

    def plot_xai_summary(
        self,
        images:      np.ndarray,
        gradcam_maps: list[np.ndarray],
        pred_labels: list[str],
        pred_probs:  list[float],
        true_labels: list[str] | None = None,
        n_samples:   int = 8,
        filename:    str = "xai_summary_grid.png",
    ) -> Path:
        """Grid of N rows: MRI | Grad-CAM | overlay, one row per sample.

        Args:
            images:       (N, H, W, C) float32.
            gradcam_maps: List of (H, W) Grad-CAM heatmaps.
            pred_labels:  Predicted class names.
            pred_probs:   Predicted probabilities.
            true_labels:  Ground-truth labels (optional).
            n_samples:    Max rows to show.
            filename:     Output filename.

        Returns:
            Path to saved PNG.
        """
        n      = min(n_samples, len(images))
        idxs   = random.sample(range(len(images)), n) if len(images) > n else list(range(n))

        fig    = plt.figure(figsize=(12, 4 * n))
        gs     = gridspec.GridSpec(n, 3, figure=fig, hspace=0.3, wspace=0.05)
        fig.patch.set_facecolor(_DARK_BG)

        col_titles = ["MRI Input", "Grad-CAM", "Overlay"]

        for row, idx in enumerate(idxs):
            img_disp = self._to_display(images[idx])
            hm       = gradcam_maps[idx]
            overlay  = self._blend_heatmap(img_disp, hm, alpha=self.alpha)
            hm_rgb   = self._heatmap_to_rgb(hm, self.cmap)

            for col, (data, cmap) in enumerate([
                (img_disp, "gray" if img_disp.ndim == 2 else None),
                (hm_rgb,   None),
                (overlay,  None),
            ]):
                ax = fig.add_subplot(gs[row, col])
                ax.set_facecolor(_CARD_BG)
                ax.imshow(data, cmap=cmap)
                ax.axis("off")
                if row == 0:
                    ax.set_title(col_titles[col], color=_TEXT, fontsize=10)

            # Row label
            pred  = pred_labels[idx]
            prob  = pred_probs[idx]
            gt    = true_labels[idx] if true_labels else None
            color = (_GREEN if gt == pred else _RED) if gt else _TEXT
            fig.text(
                0.01, 1 - (row + 0.5) / n,
                f"{pred}\n{prob:.1%}" + (f"\nGT:{gt}" if gt else ""),
                va="center", ha="left", color=color, fontsize=8,
                transform=fig.transFigure,
            )

        fig.suptitle("XAI — Grad-CAM Summary", color=_TEXT, fontsize=13, y=1.01)

        out = self.plots_dir / filename
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_DARK_BG)
        plt.close(fig)
        logger.info(f"Saved XAI summary grid -> {out}")
        return out

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_display(image: np.ndarray) -> np.ndarray:
        """Convert image to uint8-range float for display.

        Handles:
          - (H, W, C) or (H, W) arrays
          - [0,1] or [0,255] or arbitrary float ranges
        """
        img = image.squeeze()
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        # If values are in roughly [0, 1], keep; otherwise normalise
        if img.max() > 2.0:
            img = img / 255.0
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def _heatmap_to_rgb(heatmap: np.ndarray, cmap: str = "jet") -> np.ndarray:
        """Convert (H, W) float [0,1] heatmap to (H, W, 3) RGB via colormap."""
        import matplotlib.cm as cm
        colormap = cm.get_cmap(cmap)
        rgb = colormap(heatmap)[..., :3]   # (H, W, 3) float
        return rgb.astype(np.float32)

    @staticmethod
    def _blend_heatmap(
        img:     np.ndarray,
        heatmap: np.ndarray,
        alpha:   float = 0.45,
        cmap:    str   = "jet",
    ) -> np.ndarray:
        """Blend heatmap over grayscale/RGB image.

        Args:
            img:     (H, W) or (H, W, 3) float [0,1].
            heatmap: (H, W) float [0,1].
            alpha:   Heatmap opacity.
            cmap:    Matplotlib colormap for heatmap.

        Returns:
            (H, W, 3) float [0,1] blended image.
        """
        import matplotlib.cm as cm

        # Ensure img is RGB
        if img.ndim == 2:
            rgb = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            rgb = np.concatenate([img, img, img], axis=-1)
        else:
            rgb = img.copy()

        # Resize heatmap to match image if needed
        if heatmap.shape != rgb.shape[:2]:
            hm_resized = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
        else:
            hm_resized = heatmap

        colormap   = cm.get_cmap(cmap)
        heat_rgb   = colormap(hm_resized)[..., :3].astype(np.float32)

        blended    = (1 - alpha) * rgb + alpha * heat_rgb
        return np.clip(blended, 0, 1).astype(np.float32)
