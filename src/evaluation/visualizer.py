"""
Evaluation Visualizer — Phase 2

Generates publication-ready evaluation plots:

  1. Training history — loss + accuracy curves (stage 1 + stage 2)
  2. Confusion matrix — normalized with per-class count annotations
  3. ROC curves — one curve per class + macro average
  4. Per-class metrics bar chart — Precision / Recall / F1 grouped bars
  5. Model comparison radar chart — compares all ensemble members

All plots saved as high-DPI PNGs to outputs/plots/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.evaluation.metrics import EvaluationResult
from src.training.trainer import TrainingResult
from src.utils.config import AppConfig

matplotlib.use("Agg")

# Consistent dark theme (same as Phase 1 EDA)
BACKGROUND   = "#1A1A2E"
AXES_BG      = "#16213E"
TEXT_COLOR   = "#ECF0F1"
GRID_COLOR   = "#2C3E50"
CLASS_COLORS = {
    "glioma":      "#E74C3C",
    "meningioma":  "#3498DB",
    "no_tumor":    "#2ECC71",
    "pituitary":   "#F39C12",
}
METRIC_COLORS = {"precision": "#9B59B6", "recall": "#1ABC9C", "f1": "#E67E22"}
DPI = 150


def _dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor":   AXES_BG,
        "axes.edgecolor":   GRID_COLOR,
        "axes.labelcolor":  TEXT_COLOR,
        "axes.titlecolor":  TEXT_COLOR,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
        "text.color":       TEXT_COLOR,
        "grid.color":       GRID_COLOR,
        "grid.alpha":       0.3,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   13,
        "legend.facecolor": AXES_BG,
        "legend.edgecolor": GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
    })


class EvaluationVisualizer:
    """Generates and saves all Phase 2 evaluation plots."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.plots_dir = config.paths.plots_dir
        self.class_names = config.classes.names
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        _dark_style()

    # ------------------------------------------------------------------
    # 1. Training History
    # ------------------------------------------------------------------

    def plot_training_history(
        self, training_result: TrainingResult
    ) -> Path:
        """Loss + accuracy curves for Stage 1 + Stage 2 combined."""
        history = training_result.combined_history
        if not history:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(BACKGROUND)
        fig.suptitle(
            f"Training History — {training_result.backbone_name}",
            fontsize=14, fontweight="bold", color=TEXT_COLOR,
        )

        s1_len = len(training_result.stage1_history.get("loss", []))
        total_epochs = len(history.get("loss", []))
        epochs = range(1, total_epochs + 1)

        for ax, metric, val_metric, title in zip(
            axes,
            ["loss", "accuracy"],
            ["val_loss", "val_accuracy"],
            ["Loss", "Accuracy"],
        ):
            ax.set_facecolor(AXES_BG)
            train_vals = history.get(metric, [])
            val_vals   = history.get(val_metric, [])

            ax.plot(epochs, train_vals, color="#3498DB", linewidth=2, label="Train")
            ax.plot(epochs, val_vals,   color="#E74C3C", linewidth=2, label="Val", linestyle="--")

            # Stage boundary line
            if s1_len > 0 and s1_len < total_epochs:
                ax.axvline(
                    x=s1_len + 0.5,
                    color="#F39C12", linewidth=1.5, linestyle=":",
                    label=f"Stage 1→2 (epoch {s1_len})",
                )

            # Best val marker
            if val_metric == "val_accuracy" and val_vals:
                best_ep = int(np.argmax(val_vals)) + 1
                best_val = max(val_vals)
                ax.scatter(
                    best_ep, best_val,
                    color="#2ECC71", s=80, zorder=5,
                    label=f"Best: {best_val:.4f} @ ep{best_ep}",
                )
            elif val_metric == "val_loss" and val_vals:
                best_ep = int(np.argmin(val_vals)) + 1
                best_val = min(val_vals)
                ax.scatter(
                    best_ep, best_val,
                    color="#2ECC71", s=80, zorder=5,
                    label=f"Best: {best_val:.4f} @ ep{best_ep}",
                )

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.legend(fontsize=9)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

        plt.tight_layout()
        out = self.plots_dir / f"history_{training_result.backbone_name}.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND)
        plt.close()
        return out

    # ------------------------------------------------------------------
    # 2. Confusion Matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self, eval_result: EvaluationResult) -> Path:
        """Normalized confusion matrix with count annotations."""
        cm = eval_result.confusion_matrix
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor(BACKGROUND)
        ax.set_facecolor(AXES_BG)

        # Use a colormap that looks good on dark background
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
        cbar.set_label("Normalized Proportion", color=TEXT_COLOR)

        tick_labels = [c.replace("_", "\n").title() for c in self.class_names]
        ax.set_xticks(range(len(self.class_names)))
        ax.set_yticks(range(len(self.class_names)))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)

        # Annotate each cell with count / normalized value
        thresh = 0.5
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                color = "white" if cm_norm[i, j] > thresh else TEXT_COLOR
                ax.text(
                    j, i,
                    f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold",
                )

        ax.set_xlabel("Predicted Label", labelpad=10)
        ax.set_ylabel("True Label", labelpad=10)
        ax.set_title(
            f"Confusion Matrix — {eval_result.backbone_name}\n"
            f"Accuracy: {eval_result.accuracy:.4f}",
            pad=12, fontweight="bold",
        )

        plt.tight_layout()
        out = self.plots_dir / f"confusion_matrix_{eval_result.backbone_name}.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND)
        plt.close()
        return out

    # ------------------------------------------------------------------
    # 3. ROC Curves
    # ------------------------------------------------------------------

    def plot_roc_curves(self, eval_result: EvaluationResult) -> Path:
        """One ROC curve per class + macro average AUC."""
        n_classes = len(self.class_names)
        y_true_bin = label_binarize(eval_result.y_true, classes=list(range(n_classes)))
        y_proba    = eval_result.y_proba

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor(BACKGROUND)
        ax.set_facecolor(AXES_BG)

        fpr_all, tpr_all = [], []
        for i, cls in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            color = CLASS_COLORS.get(cls, TEXT_COLOR)
            ax.plot(
                fpr, tpr,
                color=color, linewidth=2,
                label=f"{cls.replace('_', ' ').title()} (AUC={roc_auc:.3f})",
            )

        # Macro average ROC — interpolate each class curve onto a common FPR grid
        all_fpr = np.unique(np.concatenate(fpr_all))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(
            all_fpr, mean_tpr,
            color="white", linewidth=2.5, linestyle="--",
            label=f"Macro Average (AUC={macro_auc:.3f})",
        )

        # Random classifier baseline
        ax.plot([0, 1], [0, 1], color="#7F8C8D", linewidth=1, linestyle=":", label="Random")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(
            f"ROC Curves — {eval_result.backbone_name}\n"
            f"Macro ROC-AUC: {eval_result.roc_auc_macro:.4f}",
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_axisbelow(True)

        plt.tight_layout()
        out = self.plots_dir / f"roc_curves_{eval_result.backbone_name}.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND)
        plt.close()
        return out

    # ------------------------------------------------------------------
    # 4. Per-Class Metrics Bar Chart
    # ------------------------------------------------------------------

    def plot_per_class_metrics(self, eval_result: EvaluationResult) -> Path:
        """Grouped bar chart: Precision / Recall / F1 per class."""
        metrics_list = ["precision", "recall", "f1"]
        x = np.arange(len(self.class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor(BACKGROUND)
        ax.set_facecolor(AXES_BG)

        for i, metric in enumerate(metrics_list):
            values = [
                eval_result.per_class_metrics[cls][metric]
                for cls in self.class_names
            ]
            bars = ax.bar(
                x + i * width, values, width,
                label=metric.capitalize(),
                color=METRIC_COLORS[metric],
                alpha=0.85,
                edgecolor=BACKGROUND,
                linewidth=0.8,
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR,
                )

        ax.set_xticks(x + width)
        ax.set_xticklabels(
            [c.replace("_", " ").title() for c in self.class_names],
            fontsize=10,
        )
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title(
            f"Per-Class Precision / Recall / F1 — {eval_result.backbone_name}\n"
            f"Macro F1: {eval_result.macro_f1:.4f}  |  Accuracy: {eval_result.accuracy:.4f}",
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        out = self.plots_dir / f"per_class_metrics_{eval_result.backbone_name}.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND)
        plt.close()
        return out

    # ------------------------------------------------------------------
    # 5. Model Comparison (Ensemble summary)
    # ------------------------------------------------------------------

    def plot_model_comparison(self, results: list[EvaluationResult]) -> Path:
        """Horizontal bar chart comparing all models on key metrics."""
        metrics_to_plot = ["accuracy", "macro_f1", "roc_auc_macro", "mcc"]
        labels = [r.backbone_name for r in results]
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(results)

        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(BACKGROUND)
        ax.set_facecolor(AXES_BG)
        model_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]

        for i, result in enumerate(results):
            values = [
                result.accuracy,
                result.macro_f1,
                result.roc_auc_macro,
                result.mcc,
            ]
            bars = ax.bar(
                x + i * width, values, width,
                label=result.backbone_name,
                color=model_colors[i % len(model_colors)],
                alpha=0.85,
                edgecolor=BACKGROUND,
                linewidth=0.8,
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR,
                )

        ax.set_xticks(x + width * (len(results) - 1) / 2)
        ax.set_xticklabels(
            ["Accuracy", "Macro F1", "ROC-AUC", "MCC"],
            fontsize=11,
        )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison — Ensemble Members", fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        out = self.plots_dir / "model_comparison.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND)
        plt.close()
        return out

    # ------------------------------------------------------------------
    # Convenience: run all plots for one model
    # ------------------------------------------------------------------

    def run_all(
        self,
        training_result: TrainingResult,
        eval_result: EvaluationResult,
    ) -> list[Path]:
        """Generate all plots for one trained model."""
        saved = []
        saved.append(self.plot_training_history(training_result))
        saved.append(self.plot_confusion_matrix(eval_result))
        saved.append(self.plot_roc_curves(eval_result))
        saved.append(self.plot_per_class_metrics(eval_result))
        return [p for p in saved if p is not None]
