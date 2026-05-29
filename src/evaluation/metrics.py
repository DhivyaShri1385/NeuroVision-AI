"""
Classification Metrics — Phase 2

Computes the full suite of evaluation metrics from model predictions:

  - Accuracy
  - Per-class Precision, Recall, F1-Score
  - Macro / Weighted averages
  - ROC-AUC (One-vs-Rest, macro average)
  - Matthews Correlation Coefficient (MCC)
  - Cohen's Kappa
  - Confusion matrix

All metrics are returned in a structured dict and also logged.
Designed to be called once after test-set evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

from src.utils.config import AppConfig
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """All evaluation metrics for one model."""
    backbone_name: str
    accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    roc_auc_macro: float = 0.0
    mcc: float = 0.0           # Matthews Correlation Coefficient
    kappa: float = 0.0          # Cohen's Kappa
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    y_proba: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary_dict(self) -> dict[str, float]:
        """Return flat dict for CSV / JSON logging."""
        return {
            "accuracy":         round(self.accuracy, 4),
            "macro_precision":  round(self.macro_precision, 4),
            "macro_recall":     round(self.macro_recall, 4),
            "macro_f1":         round(self.macro_f1, 4),
            "weighted_f1":      round(self.weighted_f1, 4),
            "roc_auc_macro":    round(self.roc_auc_macro, 4),
            "mcc":              round(self.mcc, 4),
            "kappa":            round(self.kappa, 4),
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ClassificationEvaluator:
    """Runs full evaluation on test set predictions."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.class_names = config.classes.names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: tf.keras.Model,
        test_ds: tf.data.Dataset,
        backbone_name: str,
    ) -> EvaluationResult:
        """Run inference on the test set and compute all metrics.

        Args:
            model: Trained tf.keras.Model.
            test_ds: Test tf.data.Dataset (images, one_hot_labels).
            backbone_name: Name for labeling the result.

        Returns:
            EvaluationResult with all metrics filled in.
        """
        logger.info(f"Evaluating {backbone_name} on test set...")
        y_true, y_proba = self._collect_predictions(model, test_ds)
        return self.compute_metrics(y_true, y_proba, backbone_name)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        backbone_name: str = "model",
    ) -> EvaluationResult:
        """Compute all metrics from ground truth and predicted probabilities.

        Args:
            y_true:  Integer class labels, shape (N,).
            y_proba: Softmax probabilities, shape (N, num_classes).
            backbone_name: Label for the result.

        Returns:
            Fully populated EvaluationResult.
        """
        y_pred = np.argmax(y_proba, axis=1)
        num_classes = len(self.class_names)

        # Basic accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Sklearn classification report → per-class P/R/F1
        report: dict[str, Any] = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        per_class: dict[str, dict[str, float]] = {}
        for cls in self.class_names:
            per_class[cls] = {
                "precision": report[cls]["precision"],
                "recall":    report[cls]["recall"],
                "f1":        report[cls]["f1-score"],
                "support":   report[cls]["support"],
            }

        macro_precision = report["macro avg"]["precision"]
        macro_recall    = report["macro avg"]["recall"]
        macro_f1        = report["macro avg"]["f1-score"]
        weighted_f1     = report["weighted avg"]["f1-score"]

        # ROC-AUC (one-vs-rest, macro)
        try:
            roc_auc = roc_auc_score(
                y_true, y_proba,
                multi_class="ovr",
                average="macro",
            )
        except ValueError as e:
            logger.warning(f"ROC-AUC computation failed: {e}")
            roc_auc = 0.0

        # MCC and Kappa
        mcc   = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        result = EvaluationResult(
            backbone_name=backbone_name,
            accuracy=accuracy,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            roc_auc_macro=roc_auc,
            mcc=mcc,
            kappa=kappa,
            per_class_metrics=per_class,
            confusion_matrix=cm,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        self._log_result(result)
        return result

    def compare_models(self, results: list[EvaluationResult]) -> None:
        """Log a side-by-side comparison table of multiple models."""
        logger.info("\n" + "=" * 70)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 70)
        header = f"{'Model':<20} {'Accuracy':>9} {'F1 Macro':>9} {'ROC-AUC':>9} {'MCC':>9}"
        logger.info(header)
        logger.info("-" * 70)
        for r in sorted(results, key=lambda x: x.accuracy, reverse=True):
            logger.info(
                f"{r.backbone_name:<20} "
                f"{r.accuracy:>9.4f} "
                f"{r.macro_f1:>9.4f} "
                f"{r.roc_auc_macro:>9.4f} "
                f"{r.mcc:>9.4f}"
            )
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_predictions(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference and collect (y_true, y_proba) arrays."""
        all_true, all_proba = [], []

        for batch_images, batch_labels in dataset:
            proba = model.predict(batch_images, verbose=0)
            # batch_labels are one-hot encoded
            true_cls = np.argmax(batch_labels.numpy(), axis=1)
            all_true.append(true_cls)
            all_proba.append(proba)

        y_true  = np.concatenate(all_true,  axis=0)
        y_proba = np.concatenate(all_proba, axis=0)
        logger.debug(f"Collected {len(y_true)} predictions")
        return y_true, y_proba

    def _log_result(self, result: EvaluationResult) -> None:
        """Pretty-print evaluation results."""
        logger.info("\n" + "=" * 60)
        logger.info(f"EVALUATION RESULTS — {result.backbone_name}")
        logger.info("=" * 60)
        logger.info(f"  Accuracy         : {result.accuracy:.4f}  ({result.accuracy * 100:.2f}%)")
        logger.info(f"  Macro Precision  : {result.macro_precision:.4f}")
        logger.info(f"  Macro Recall     : {result.macro_recall:.4f}")
        logger.info(f"  Macro F1         : {result.macro_f1:.4f}")
        logger.info(f"  Weighted F1      : {result.weighted_f1:.4f}")
        logger.info(f"  ROC-AUC (macro)  : {result.roc_auc_macro:.4f}")
        logger.info(f"  MCC              : {result.mcc:.4f}")
        logger.info(f"  Cohen's Kappa    : {result.kappa:.4f}")
        logger.info("\n  Per-Class Metrics:")
        logger.info(f"  {'Class':<14} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>8}")
        logger.info("  " + "-" * 42)
        for cls, m in result.per_class_metrics.items():
            logger.info(
                f"  {cls:<14} "
                f"{m['precision']:>7.4f} "
                f"{m['recall']:>7.4f} "
                f"{m['f1']:>7.4f} "
                f"{int(m['support']):>8}"
            )
        logger.info("=" * 60)
