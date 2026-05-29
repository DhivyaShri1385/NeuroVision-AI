"""
Ensemble Predictor — Phase 2

Combines EfficientNetB3 + ResNet50 + DenseNet121 via soft voting
(average of class probability vectors).

Soft voting outperforms hard voting because it leverages calibrated
confidence scores rather than just the winning class per model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.utils.config import AppConfig
from src.utils.logger import logger


@dataclass
class EnsemblePrediction:
    """Structured output from ensemble inference."""
    probabilities: np.ndarray          # (N, num_classes) averaged proba
    predicted_class_idx: np.ndarray    # (N,) argmax
    predicted_class_names: list[str]   # (N,) human-readable labels
    per_model_probabilities: dict[str, np.ndarray]  # raw per-model outputs
    confidence: np.ndarray             # (N,) max probability


class EnsemblePredictor:
    """Loads saved models and combines predictions via soft voting.

    Usage:
        predictor = EnsemblePredictor(config)
        predictor.load_models({
            "EfficientNetB3": "models/EfficientNetB3_best.keras",
            "ResNet50":        "models/ResNet50_best.keras",
            "DenseNet121":     "models/DenseNet121_best.keras",
        })
        result = predictor.predict(image_batch)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.class_names = config.classes.names
        self._models: dict[str, tf.keras.Model] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_models(self, model_paths: dict[str, str | Path]) -> None:
        """Load .keras model files from disk.

        Args:
            model_paths: Dict mapping backbone name to saved model path.
                         Example: {"EfficientNetB3": "models/eff_best.keras"}
        """
        for name, path in model_paths.items():
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            logger.info(f"Loading ensemble member: {name} from {path}")
            self._models[name] = tf.keras.models.load_model(str(path))
            logger.info(f"  Loaded {name} OK")

    def predict(self, images: np.ndarray | tf.Tensor) -> EnsemblePrediction:
        """Run inference and return soft-voted ensemble predictions.

        Args:
            images: Float32 array of shape (N, H, W, 3), ImageNet-normalized.

        Returns:
            EnsemblePrediction dataclass.
        """
        if not self._models:
            raise RuntimeError("No models loaded. Call load_models() first.")

        per_model = {}
        for name, model in self._models.items():
            proba = model.predict(images, verbose=0)
            per_model[name] = proba
            logger.debug(f"  {name} inference done, shape={proba.shape}")

        # Soft voting — uniform average across all models
        avg_proba = np.mean(list(per_model.values()), axis=0)  # (N, num_classes)
        pred_idx = np.argmax(avg_proba, axis=1)
        pred_names = [self.class_names[i] for i in pred_idx]
        confidence = np.max(avg_proba, axis=1)

        return EnsemblePrediction(
            probabilities=avg_proba,
            predicted_class_idx=pred_idx,
            predicted_class_names=pred_names,
            per_model_probabilities=per_model,
            confidence=confidence,
        )

    def predict_single(self, image: np.ndarray) -> EnsemblePrediction:
        """Convenience wrapper for single-image inference.

        Args:
            image: Float32 array (H, W, 3).

        Returns:
            EnsemblePrediction with N=1.
        """
        batch = np.expand_dims(image, axis=0)
        return self.predict(batch)

    @property
    def model_names(self) -> list[str]:
        return list(self._models.keys())

    @property
    def is_ready(self) -> bool:
        return len(self._models) > 0
