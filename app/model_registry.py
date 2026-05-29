"""
Model Registry — Phase 5

Loads all ML models once at API startup and keeps them in memory.
Every request handler accesses models via `request.app.state.registry`
rather than reloading from disk.

Models managed:
  classifier        — Phase 2 EfficientNetB3 (or ResNet50 / DenseNet121)
  seg_model         — Phase 3 AttentionUNet (or UNet)
  gradcam           — Phase 4 GradCAM explainer (wraps classifier)
  smoothgrad        — Phase 4 SmoothGrad explainer
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path so src.* imports work when launched from app/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tensorflow as tf

from src.utils.config import AppConfig
from src.utils.logger import logger
from src.xai.gradcam  import GradCAM
from src.xai.saliency import SmoothGrad


class ModelRegistry:
    """Singleton-style model store loaded once at FastAPI startup.

    Attributes:
        classifier:   Phase 2 classifier tf.keras.Model.
        seg_model:    Phase 3 segmentation tf.keras.Model.
        gradcam:      GradCAM explainer instance.
        smoothgrad:   SmoothGrad explainer instance.
        class_names:  List of class name strings from config.
        config:       AppConfig frozen dataclass.
    """

    def __init__(self, config: AppConfig):
        self.config      = config
        self.class_names = config.classes.names
        self.classifier: tf.keras.Model | None  = None
        self.seg_model:  tf.keras.Model | None  = None
        self.gradcam:    GradCAM | None          = None
        self.smoothgrad: SmoothGrad | None       = None
        self._loaded:    list[str]               = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all available models.  Missing checkpoints are skipped with
        a warning (the corresponding endpoint will return 503)."""
        self._load_classifier()
        self._load_segmentation()
        self._init_explainers()
        logger.info(f"ModelRegistry ready | loaded={self._loaded}")

    @property
    def loaded_names(self) -> list[str]:
        return list(self._loaded)

    def classifier_ready(self) -> bool:
        return self.classifier is not None

    def seg_ready(self) -> bool:
        return self.seg_model is not None

    def xai_ready(self) -> bool:
        return self.gradcam is not None

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_classifier(self) -> None:
        models_dir  = self.config.paths.models_dir
        candidates  = [
            models_dir / "EfficientNetB3_best.keras",
            models_dir / "EfficientNetB3_final.keras",
            models_dir / "ResNet50_best.keras",
            models_dir / "DenseNet121_best.keras",
        ]
        for ckpt in candidates:
            if ckpt.exists():
                logger.info(f"Loading classifier: {ckpt.name}")
                try:
                    self.classifier = tf.keras.models.load_model(
                        str(ckpt), compile=False
                    )
                    self._loaded.append(f"classifier({ckpt.stem})")
                    logger.info(
                        f"Classifier loaded | params="
                        f"{self.classifier.count_params():,}"
                    )
                    return
                except Exception as exc:
                    logger.warning(f"Could not load {ckpt}: {exc}")

        logger.warning(
            "No classifier checkpoint found. "
            "Train first with: python scripts/run_phase2.py\n"
            "POST /classify and /explain will return 503."
        )

    def _load_segmentation(self) -> None:
        models_dir  = self.config.paths.models_dir
        candidates  = [
            models_dir / "attentionunet_best.keras",
            models_dir / "attentionunet_final.keras",
            models_dir / "unet_best.keras",
            models_dir / "unet_final.keras",
        ]
        for ckpt in candidates:
            if ckpt.exists():
                logger.info(f"Loading segmentation model: {ckpt.name}")
                try:
                    self.seg_model = tf.keras.models.load_model(
                        str(ckpt), compile=False
                    )
                    self._loaded.append(f"segmentation({ckpt.stem})")
                    logger.info(
                        f"Seg model loaded | params="
                        f"{self.seg_model.count_params():,}"
                    )
                    return
                except Exception as exc:
                    logger.warning(f"Could not load {ckpt}: {exc}")

        logger.warning(
            "No segmentation checkpoint found. "
            "Train first with: python scripts/run_phase3.py\n"
            "POST /segment will return 503."
        )

    def _init_explainers(self) -> None:
        if self.classifier is None:
            return
        try:
            self.gradcam    = GradCAM(self.classifier, self.config)
            self.smoothgrad = SmoothGrad(
                self.classifier, self.config,
                n_samples=20, noise_std=0.15,
            )
            self._loaded.append("GradCAM")
            self._loaded.append("SmoothGrad")
            logger.info("XAI explainers initialised")
        except Exception as exc:
            logger.warning(f"Could not init explainers: {exc}")
