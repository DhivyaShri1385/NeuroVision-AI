"""
Classifier Model Builders — Phase 2

Builds three CNN classifiers using ImageNet pretrained backbones:
  - EfficientNetB3  (primary — best accuracy/param trade-off)
  - ResNet50        (ensemble member — different inductive bias)
  - DenseNet121     (ensemble member — dense skip connections)

Architecture pattern (same for all three):
  backbone (frozen in stage 1)
      ↓
  GlobalAveragePooling2D
      ↓
  BatchNormalization
      ↓
  Dense(512, relu) + L2 regularization
      ↓
  Dropout(p)
      ↓
  Dense(num_classes, softmax)

All models are returned UN-COMPILED — compilation happens in the trainer
so the learning rate can be changed between Stage 1 and Stage 2.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.applications import EfficientNetB3, ResNet50, DenseNet121

from src.utils.config import AppConfig
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Supported backbones registry
# ---------------------------------------------------------------------------

BACKBONE_REGISTRY: dict[str, callable] = {
    "EfficientNetB3": EfficientNetB3,
    "ResNet50":       ResNet50,
    "DenseNet121":    DenseNet121,
}


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_classifier(
    backbone_name: str,
    config: AppConfig,
    input_shape: tuple[int, int, int] | None = None,
) -> Model:
    """Build a transfer-learning classifier with a custom head.

    The backbone is loaded with ImageNet weights and all layers are initially
    set to non-trainable. The trainer will unfreeze layers as needed.

    Args:
        backbone_name: One of 'EfficientNetB3', 'ResNet50', 'DenseNet121'.
        config: AppConfig — reads num_classes, image_size, dropout, l2.
        input_shape: Override image shape (default: from config).

    Returns:
        Uncompiled tf.keras.Model with frozen backbone.
    """
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone: '{backbone_name}'. "
            f"Choose from: {list(BACKBONE_REGISTRY.keys())}"
        )

    h, w = input_shape[:2] if input_shape else config.preprocessing.image_size
    c = config.preprocessing.channels
    num_classes = config.classes.num_classes
    dropout_rate = config.training.dropout_rate
    l2_strength = config.training.l2_regularization

    # ------------------------------------------------------------------
    # 1. Backbone — pretrained on ImageNet, fully frozen for Stage 1
    # ------------------------------------------------------------------
    backbone_cls = BACKBONE_REGISTRY[backbone_name]

    # All backbones used as feature extractors (include_top=False).
    # Our tf.data pipeline applies ImageNet mean/std normalization,
    # so images arrive in approx [-2.1, 2.6] range — backbone receives
    # them directly without additional internal rescaling.
    backbone_kwargs = dict(
        include_top=False,
        weights="imagenet",
        input_shape=(h, w, c),
        pooling=None,
    )

    backbone = backbone_cls(**backbone_kwargs)
    backbone.trainable = False  # Frozen for Stage 1

    # ------------------------------------------------------------------
    # 2. Custom classification head
    # ------------------------------------------------------------------
    inputs = tf.keras.Input(shape=(h, w, c), name="mri_input")
    x = backbone(inputs, training=False)  # training=False keeps BN in inference mode

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=regularizers.L2(l2_strength),
        name="dense_512",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",  # Always float32 output regardless of mixed precision
        name="predictions",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name=backbone_name)

    # Log parameter counts
    total   = model.count_params()
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    logger.info(
        f"Built {backbone_name} | "
        f"total params: {total:,} | "
        f"trainable (stage 1): {trainable:,} | "
        f"frozen: {total - trainable:,}"
    )
    return model


def unfreeze_top_layers(model: Model, n_layers: int = 30) -> None:
    """Unfreeze the top N layers of the backbone for Stage 2 fine-tuning.

    Args:
        model: The classifier model (backbone is the second layer).
        n_layers: Number of layers to unfreeze from the end of the backbone.
                  30 works well for EfficientNetB3 and ResNet50.
                  Use 20 for DenseNet121.
    """
    backbone = model.layers[1]  # backbone is always at index 1
    backbone.trainable = True

    # Freeze all layers except the last n_layers
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False

    # Always keep BatchNormalization layers frozen — critical for transfer learning.
    # Unfreezing BN layers during fine-tuning on small datasets causes catastrophic
    # forgetting because the batch statistics shift away from ImageNet distribution.
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_after = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    total = model.count_params()
    logger.info(
        f"Unfroze top {n_layers} layers of {backbone.name} | "
        f"trainable (stage 2): {trainable_after:,} / {total:,}"
    )


def get_last_conv_layer_name(backbone_name: str) -> str:
    """Return the name of the last convolutional layer for Grad-CAM (Phase 4)."""
    layer_map = {
        "EfficientNetB3": "top_conv",
        "ResNet50":        "conv5_block3_out",
        "DenseNet121":     "relu",
    }
    return layer_map.get(backbone_name, "top_conv")


class ModelFactory:
    """Static factory — creates any supported classifier by name."""

    @staticmethod
    def create(backbone_name: str, config: AppConfig) -> Model:
        return build_classifier(backbone_name, config)

    @staticmethod
    def create_all(config: AppConfig) -> dict[str, Model]:
        """Build all ensemble members."""
        return {
            name: build_classifier(name, config)
            for name in config.model["ensemble"]
        }
