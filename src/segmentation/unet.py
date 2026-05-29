"""
Standard U-Net — Phase 3

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015. https://arxiv.org/abs/1505.04597

Architecture overview:
  Input (256×256×1)
      │
  Encoder (4 levels, each: Conv3×3→BN→ReLU × 2 → MaxPool2×2)
      │          filters: [64, 128, 256, 512]
  Bottleneck (Conv3×3→BN→ReLU × 2)
      │          filters: 1024
  Decoder (4 levels, each: TransposedConv2×2 → Concat skip → Conv3×3×2)
      │
  Output Conv1×1 → Sigmoid → (256×256×1) binary mask

All models returned UN-COMPILED. Compilation in trainer so loss can vary.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model

from src.utils.config import AppConfig, SegmentationConfig
from src.utils.logger import logger


def _conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    batch_norm: bool = True,
    dropout: float = 0.0,
    name_prefix: str = "",
) -> tf.Tensor:
    """Two consecutive Conv2D → BN → ReLU layers (standard U-Net block)."""
    x = layers.Conv2D(
        filters, kernel_size, padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv1",
    )(x)
    if batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(
        filters, kernel_size, padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv2",
    )(x)
    if batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)

    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name_prefix}_drop")(x)
    return x


def _encoder_block(
    x: tf.Tensor,
    filters: int,
    batch_norm: bool,
    dropout: float,
    level: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Encoder block: conv_block + MaxPool. Returns (pooled, skip)."""
    skip = _conv_block(x, filters, batch_norm=batch_norm, dropout=dropout,
                       name_prefix=f"enc{level}")
    pooled = layers.MaxPooling2D(2, name=f"enc{level}_pool")(skip)
    return pooled, skip


def _decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    batch_norm: bool,
    dropout: float,
    level: int,
) -> tf.Tensor:
    """Decoder block: UpSample → Concat skip → conv_block."""
    x = layers.Conv2DTranspose(
        filters, kernel_size=2, strides=2, padding="same",
        name=f"dec{level}_upsample",
    )(x)
    x = layers.Concatenate(name=f"dec{level}_concat")([x, skip])
    x = _conv_block(x, filters, batch_norm=batch_norm, dropout=dropout,
                    name_prefix=f"dec{level}")
    return x


def build_unet(config: AppConfig) -> Model:
    """Build and return a standard U-Net model.

    Args:
        config: AppConfig with segmentation sub-config.

    Returns:
        Uncompiled tf.keras.Model.
    """
    seg = config.segmentation
    h, w = seg.input_size
    c    = seg.channels
    filters = seg.filters          # e.g. [64, 128, 256, 512]
    bn_filters = seg.bottleneck_filters
    bn      = seg.batch_norm
    drop    = seg.dropout

    inputs = tf.keras.Input(shape=(h, w, c), name="mri_input")

    # ---- Encoder ----
    x = inputs
    skips = []
    for level, f in enumerate(filters):
        x, skip = _encoder_block(x, f, bn, drop * 0.5 if level < 2 else drop, level)
        skips.append(skip)

    # ---- Bottleneck ----
    x = _conv_block(x, bn_filters, batch_norm=bn, dropout=drop, name_prefix="bottleneck")

    # ---- Decoder ----
    for level, (f, skip) in enumerate(zip(reversed(filters), reversed(skips))):
        x = _decoder_block(x, skip, f, bn, drop * 0.5 if level > 1 else drop,
                           level=len(filters) + level)

    # ---- Output ----
    outputs = layers.Conv2D(
        1, kernel_size=1, activation="sigmoid",
        dtype="float32", name="seg_output",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="UNet")
    total = model.count_params()
    logger.info(f"Built UNet | params: {total:,} | input: {(h, w, c)}")
    return model
