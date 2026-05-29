"""
Attention U-Net — Phase 3

Reference: Oktay et al., "Attention U-Net: Learning Where to Look for
the Pancreas", MIDL 2018. https://arxiv.org/abs/1804.03999

Architecture overview:
  Input (256×256×1)
      │
  Encoder (4 levels, each: Conv3×3→BN→ReLU × 2 → MaxPool2×2)
      │          filters: [64, 128, 256, 512]
  Bottleneck (Conv3×3→BN→ReLU × 2)
      │          filters: 1024
  Decoder (4 levels, each: AttentionGate → TransposedConv2×2 → Concat → Conv3×3×2)
      │
  Output Conv1×1 → Sigmoid → (256×256×1) binary mask

Attention Gate:
  g (gating signal, from decoder)   ──┐
  x (skip connection, from encoder)  ──┼──> W_g(g) + W_x(x) → ReLU → W_psi → Sigmoid → alpha
                                        └──> alpha * x  (soft-gated skip)

All models returned UN-COMPILED. Compilation in trainer so loss can vary.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model

from src.utils.config import AppConfig
from src.utils.logger import logger

# Re-use the shared building blocks from unet.py
from src.segmentation.unet import _conv_block, _encoder_block


# ---------------------------------------------------------------------------
# Attention Gate
# ---------------------------------------------------------------------------

def _attention_gate(
    x: tf.Tensor,
    g: tf.Tensor,
    inter_channels: int,
    name_prefix: str = "",
) -> tf.Tensor:
    """Additive attention gate (Oktay et al. 2018).

    Args:
        x:              Skip-connection feature map  (H, W, C_x)
        g:              Gating signal from decoder   (H/2, W/2, C_g)  — upsampled to match x
        inter_channels: Number of intermediate channels (typically C_x // 2)
        name_prefix:    Prefix for layer names.

    Returns:
        alpha * x  where alpha ∈ [0,1]^{H×W} (soft spatial attention mask)
    """
    # Project gating signal to inter_channels at x's spatial resolution
    theta_x = layers.Conv2D(
        inter_channels, kernel_size=1, strides=1, padding="same",
        use_bias=True, name=f"{name_prefix}_theta_x",
    )(x)

    # Project gating signal — upsample by 2 so it matches x's spatial dims
    phi_g = layers.Conv2D(
        inter_channels, kernel_size=1, strides=1, padding="same",
        use_bias=True, name=f"{name_prefix}_phi_g",
    )(g)
    phi_g = layers.UpSampling2D(size=(2, 2), interpolation="bilinear",
                                name=f"{name_prefix}_phi_g_up")(phi_g)

    # Additive merge → ReLU
    f = layers.Add(name=f"{name_prefix}_add")([theta_x, phi_g])
    f = layers.Activation("relu", name=f"{name_prefix}_relu")(f)

    # Psi projection → Sigmoid → attention coefficients
    psi = layers.Conv2D(
        1, kernel_size=1, strides=1, padding="same",
        use_bias=True, name=f"{name_prefix}_psi",
    )(f)
    psi = layers.Activation("sigmoid", name=f"{name_prefix}_sigmoid")(psi)

    # Gate the skip connection (broadcast psi over all channels)
    return layers.Multiply(name=f"{name_prefix}_gated")([x, psi])


# ---------------------------------------------------------------------------
# Attention Decoder block
# ---------------------------------------------------------------------------

def _attention_decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    batch_norm: bool,
    dropout: float,
    level: int,
) -> tf.Tensor:
    """Attention decoder block:
    AttentionGate(skip, g=x) → Upsample x (×2) → Concat gated_skip → conv_block.
    """
    inter_ch = max(filters // 2, 1)
    gated_skip = _attention_gate(
        skip, x, inter_channels=inter_ch,
        name_prefix=f"attn{level}",
    )

    # Upsample decoder feature map
    x = layers.Conv2DTranspose(
        filters, kernel_size=2, strides=2, padding="same",
        name=f"attn_dec{level}_upsample",
    )(x)

    # Concat attended skip with upsampled x
    x = layers.Concatenate(name=f"attn_dec{level}_concat")([x, gated_skip])

    # Double conv block
    x = _conv_block(
        x, filters,
        batch_norm=batch_norm, dropout=dropout,
        name_prefix=f"attn_dec{level}",
    )
    return x


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_attention_unet(config: AppConfig) -> Model:
    """Build and return an Attention U-Net model.

    Args:
        config: AppConfig with segmentation sub-config.

    Returns:
        Uncompiled tf.keras.Model.
    """
    seg = config.segmentation
    h, w = seg.input_size
    c    = seg.channels
    filters    = seg.filters           # [64, 128, 256, 512]
    bn_filters = seg.bottleneck_filters  # 1024
    bn         = seg.batch_norm
    drop       = seg.dropout

    inputs = tf.keras.Input(shape=(h, w, c), name="mri_input")

    # ---- Encoder ----
    x = inputs
    skips: list[tf.Tensor] = []
    for level, f in enumerate(filters):
        # Use lighter dropout in early layers, full dropout in deep layers
        level_drop = drop * 0.5 if level < 2 else drop
        x, skip = _encoder_block(x, f, bn, level_drop, level)
        skips.append(skip)

    # ---- Bottleneck ----
    x = _conv_block(x, bn_filters, batch_norm=bn, dropout=drop,
                    name_prefix="bottleneck")

    # ---- Attention Decoder ----
    for level, (f, skip) in enumerate(zip(reversed(filters), reversed(skips))):
        dec_drop = drop * 0.5 if level > 1 else drop
        x = _attention_decoder_block(
            x, skip, f, bn, dec_drop,
            level=len(filters) + level,
        )

    # ---- Output ----
    outputs = layers.Conv2D(
        1, kernel_size=1, activation="sigmoid",
        dtype="float32", name="seg_output",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="AttentionUNet")
    total = model.count_params()
    logger.info(f"Built AttentionUNet | params: {total:,} | input: {(h, w, c)}")
    return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_segmentation_model(config: AppConfig) -> Model:
    """Return the correct segmentation model based on config.segmentation.architecture."""
    arch = config.segmentation.architecture
    if arch == "UNet":
        from src.segmentation.unet import build_unet
        return build_unet(config)
    elif arch == "AttentionUNet":
        return build_attention_unet(config)
    else:
        raise ValueError(
            f"Unknown segmentation architecture: '{arch}'. "
            "Valid options: 'UNet', 'AttentionUNet'."
        )
