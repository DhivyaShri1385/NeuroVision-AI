"""
POST /explain — XAI explanation endpoint.

Accepts a multipart MRI image and an optional method parameter, and returns:
  - Raw heatmap as base64 PNG
  - Heatmap blended over the original MRI as base64 PNG
  - Predicted class and confidence

Supported methods:
  gradcam      — Gradient × Input (class-discriminative, fast)
  smoothgrad   — Averaged gradient over N noisy copies (sharper)
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Query, status

from app.schemas import ExplanationResponse
from app.preprocessing import (
    prepare_for_classifier,
    array_to_b64,
    overlay_heatmap_on_image,
)

router = APIRouter(prefix="/explain", tags=["explainability"])

_VALID_METHODS = {"gradcam", "smoothgrad"}


@router.post(
    "",
    response_model=ExplanationResponse,
    summary="Generate XAI explanation heatmap",
    description=(
        "Upload a brain MRI image and receive a visual explanation of the "
        "model's prediction using Grad-CAM or SmoothGrad.\n\n"
        "| Method | Description |\n"
        "|--------|-------------|\n"
        "| `gradcam` | Gradient × Input — fast, class-discriminative |\n"
        "| `smoothgrad` | Average gradient over N noisy copies — sharper |\n"
    ),
)
async def explain(
    request: Request,
    file:   UploadFile = File(..., description="Brain MRI image file."),
    method: str        = Query("gradcam", description="XAI method: gradcam | smoothgrad"),
    target_class: int | None = Query(
        None,
        description="Target class index (0-3). Defaults to predicted class.",
    ),
) -> ExplanationResponse:
    registry = request.app.state.registry
    config   = request.app.state.config

    if not registry.xai_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "XAI explainers not available — classifier must be loaded first. "
                "Train: python scripts/run_phase2.py"
            ),
        )

    method = method.lower()
    if method not in _VALID_METHODS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown method '{method}'. Valid: {sorted(_VALID_METHODS)}",
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected an image file, got '{file.content_type}'.",
        )

    try:
        image_bytes = await file.read()
        img_tensor  = prepare_for_classifier(image_bytes, config)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # Compute heatmap
    t0 = time.perf_counter()
    if method == "gradcam":
        heatmap, pred_idx, probs = registry.gradcam.compute(
            img_tensor, class_idx=target_class
        )
    else:  # smoothgrad
        heatmap, pred_idx, probs = registry.smoothgrad.compute(
            img_tensor, class_idx=target_class
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pred_class = config.classes.names[pred_idx]
    confidence = float(probs[pred_idx])

    # Encode outputs
    xai_cfg     = config.xai
    heatmap_b64 = array_to_b64(heatmap)
    overlay_b64 = overlay_heatmap_on_image(
        image_bytes, heatmap,
        alpha   = xai_cfg.overlay_alpha,
    )

    return ExplanationResponse(
        method            = method,
        heatmap_b64       = heatmap_b64,
        overlay_b64       = overlay_b64,
        predicted_class   = pred_class,
        confidence        = confidence,
        inference_time_ms = round(elapsed_ms, 2),
    )
