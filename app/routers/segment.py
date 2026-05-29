"""
POST /segment — Brain tumour segmentation endpoint.

Accepts a multipart MRI image upload and returns:
  - Binary segmentation mask (base64 PNG)
  - Mask overlaid on the original MRI (base64 PNG)
  - Foreground pixel ratio (tumour area fraction)
"""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, status

from app.schemas import SegmentationResponse
from app.preprocessing import (
    prepare_for_segmentation,
    mask_to_b64,
    overlay_mask_on_image,
)

router = APIRouter(prefix="/segment", tags=["segmentation"])


@router.post(
    "",
    response_model=SegmentationResponse,
    summary="Segment tumour region",
    description=(
        "Upload a brain MRI image and receive a binary tumour segmentation mask "
        "along with a coloured overlay on the original image.\n\n"
        "The model uses an **Attention U-Net** trained on pseudo-masks generated "
        "via Otsu thresholding + morphological refinement."
    ),
)
async def segment(
    request: Request,
    file: UploadFile = File(..., description="Brain MRI image file."),
) -> SegmentationResponse:
    registry = request.app.state.registry
    config   = request.app.state.config

    if not registry.seg_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Segmentation model not available. "
                "Train first: python scripts/run_phase3.py"
            ),
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected an image file, got '{file.content_type}'.",
        )

    try:
        image_bytes = await file.read()
        img_tensor  = prepare_for_segmentation(image_bytes, config)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # Inference
    t0   = time.perf_counter()
    pred = registry.seg_model(img_tensor, training=False).numpy()[0]  # (H, W, 1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Binarise
    mask = (pred.squeeze() >= 0.5).astype(np.float32)  # (H, W)
    foreground_ratio = float(mask.mean())

    # Encode outputs
    mask_b64    = mask_to_b64(mask)
    overlay_b64 = overlay_mask_on_image(
        image_bytes, mask,
        alpha=0.45,
        color=(220, 50, 50),   # Red tumour overlay
    )

    return SegmentationResponse(
        mask_b64         = mask_b64,
        overlay_b64      = overlay_b64,
        foreground_ratio = round(foreground_ratio, 4),
        model_name       = registry.seg_model.name,
        inference_time_ms= round(elapsed_ms, 2),
    )
