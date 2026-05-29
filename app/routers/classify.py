"""
POST /classify — Brain tumour classification endpoint.

Accepts a multipart MRI image upload and returns the predicted tumour class,
confidence score, and full softmax probability vector.
"""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, status

from app.schemas import ClassificationResponse
from app.preprocessing import prepare_for_classifier

router = APIRouter(prefix="/classify", tags=["classification"])


@router.post(
    "",
    response_model=ClassificationResponse,
    summary="Classify brain tumour type",
    description=(
        "Upload a brain MRI image (JPEG/PNG/BMP/TIFF) and receive the predicted "
        "tumour class with confidence score and full probability distribution.\n\n"
        "**Classes:** glioma · meningioma · no_tumor · pituitary"
    ),
)
async def classify(
    request: Request,
    file: UploadFile = File(..., description="Brain MRI image file."),
) -> ClassificationResponse:
    registry = request.app.state.registry
    config   = request.app.state.config

    # Guard: model must be loaded
    if not registry.classifier_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Classifier not available. "
                "Train the model first: python scripts/run_phase2.py"
            ),
        )

    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected an image file, got '{file.content_type}'.",
        )

    # Read and preprocess
    try:
        image_bytes = await file.read()
        img_tensor  = prepare_for_classifier(image_bytes, config)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # Inference
    t0    = time.perf_counter()
    preds = registry.classifier(img_tensor, training=False).numpy()[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pred_idx   = int(np.argmax(preds))
    pred_class = config.classes.names[pred_idx]
    confidence = float(preds[pred_idx])

    probabilities = {
        name: float(prob)
        for name, prob in zip(config.classes.names, preds)
    }

    return ClassificationResponse(
        predicted_class   = pred_class,
        class_index       = pred_idx,
        confidence        = confidence,
        probabilities     = probabilities,
        model_name        = registry.classifier.name,
        inference_time_ms = round(elapsed_ms, 2),
    )
