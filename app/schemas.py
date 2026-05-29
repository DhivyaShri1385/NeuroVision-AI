"""
Pydantic v2 request / response schemas — Phase 5

All image data is exchanged as base64-encoded PNG strings so the API
is stateless and works without a shared filesystem between client and server.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# /classify
# ---------------------------------------------------------------------------

class ClassificationResponse(BaseModel):
    """Response for POST /classify."""

    predicted_class: str = Field(
        description="Predicted tumour class (glioma | meningioma | no_tumor | pituitary)."
    )
    class_index: int = Field(description="0-based class index.")
    confidence: float = Field(ge=0.0, le=1.0, description="Softmax probability of predicted class.")
    probabilities: dict[str, float] = Field(
        description="Softmax probability for every class."
    )
    model_name: str = Field(description="Backbone model used for inference.")
    inference_time_ms: float = Field(description="Server-side inference time in ms.")


# ---------------------------------------------------------------------------
# /segment
# ---------------------------------------------------------------------------

class SegmentationResponse(BaseModel):
    """Response for POST /segment."""

    mask_b64: str = Field(
        description="Base64-encoded PNG of the binary segmentation mask (0/255)."
    )
    overlay_b64: str = Field(
        description="Base64-encoded PNG of the mask overlaid on the input MRI."
    )
    foreground_ratio: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of pixels classified as tumour."
    )
    model_name: str = Field(description="Segmentation architecture used.")
    inference_time_ms: float = Field(description="Server-side inference time in ms.")


# ---------------------------------------------------------------------------
# /explain
# ---------------------------------------------------------------------------

class ExplanationResponse(BaseModel):
    """Response for POST /explain."""

    method: str = Field(
        description="XAI method used (GradCAM | SmoothGrad | IntegratedGradients)."
    )
    heatmap_b64: str = Field(
        description="Base64-encoded PNG of the raw heatmap."
    )
    overlay_b64: str = Field(
        description="Base64-encoded PNG of the heatmap blended over the input MRI."
    )
    predicted_class: str = Field(description="Class the heatmap is attributed to.")
    confidence: float = Field(ge=0.0, le=1.0)
    inference_time_ms: float = Field(description="Server-side inference time in ms.")


# ---------------------------------------------------------------------------
# /analyze  (combined endpoint)
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    """Response for POST /analyze — full pipeline in one call."""

    classification: ClassificationResponse
    segmentation: SegmentationResponse
    explanation: ExplanationResponse
    total_inference_time_ms: float


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: list[str]
    version: str
