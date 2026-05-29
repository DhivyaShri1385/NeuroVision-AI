"""
NeuroVision AI — FastAPI Application — Phase 5

Start the server:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Or via the convenience script:
    python scripts/run_phase5.py

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)

Endpoints:
    GET  /health           — liveness + readiness probe
    GET  /classes          — list of tumour class names
    POST /classify         — image → class + confidence
    POST /segment          — image → binary mask + overlay
    POST /explain          — image → Grad-CAM / SmoothGrad heatmap
    POST /analyze          — image → full pipeline in one call
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root on sys.path so src.* imports work regardless of
# where uvicorn is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.config import load_config
from src.utils.logger import logger, setup_logger
from app.model_registry import ModelRegistry
from app.schemas import (
    HealthResponse,
    ClassificationResponse,
    SegmentationResponse,
    ExplanationResponse,
    AnalysisResponse,
    ErrorResponse,
)
from app.preprocessing import (
    prepare_for_classifier,
    prepare_for_segmentation,
    array_to_b64,
    mask_to_b64,
    overlay_mask_on_image,
    overlay_heatmap_on_image,
)
from app.routers import classify, segment, explain, health, report


# ---------------------------------------------------------------------------
# Lifespan: load models on startup, release on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models before accepting requests."""
    config = load_config()
    setup_logger(log_dir=config.paths.logs_dir, level="INFO")

    logger.info("NeuroVision AI API starting up — loading models ...")
    registry = ModelRegistry(config)
    registry.load()

    # Stash on app.state so routers can access them via request.app.state
    app.state.config   = config
    app.state.registry = registry

    logger.info(
        f"API ready | models={registry.loaded_names} | "
        f"docs=http://localhost:8000/docs"
    )
    yield  # ← server runs here

    logger.info("API shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "NeuroVision AI",
    description = (
        "Explainable Brain Tumour Diagnosis & Segmentation API.\n\n"
        "Classifies MRI images into **glioma / meningioma / no_tumor / pituitary**, "
        "generates pixel-level tumour masks, and produces Grad-CAM / SmoothGrad "
        "explainability heatmaps."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ---------------------------------------------------------------------------
# CORS — allow React frontend on localhost:3000 (and any origin in dev)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(classify.router)
app.include_router(segment.router)
app.include_router(explain.router)
app.include_router(report.router)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["root"], include_in_schema=False)
async def root():
    return {
        "message": "NeuroVision AI API",
        "docs":    "/docs",
        "health":  "/health",
    }


# ---------------------------------------------------------------------------
# GET /classes — list of class names
# ---------------------------------------------------------------------------

@app.get(
    "/classes",
    tags=["meta"],
    summary="List tumour classes",
    response_model=dict,
)
async def get_classes(request: Request):
    """Returns all class names and their integer indices."""
    config = request.app.state.config
    return {
        "classes":   config.classes.names,
        "label_map": config.classes.label_map,
        "num_classes": config.classes.num_classes,
    }


# ---------------------------------------------------------------------------
# POST /analyze — full pipeline: classify + segment + explain in one request
# ---------------------------------------------------------------------------

@app.post(
    "/analyze",
    response_model   = AnalysisResponse,
    tags             = ["full pipeline"],
    summary          = "Full analysis: classify + segment + explain",
    description      = (
        "One-shot endpoint that runs the complete pipeline on a single MRI image:\n\n"
        "1. **Classify** — tumour type + confidence\n"
        "2. **Segment**  — binary tumour mask\n"
        "3. **Explain**  — Grad-CAM heatmap\n\n"
        "Returns all three results in a single JSON response."
    ),
)
async def analyze(
    request: Request,
    file:    UploadFile = File(..., description="Brain MRI image file."),
    xai_method: str = Query("gradcam", description="XAI method: gradcam | smoothgrad"),
) -> AnalysisResponse:
    registry = request.app.state.registry
    config   = request.app.state.config
    t_total  = time.perf_counter()

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected an image, got '{file.content_type}'.",
        )

    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}")

    # ---- Classification ----
    if not registry.classifier_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier not loaded. Train with run_phase2.py first.",
        )
    try:
        clf_tensor = prepare_for_classifier(image_bytes, config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    t0          = time.perf_counter()
    clf_preds   = registry.classifier(clf_tensor, training=False).numpy()[0]
    clf_ms      = (time.perf_counter() - t0) * 1000
    pred_idx    = int(np.argmax(clf_preds))
    pred_class  = config.classes.names[pred_idx]
    clf_resp    = ClassificationResponse(
        predicted_class   = pred_class,
        class_index       = pred_idx,
        confidence        = float(clf_preds[pred_idx]),
        probabilities     = {n: float(p) for n, p in zip(config.classes.names, clf_preds)},
        model_name        = registry.classifier.name,
        inference_time_ms = round(clf_ms, 2),
    )

    # ---- Segmentation ----
    if registry.seg_ready():
        try:
            seg_tensor = prepare_for_segmentation(image_bytes, config)
        except ValueError:
            seg_tensor = None

        if seg_tensor is not None:
            t0      = time.perf_counter()
            seg_out = registry.seg_model(seg_tensor, training=False).numpy()[0]
            seg_ms  = (time.perf_counter() - t0) * 1000
            mask    = (seg_out.squeeze() >= 0.5).astype(np.float32)
            seg_resp = SegmentationResponse(
                mask_b64         = mask_to_b64(mask),
                overlay_b64      = overlay_mask_on_image(image_bytes, mask),
                foreground_ratio = round(float(mask.mean()), 4),
                model_name       = registry.seg_model.name,
                inference_time_ms= round(seg_ms, 2),
            )
        else:
            seg_resp = _empty_seg_response(registry)
    else:
        seg_resp = _empty_seg_response(registry)

    # ---- XAI ----
    if registry.xai_ready():
        t0 = time.perf_counter()
        method_fn = (
            registry.gradcam.compute
            if xai_method.lower() != "smoothgrad"
            else registry.smoothgrad.compute
        )
        hm, xai_class_idx, xai_probs = method_fn(clf_tensor)
        xai_ms = (time.perf_counter() - t0) * 1000

        xai_cfg = config.xai
        xai_resp = ExplanationResponse(
            method            = xai_method,
            heatmap_b64       = array_to_b64(hm),
            overlay_b64       = overlay_heatmap_on_image(
                image_bytes, hm, alpha=xai_cfg.overlay_alpha
            ),
            predicted_class   = config.classes.names[xai_class_idx],
            confidence        = float(xai_probs[xai_class_idx]),
            inference_time_ms = round(xai_ms, 2),
        )
    else:
        xai_resp = ExplanationResponse(
            method="unavailable", heatmap_b64="", overlay_b64="",
            predicted_class=pred_class, confidence=float(clf_preds[pred_idx]),
            inference_time_ms=0.0,
        )

    total_ms = (time.perf_counter() - t_total) * 1000
    return AnalysisResponse(
        classification         = clf_resp,
        segmentation           = seg_resp,
        explanation            = xai_resp,
        total_inference_time_ms= round(total_ms, 2),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_seg_response(registry) -> SegmentationResponse:
    return SegmentationResponse(
        mask_b64         = "",
        overlay_b64      = "",
        foreground_ratio = 0.0,
        model_name       = "unavailable",
        inference_time_ms= 0.0,
    )
