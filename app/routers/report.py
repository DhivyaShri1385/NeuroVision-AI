"""
POST /report — Generate a PDF analysis report.

Runs the full analysis pipeline (classify + segment + explain) on the
uploaded MRI and returns a downloadable PDF report.
"""

from __future__ import annotations

import time
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Query, status
from fastapi.responses import Response

from app.preprocessing import (
    prepare_for_classifier, prepare_for_segmentation,
    array_to_b64, mask_to_b64, overlay_mask_on_image, overlay_heatmap_on_image,
)

router = APIRouter(prefix="/report", tags=["report"])


@router.post(
    "",
    summary="Generate PDF analysis report",
    description=(
        "Runs the full NeuroVision AI pipeline (classify → segment → explain) "
        "on the uploaded MRI and returns a downloadable PDF report.\n\n"
        "The PDF includes the original MRI, diagnosis table, segmentation overlay, "
        "and Grad-CAM explainability heatmap."
    ),
    response_class=Response,
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "PDF report file.",
        }
    },
)
async def generate_report(
    request:    Request,
    file:       UploadFile = File(..., description="Brain MRI image file."),
    xai_method: str = Query("gradcam", description="XAI method: gradcam | smoothgrad"),
    session_id: str | None = Query(None, description="Optional session identifier."),
) -> Response:
    registry = request.app.state.registry
    config   = request.app.state.config

    if not registry.classifier_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier not loaded. Train with run_phase2.py first.",
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail=f"Expected image, got {file.content_type}.")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File read failed: {e}")

    # ── Classification ──────────────────────────────────────────────
    try:
        clf_tensor = prepare_for_classifier(image_bytes, config)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    t0        = time.perf_counter()
    preds     = registry.classifier(clf_tensor, training=False).numpy()[0]
    clf_ms    = (time.perf_counter() - t0) * 1000
    pred_idx  = int(np.argmax(preds))
    clf_dict  = {
        "predicted_class":   config.classes.names[pred_idx],
        "class_index":       pred_idx,
        "confidence":        float(preds[pred_idx]),
        "probabilities":     {n: float(p) for n, p in zip(config.classes.names, preds)},
        "model_name":        registry.classifier.name,
        "inference_time_ms": round(clf_ms, 2),
    }

    # ── Segmentation (optional) ─────────────────────────────────────
    seg_dict = None
    if registry.seg_ready():
        try:
            seg_tensor = prepare_for_segmentation(image_bytes, config)
            t0 = time.perf_counter()
            seg_out = registry.seg_model(seg_tensor, training=False).numpy()[0]
            seg_ms  = (time.perf_counter() - t0) * 1000
            mask    = (seg_out.squeeze() >= 0.5).astype(np.float32)
            seg_dict = {
                "mask_b64":         mask_to_b64(mask),
                "overlay_b64":      overlay_mask_on_image(image_bytes, mask),
                "foreground_ratio": round(float(mask.mean()), 4),
                "model_name":       registry.seg_model.name,
                "inference_time_ms": round(seg_ms, 2),
            }
        except Exception:
            seg_dict = None

    # ── XAI ─────────────────────────────────────────────────────────
    xai_dict = None
    if registry.xai_ready():
        try:
            fn = (registry.gradcam.compute if xai_method.lower() != "smoothgrad"
                  else registry.smoothgrad.compute)
            t0 = time.perf_counter()
            hm, xai_cls_idx, xai_probs = fn(clf_tensor)
            xai_ms = (time.perf_counter() - t0) * 1000
            xai_cfg = config.xai
            xai_dict = {
                "method":            xai_method,
                "heatmap_b64":       array_to_b64(hm),
                "overlay_b64":       overlay_heatmap_on_image(image_bytes, hm, alpha=xai_cfg.overlay_alpha),
                "predicted_class":   config.classes.names[xai_cls_idx],
                "confidence":        float(xai_probs[xai_cls_idx]),
                "inference_time_ms": round(xai_ms, 2),
            }
        except Exception:
            xai_dict = None

    # ── Generate PDF ────────────────────────────────────────────────
    from src.reports.generator import ReportGenerator
    gen       = ReportGenerator(config)
    pdf_bytes = gen.generate(
        image_bytes    = image_bytes,
        classification = clf_dict,
        segmentation   = seg_dict,
        explanation    = xai_dict,
        session_id     = session_id,
    )

    filename  = f"neurovision_{clf_dict['predicted_class']}_{session_id or 'report'}.pdf"
    return Response(
        content      = pdf_bytes,
        media_type   = "application/pdf",
        headers      = {"Content-Disposition": f'attachment; filename="{filename}"'},
    )
