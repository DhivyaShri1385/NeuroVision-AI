"""
Phase 7 — Generate a sample PDF report without the API server.

Loads the trained models, runs the full pipeline on N test images,
and saves PDF reports to outputs/reports/.

Usage:
    python scripts/run_phase7.py                    # 1 report, auto image
    python scripts/run_phase7.py --n 5             # 5 reports
    python scripts/run_phase7.py --image path.jpg  # specific image
    python scripts/run_phase7.py --quick-test      # 1 image, no SmoothGrad
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.utils.config import load_config
from src.utils.logger import logger, setup_logger
from src.xai.gradcam  import GradCAM
from src.xai.saliency import SmoothGrad
from src.reports.generator import ReportGenerator
from app.preprocessing import (
    prepare_for_classifier, prepare_for_segmentation,
    array_to_b64, mask_to_b64,
    overlay_mask_on_image, overlay_heatmap_on_image,
)


def _parse_args():
    p = argparse.ArgumentParser(description="NeuroVision AI — Phase 7 PDF Reports")
    p.add_argument("--n",          type=int, default=1, help="Number of reports to generate.")
    p.add_argument("--image",      default=None, help="Path to a specific MRI image.")
    p.add_argument("--xai-method", default="gradcam", choices=["gradcam", "smoothgrad"])
    p.add_argument("--quick-test", action="store_true", help="1 image, minimal XAI.")
    return p.parse_args()


def _find_test_images(config, n: int) -> list[Path]:
    test_dir = config.paths.test_dir
    images   = sorted(Path(test_dir).rglob("*.jpg"))[:n]
    if not images:
        images = sorted(Path(test_dir).rglob("*.png"))[:n]
    return images


def _load_classifier(config):
    models_dir = config.paths.models_dir
    for name in ["EfficientNetB3_best.keras", "ResNet50_best.keras", "DenseNet121_best.keras"]:
        p = models_dir / name
        if p.exists():
            logger.info(f"Loading classifier: {p.name}")
            return tf.keras.models.load_model(str(p), compile=False)
    logger.warning("No classifier checkpoint — building untrained model")
    from src.models.classifier import build_classifier
    return build_classifier("EfficientNetB3", config)


def _load_seg(config):
    models_dir = config.paths.models_dir
    for name in ["attentionunet_best.keras", "unet_best.keras"]:
        p = models_dir / name
        if p.exists():
            logger.info(f"Loading seg model: {p.name}")
            return tf.keras.models.load_model(str(p), compile=False)
    return None


def generate_report_for_image(
    image_path: Path, config, classifier, seg_model, gcam, sg_model,
    xai_method: str, reports_dir: Path
) -> Path:
    logger.info(f"Processing: {image_path.name}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # ── Classify ────────────────────────────────────────────────
    clf_t   = prepare_for_classifier(image_bytes, config)
    t0      = time.perf_counter()
    preds   = classifier(clf_t, training=False).numpy()[0]
    clf_ms  = round((time.perf_counter() - t0) * 1000, 1)
    pred_idx= int(np.argmax(preds))
    clf_dict = {
        "predicted_class":   config.classes.names[pred_idx],
        "class_index":       pred_idx,
        "confidence":        float(preds[pred_idx]),
        "probabilities":     {n: float(p) for n, p in zip(config.classes.names, preds)},
        "model_name":        classifier.name,
        "inference_time_ms": clf_ms,
    }
    logger.info(f"  class={clf_dict['predicted_class']} conf={clf_dict['confidence']:.1%}")

    # ── Segment ─────────────────────────────────────────────────
    seg_dict = None
    if seg_model is not None:
        try:
            seg_t = prepare_for_segmentation(image_bytes, config)
            t0 = time.perf_counter()
            seg_out = seg_model(seg_t, training=False).numpy()[0]
            seg_ms  = round((time.perf_counter() - t0) * 1000, 1)
            mask    = (seg_out.squeeze() >= 0.5).astype(np.float32)
            seg_dict = {
                "mask_b64":         mask_to_b64(mask),
                "overlay_b64":      overlay_mask_on_image(image_bytes, mask),
                "foreground_ratio": round(float(mask.mean()), 4),
                "model_name":       seg_model.name,
                "inference_time_ms": seg_ms,
            }
        except Exception as e:
            logger.warning(f"Segmentation failed: {e}")

    # ── XAI ─────────────────────────────────────────────────────
    xai_dict = None
    if gcam is not None:
        try:
            fn     = gcam.compute if xai_method != "smoothgrad" else sg_model.compute
            t0     = time.perf_counter()
            hm, xai_idx, xai_probs = fn(clf_t)
            xai_ms = round((time.perf_counter() - t0) * 1000, 1)
            xai_cfg = config.xai
            xai_dict = {
                "method":            xai_method,
                "heatmap_b64":       array_to_b64(hm),
                "overlay_b64":       overlay_heatmap_on_image(image_bytes, hm, alpha=xai_cfg.overlay_alpha),
                "predicted_class":   config.classes.names[xai_idx],
                "confidence":        float(xai_probs[xai_idx]),
                "inference_time_ms": xai_ms,
            }
        except Exception as e:
            logger.warning(f"XAI failed: {e}")

    # ── Generate PDF ────────────────────────────────────────────
    gen      = ReportGenerator(config)
    session  = f"NV-{image_path.stem[:10]}"
    pdf_bytes = gen.generate(
        image_bytes    = image_bytes,
        classification = clf_dict,
        segmentation   = seg_dict,
        explanation    = xai_dict,
        session_id     = session,
    )

    out = reports_dir / f"report_{image_path.stem}_{clf_dict['predicted_class']}.pdf"
    out.write_bytes(pdf_bytes)
    logger.info(f"  Saved: {out.name}  ({len(pdf_bytes)//1024} KB)")
    return out


def main():
    args   = _parse_args()
    config = load_config()
    setup_logger(log_dir=config.paths.logs_dir, level="INFO")

    logger.info("=" * 56)
    logger.info("NeuroVision AI — Phase 7: PDF Report Generation")
    logger.info("=" * 56)

    # Load models
    classifier = _load_classifier(config)
    seg_model  = _load_seg(config)
    gcam       = GradCAM(classifier, config)
    sg_mdl     = SmoothGrad(classifier, config, n_samples=5 if args.quick_test else 20)

    # Find images
    if args.image:
        images = [Path(args.image)]
    else:
        n_imgs = 1 if args.quick_test else args.n
        images = _find_test_images(config, n_imgs)

    if not images:
        logger.error("No test images found. Check DATASET_ROOT in .env")
        sys.exit(1)

    reports_dir = config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {len(images)} report(s) -> {reports_dir}")

    saved = []
    for img_path in images:
        try:
            out = generate_report_for_image(
                img_path, config, classifier, seg_model,
                gcam, sg_mdl, args.xai_method, reports_dir,
            )
            saved.append(out)
        except Exception as e:
            logger.error(f"Failed for {img_path.name}: {e}")

    print()
    print("=" * 56)
    print(f"  Phase 7 Complete — {len(saved)}/{len(images)} reports saved")
    print(f"  Output: {reports_dir}")
    for s in saved:
        print(f"    {s.name}")
    print("=" * 56)


if __name__ == "__main__":
    main()
