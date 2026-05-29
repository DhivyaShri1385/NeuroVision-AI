"""
Phase 4 Orchestration — XAI (Explainability)
=============================================

Generates Grad-CAM, SmoothGrad, and Integrated Gradients explanations
for predictions made by the Phase 2 classifier.

Pipeline:
  1. Load config + logger
  2. Load trained classifier model (best checkpoint from Phase 2)
  3. Build test tf.data pipeline and sample N images
  4. Compute Grad-CAM heatmaps for each sample
  5. Compute SmoothGrad saliency maps
  6. Compute Integrated Gradients maps
  7. Generate visualisations:
     a. Single-image Grad-CAM overlays
     b. XAI comparison (GradCAM | SmoothGrad | IG) per sample
     c. Per-class activation grid (one Grad-CAM per class)
     d. Multi-image summary grid
  8. Print results summary

CLI flags:
  --backbone       EfficientNetB3 | ResNet50 | DenseNet121
                   (used to locate checkpoint file)
  --model-path     Direct path to a .keras checkpoint file
  --n-samples      Number of test images to explain  (default 12)
  --no-smoothgrad  Skip SmoothGrad (slow on CPU; ~50 forward passes/image)
  --no-ig          Skip Integrated Gradients
  --quick-test     3 samples, 5 SmoothGrad steps — fast smoke test

Examples:
  python scripts/run_phase4.py
  python scripts/run_phase4.py --backbone EfficientNetB3 --n-samples 20
  python scripts/run_phase4.py --model-path models/EfficientNetB3_best.keras
  python scripts/run_phase4.py --quick-test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path ──────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.utils.config import load_config, AppConfig
from src.utils.logger import logger, setup_logger
from src.data.data_loader import BrainTumorDataLoader
from src.xai.gradcam  import GradCAM
from src.xai.saliency import SmoothGrad, IntegratedGradients
from src.xai.visualizer import XAIVisualizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroVision AI — Phase 4 XAI")
    p.add_argument("--backbone",   default="EfficientNetB3",
                   choices=["EfficientNetB3", "ResNet50", "DenseNet121"],
                   help="Backbone name (used to find checkpoint).")
    p.add_argument("--model-path", default=None,
                   help="Direct path to .keras model file (overrides --backbone).")
    p.add_argument("--n-samples",  type=int, default=12,
                   help="Number of test images to explain.")
    p.add_argument("--no-smoothgrad", action="store_true",
                   help="Skip SmoothGrad (saves time on CPU).")
    p.add_argument("--no-ig",      action="store_true",
                   help="Skip Integrated Gradients.")
    p.add_argument("--quick-test", action="store_true",
                   help="3 samples, 5 SG steps — smoke test.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(config: AppConfig, backbone: str, model_path: str | None) -> tf.keras.Model:
    """Load the best Phase 2 classifier checkpoint."""
    if model_path:
        ckpt = Path(model_path)
    else:
        # Try best checkpoint, then final
        ckpt = config.paths.models_dir / f"{backbone}_best.keras"
        if not ckpt.exists():
            ckpt = config.paths.models_dir / f"{backbone.lower()}_best.keras"
        if not ckpt.exists():
            ckpt = config.paths.models_dir / f"{backbone}_final.keras"

    if not ckpt.exists():
        logger.warning(
            f"No trained checkpoint found at {ckpt}.\n"
            "Building untrained model for pipeline demonstration.\n"
            "Run 'python scripts/run_phase2.py' to train first."
        )
        return _build_untrained_model(config, backbone)

    logger.info(f"Loading model from {ckpt}")
    model = tf.keras.models.load_model(str(ckpt), compile=False)
    logger.info(f"Loaded model: {model.name} | params={model.count_params():,}")
    return model


def _build_untrained_model(config: AppConfig, backbone: str) -> tf.keras.Model:
    """Build a fresh (untrained) model for smoke-testing the XAI pipeline."""
    from src.models.classifier import build_classifier
    model = build_classifier(backbone, config)
    logger.warning(f"Using UNTRAINED {backbone} — explanations will be random.")
    return model


# ---------------------------------------------------------------------------
# Data sampling
# ---------------------------------------------------------------------------

def _sample_test_images(
    config: AppConfig,
    n: int,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Pull N images + labels from the test split.

    Returns:
        images:      (N, H, W, C) float32.
        label_idxs:  List of integer class indices.
        label_names: List of class name strings.
    """
    loader = BrainTumorDataLoader(config)
    test_ds = loader.get_test_dataset_from_dir()

    imgs_list:  list[np.ndarray] = []
    lbls_list:  list[np.ndarray] = []

    for imgs, lbls in test_ds:
        imgs_list.append(imgs.numpy())
        lbls_list.append(lbls.numpy())
        if sum(x.shape[0] for x in imgs_list) >= n:
            break

    images_all = np.concatenate(imgs_list, axis=0)[:n]
    labels_all = np.concatenate(lbls_list, axis=0)[:n]

    # Convert one-hot labels to indices
    label_idxs  = list(np.argmax(labels_all, axis=1).astype(int))
    label_names = [config.classes.names[i] for i in label_idxs]

    logger.info(f"Sampled {len(images_all)} test images")
    return images_all, label_idxs, label_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = _parse_args()
    config = load_config()

    setup_logger(
        log_dir=config.paths.logs_dir,
        level="DEBUG" if args.quick_test else "INFO",
    )

    n_samples   = 3  if args.quick_test else args.n_samples
    sg_steps    = 5  if args.quick_test else 50
    ig_steps    = 10 if args.quick_test else 50
    run_sg      = not args.no_smoothgrad
    run_ig      = not args.no_ig

    logger.info("=" * 60)
    logger.info(f"NeuroVision AI — Phase 4 XAI | backbone={args.backbone}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Load model
    # ------------------------------------------------------------------
    logger.info("[1/6] Loading classifier ...")
    model = _load_model(config, args.backbone, args.model_path)

    # ------------------------------------------------------------------
    # Step 2 — Sample test images
    # ------------------------------------------------------------------
    logger.info(f"[2/6] Sampling {n_samples} test images ...")
    images, true_idxs, true_names = _sample_test_images(config, n_samples)

    # ------------------------------------------------------------------
    # Step 3 — Grad-CAM
    # ------------------------------------------------------------------
    logger.info("[3/6] Computing Grad-CAM heatmaps ...")
    gcam = GradCAM(model, config)

    gradcam_maps: list[np.ndarray] = []
    pred_names:   list[str]        = []
    pred_probs:   list[float]      = []
    pred_idxs:    list[int]        = []
    all_probs:    list[np.ndarray] = []

    t0 = time.time()
    for i, img in enumerate(images):
        hm, pred_idx, probs = gcam.compute(img[np.newaxis])
        gradcam_maps.append(hm)
        pred_idxs.append(pred_idx)
        pred_names.append(config.classes.names[pred_idx])
        pred_probs.append(float(probs[pred_idx]))
        all_probs.append(probs)
        logger.debug(
            f"  [{i+1}/{n_samples}] pred={config.classes.names[pred_idx]} "
            f"({probs[pred_idx]:.1%}) | gt={true_names[i]}"
        )
    logger.info(f"Grad-CAM done | {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 4 — SmoothGrad
    # ------------------------------------------------------------------
    smoothgrad_maps: list[np.ndarray] = []
    if run_sg:
        logger.info(f"[4/6] Computing SmoothGrad ({sg_steps} samples/image) ...")
        sgm = SmoothGrad(model, config, n_samples=sg_steps, noise_std=0.15)
        t0  = time.time()
        for i, img in enumerate(images):
            hm, _, _ = sgm.compute(img[np.newaxis], class_idx=pred_idxs[i])
            smoothgrad_maps.append(hm)
            logger.debug(f"  SmoothGrad [{i+1}/{n_samples}]")
        logger.info(f"SmoothGrad done | {time.time()-t0:.1f}s")
    else:
        logger.info("[4/6] Skipping SmoothGrad (--no-smoothgrad)")
        smoothgrad_maps = [np.zeros_like(gradcam_maps[i]) for i in range(n_samples)]

    # ------------------------------------------------------------------
    # Step 5 — Integrated Gradients
    # ------------------------------------------------------------------
    ig_maps: list[np.ndarray] = []
    if run_ig:
        logger.info(f"[5/6] Computing Integrated Gradients ({ig_steps} steps/image) ...")
        igm = IntegratedGradients(model, config, steps=ig_steps, baseline=0.0)
        t0  = time.time()
        for i, img in enumerate(images):
            hm, _, _ = igm.compute(img[np.newaxis], class_idx=pred_idxs[i])
            ig_maps.append(hm)
            logger.debug(f"  IG [{i+1}/{n_samples}]")
        logger.info(f"Integrated Gradients done | {time.time()-t0:.1f}s")
    else:
        logger.info("[5/6] Skipping Integrated Gradients (--no-ig)")
        ig_maps = None

    # ------------------------------------------------------------------
    # Step 6 — Visualisations
    # ------------------------------------------------------------------
    logger.info("[6/6] Generating visualisations ...")
    viz = XAIVisualizer(config)

    # a) Individual Grad-CAM overlays (first 4 images)
    for i in range(min(4, n_samples)):
        viz.plot_gradcam_overlay(
            image      = images[i],
            heatmap    = gradcam_maps[i],
            true_label = true_names[i],
            pred_label = pred_names[i],
            pred_prob  = pred_probs[i],
            filename   = f"gradcam_{i:02d}_{pred_names[i]}.png",
        )

    # b) XAI comparison panels
    for i in range(min(4, n_samples)):
        viz.plot_xai_comparison(
            image        = images[i],
            gradcam_map  = gradcam_maps[i],
            saliency_map = smoothgrad_maps[i],
            ig_map       = ig_maps[i] if ig_maps else None,
            pred_label   = pred_names[i],
            pred_prob    = pred_probs[i],
            true_label   = true_names[i],
            filename     = f"xai_comparison_{i:02d}.png",
        )

    # c) Per-class activation grid (first image, all 4 classes)
    logger.info("  Computing per-class activation grid ...")
    class_heatmaps: dict[str, np.ndarray] = {}
    for cls_idx, cls_name in enumerate(config.classes.names):
        hm, _, _ = gcam.compute(images[0][np.newaxis], class_idx=cls_idx)
        class_heatmaps[cls_name] = hm

    viz.plot_class_activation_grid(
        image       = images[0],
        heatmaps    = class_heatmaps,
        pred_label  = pred_names[0],
        filename    = "class_activation_grid.png",
    )

    # d) Summary grid
    viz.plot_xai_summary(
        images       = images,
        gradcam_maps = gradcam_maps,
        pred_labels  = pred_names,
        pred_probs   = pred_probs,
        true_labels  = true_names,
        n_samples    = n_samples,
        filename     = "xai_summary_grid.png",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    correct  = sum(p == t for p, t in zip(pred_names, true_names))
    accuracy = correct / n_samples

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Phase 4 Complete — XAI ({args.backbone})")
    logger.info("=" * 60)
    logger.info(f"  Samples explained : {n_samples}")
    logger.info(f"  Correct preds     : {correct}/{n_samples} ({accuracy:.1%})")
    logger.info(f"  Grad-CAM layer    : {gcam.layer_name}")
    logger.info(f"  Plots saved       : {config.paths.plots_dir / 'xai'}")
    logger.info("=" * 60)

    _print_summary_table(pred_names, pred_probs, true_names, config.classes.names)


# ---------------------------------------------------------------------------
# CLI summary table
# ---------------------------------------------------------------------------

def _print_summary_table(
    pred_names:  list[str],
    pred_probs:  list[float],
    true_names:  list[str],
    class_names: list[str],
) -> None:
    print()
    print("=" * 56)
    print("  NeuroVision AI - Phase 4: XAI Results")
    print("=" * 56)
    print(f"  {'#':<4} {'Predicted':<15} {'Conf':>6}  {'GT':<15} {'OK'}")
    print("  " + "-" * 52)
    for i, (pred, prob, gt) in enumerate(zip(pred_names, pred_probs, true_names)):
        ok = "[OK]" if pred == gt else "[XX]"
        print(f"  {i:<4} {pred:<15} {prob:>6.1%}  {gt:<15} {ok}")
    print("=" * 56)
    correct = sum(p == t for p, t in zip(pred_names, true_names))
    print(f"  Accuracy: {correct}/{len(pred_names)} = {correct/len(pred_names):.1%}")
    print("=" * 56)
    print()


if __name__ == "__main__":
    main()
