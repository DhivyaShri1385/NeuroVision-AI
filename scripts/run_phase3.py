"""
Phase 3 Orchestration — Segmentation Training
=============================================

Trains a U-Net or Attention U-Net segmentation model on pseudo-masks derived
from the brain-tumor classification dataset.

Pipeline:
  1. Load config + logger
  2. Generate pseudo-masks (Otsu + morphology) — skips existing masks
  3. Build tf.data pipelines (image + mask pairs)
  4. Build model (UNet | AttentionUNet, controlled by config or --arch flag)
  5. Train with Dice + IoU callbacks
  6. Evaluate on test split
  7. Generate visualisations (history, overlays, metrics bar, sample grid)
  8. Print results summary

CLI flags:
  --arch           UNet | AttentionUNet   (overrides config.segmentation.architecture)
  --quick-test     Run 2 epochs, 1 image/class for smoke-testing
  --no-masks       Skip mask generation (use pre-existing masks)
  --masks-root     Custom path for pseudo-mask directory

Examples:
  python scripts/run_phase3.py
  python scripts/run_phase3.py --arch UNet
  python scripts/run_phase3.py --quick-test
  python scripts/run_phase3.py --no-masks --masks-root /custom/masks/path
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

# ── Ensure project root is on sys.path ──────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.utils.config import load_config, AppConfig
from src.utils.logger import logger, setup_logger
from src.segmentation.mask_generator  import PseudoMaskGenerator
from src.segmentation.data_loader     import SegmentationDataLoader
from src.segmentation.attention_unet  import build_segmentation_model
from src.segmentation.trainer         import SegmentationTrainer
from src.segmentation.visualizer      import SegmentationVisualizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroVision AI — Phase 3 Segmentation")
    p.add_argument("--arch", choices=["UNet", "AttentionUNet"], default=None,
                   help="Override architecture from config.")
    p.add_argument("--quick-test", action="store_true",
                   help="2-epoch smoke test on a tiny subset.")
    p.add_argument("--no-masks", action="store_true",
                   help="Skip mask generation; use existing masks.")
    p.add_argument("--masks-root", default=None,
                   help="Custom path to pseudo-mask root directory.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------

def _patch_config_for_quick_test(config: AppConfig) -> AppConfig:
    """Override epochs to 2 for a fast smoke test."""
    new_seg = replace(config.segmentation, epochs=2)
    return replace(config, segmentation=new_seg)


def _patch_config_arch(config: AppConfig, arch: str) -> AppConfig:
    new_seg = replace(config.segmentation, architecture=arch)
    return replace(config, segmentation=new_seg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = _parse_args()
    config = load_config()

    # Setup logger
    setup_logger(
        log_dir=config.paths.logs_dir,
        level="DEBUG" if args.quick_test else "INFO",
    )

    # Patch config
    if args.arch:
        config = _patch_config_arch(config, args.arch)
    if args.quick_test:
        config = _patch_config_for_quick_test(config)
        logger.warning("quick-test mode: 2 epochs only")

    arch_name  = config.segmentation.architecture
    masks_root = Path(args.masks_root) if args.masks_root else (
        config.paths.outputs_dir / "pseudo_masks"
    )

    logger.info("=" * 60)
    logger.info(f"NeuroVision AI — Phase 3 | arch={arch_name}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Pseudo-mask generation
    # ------------------------------------------------------------------
    if not args.no_masks:
        logger.info("[1/6] Generating pseudo-masks ...")
        gen = PseudoMaskGenerator(config, masks_root=masks_root)
        for split in ("train", "val", "test"):
            gen.generate_for_split(split)
    else:
        logger.info("[1/6] Skipping mask generation (--no-masks)")

    # ------------------------------------------------------------------
    # Step 2 — Build tf.data pipelines
    # ------------------------------------------------------------------
    logger.info("[2/6] Building data pipelines ...")
    loader = SegmentationDataLoader(config, masks_root=masks_root)
    train_ds = loader.get_train_dataset()
    val_ds   = loader.get_val_dataset()
    test_ds  = loader.get_test_dataset()

    # Smoke-check shapes
    for imgs, masks in train_ds.take(1):
        logger.info(
            f"Train batch | images={imgs.shape} dtype={imgs.dtype} "
            f"| masks={masks.shape} dtype={masks.dtype}"
        )

    # ------------------------------------------------------------------
    # Step 3 — Build model
    # ------------------------------------------------------------------
    logger.info(f"[3/6] Building {arch_name} ...")
    model = build_segmentation_model(config)
    model.summary(print_fn=lambda x: logger.debug(x), expand_nested=False)

    # ------------------------------------------------------------------
    # Step 4 — Train
    # ------------------------------------------------------------------
    logger.info("[4/6] Training ...")
    t0 = time.time()
    trainer = SegmentationTrainer(
        config, model, train_ds, val_ds,
        model_dir=config.paths.models_dir,
    )
    history = trainer.train()
    elapsed = time.time() - t0
    logger.info(f"Training took {elapsed / 60:.1f} min")

    # ------------------------------------------------------------------
    # Step 5 — Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("[5/6] Evaluating on test set ...")
    results = trainer.evaluate(test_ds)

    # ------------------------------------------------------------------
    # Step 6 — Visualise
    # ------------------------------------------------------------------
    logger.info("[6/6] Generating visualisations ...")
    viz = SegmentationVisualizer(config)

    # Training curves
    viz.plot_training_history(history, arch_name=arch_name)

    # Collect test predictions for visual plots
    imgs_list: list[np.ndarray]  = []
    true_list: list[np.ndarray]  = []
    pred_list: list[np.ndarray]  = []
    for imgs, masks in test_ds.take(4):
        preds = model(imgs, training=False).numpy()
        imgs_list.append(imgs.numpy())
        true_list.append(masks.numpy())
        pred_list.append(preds)

    imgs_np  = np.concatenate(imgs_list, axis=0)
    true_np  = np.concatenate(true_list, axis=0)
    pred_np  = np.concatenate(pred_list, axis=0)

    viz.plot_mask_overlays(imgs_np, true_np, pred_np,
                           n_samples=6, arch_name=arch_name)
    viz.plot_metrics_bar(results, arch_name=arch_name)
    viz.plot_sample_predictions(model, test_ds, n_samples=6,
                                arch_name=arch_name)

    # Save final model weights
    trainer.save_final_model()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Phase 3 Complete — {arch_name}")
    logger.info("=" * 60)
    logger.info(f"  Dice        : {results['dice']:.4f}")
    logger.info(f"  IoU         : {results['iou']:.4f}")
    logger.info(f"  Pixel Acc   : {results['pixel_acc']:.4f}")
    logger.info(f"  Precision   : {results['precision']:.4f}")
    logger.info(f"  Recall      : {results['recall']:.4f}")
    logger.info(f"  F1          : {results['f1']:.4f}")
    logger.info(f"  Plots saved : {config.paths.plots_dir / 'segmentation'}")
    logger.info(f"  Model saved : {config.paths.models_dir}")
    logger.info("=" * 60)

    _print_summary_table(results, arch_name)


def _print_summary_table(results: dict[str, float], arch_name: str) -> None:
    print()
    print("=" * 50)
    print(f"  NeuroVision AI - Phase 3: {arch_name}")
    print("=" * 50)
    rows = [
        ("Dice Coefficient", results.get("dice",      0.0)),
        ("IoU / Jaccard",    results.get("iou",       0.0)),
        ("Pixel Accuracy",   results.get("pixel_acc", 0.0)),
        ("Precision",        results.get("precision", 0.0)),
        ("Recall",           results.get("recall",    0.0)),
        ("F1 Score",         results.get("f1",        0.0)),
    ]
    for name, val in rows:
        bar   = "#" * int(val * 20)
        print(f"  {name:<20} {val:.4f}  [{bar:<20}]")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()
