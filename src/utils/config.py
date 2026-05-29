"""
Configuration loader: merges config.yaml with environment variables.

Design: load once at startup → pass a frozen AppConfig dataclass through
the call stack. No code anywhere reads YAML or os.environ directly —
only this module does. This prevents scattered config access and makes
testing trivial (just inject a different AppConfig).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Typed dataclasses — one per config section
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassConfig:
    names: list[str]
    num_classes: int
    label_map: dict[str, int]


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: tuple[int, int]
    channels: int
    mean: list[float]
    std: list[float]
    clip_percentile: list[float]


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    horizontal_flip: bool
    vertical_flip: bool
    rotation_limit: int
    shift_limit: float
    scale_limit: float
    brightness_limit: float
    contrast_limit: float
    blur_limit: int
    gaussian_noise_variance: list[float]
    elastic_transform: bool
    elastic_alpha: int
    elastic_sigma: int
    grid_distortion: bool
    augmentation_probability: float


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    stratified: bool


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    initial_learning_rate: float
    fine_tune_learning_rate: float
    warmup_epochs: int
    early_stopping_patience: int
    reduce_lr_patience: int
    reduce_lr_factor: float
    min_lr: float
    label_smoothing: float
    dropout_rate: float
    l2_regularization: float
    mixed_precision: bool
    freeze_base_epochs: int


@dataclass(frozen=True)
class PipelineConfig:
    prefetch_buffer: int
    shuffle_buffer: int
    cache_in_memory: bool
    num_parallel_calls: int


@dataclass(frozen=True)
class SegmentationConfig:
    architecture: str
    input_size: tuple[int, int]
    channels: int
    filters: list[int]
    bottleneck_filters: int
    dropout: float
    batch_norm: bool
    batch_size: int
    epochs: int
    learning_rate: float
    min_lr: float
    early_stopping_patience: int
    reduce_lr_patience: int
    reduce_lr_factor: float
    loss: str
    tversky_alpha: float
    tversky_beta: float
    pseudo_mask_blur: int
    pseudo_mask_morph_size: int


@dataclass(frozen=True)
class XAIConfig:
    gradcam_layer: str
    overlay_alpha: float
    colormap: str


@dataclass(frozen=True)
class PathConfig:
    dataset_root: Path
    project_root: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    outputs_dir: Path
    plots_dir: Path
    logs_dir: Path
    models_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    version: str
    classes: ClassConfig
    preprocessing: PreprocessingConfig
    segmentation: SegmentationConfig
    augmentation: AugmentationConfig
    split: SplitConfig
    training: TrainingConfig
    pipeline: PipelineConfig
    paths: PathConfig
    model: dict  # Raw model config dict (backbone, ensemble list, etc.)
    xai: XAIConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load and validate configuration.

    Resolution order (last wins):
    1. config.yaml defaults
    2. .env file overrides
    3. Environment variables

    Args:
        config_path: Path to config.yaml. Defaults to
                     <project_root>/configs/config.yaml resolved from
                     PROJECT_ROOT env var, or the directory containing
                     this file's grandparent.
    """
    load_dotenv()  # Load .env if present

    # Resolve config.yaml path
    if config_path is None:
        project_root_env = os.getenv("PROJECT_ROOT")
        if project_root_env:
            config_path = Path(project_root_env) / "configs" / "config.yaml"
        else:
            # Fallback: assume standard layout relative to this file
            config_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    logger.debug(f"Loaded config from {config_path}")

    # --- Resolve env var overrides ---
    dataset_root = Path(
        os.getenv("DATASET_ROOT", "")
        or _raise(ValueError("DATASET_ROOT environment variable is not set. "
                             "Copy .env.example to .env and fill in paths."))
    )

    project_root = Path(
        os.getenv("PROJECT_ROOT", str(config_path.parents[1]))
    )

    # Validate dataset directories exist
    for split_name in ("train", "valid", "test"):
        split_path = dataset_root / split_name
        if not split_path.exists():
            raise FileNotFoundError(
                f"Expected dataset split directory not found: {split_path}\n"
                f"Verify DATASET_ROOT={dataset_root} points to the dataset root."
            )

    outputs_dir = project_root / "outputs"
    paths = PathConfig(
        dataset_root=dataset_root,
        project_root=project_root,
        train_dir=dataset_root / "train",
        val_dir=dataset_root / "valid",
        test_dir=dataset_root / "test",
        outputs_dir=outputs_dir,
        plots_dir=outputs_dir / "plots",
        logs_dir=outputs_dir / "logs",
        models_dir=project_root / "models",
        reports_dir=outputs_dir / "reports",
    )

    # Create output directories
    for d in (paths.plots_dir, paths.logs_dir, paths.models_dir, paths.reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Build config objects ---
    classes_raw = raw["classes"]
    classes = ClassConfig(
        names=classes_raw["names"],
        num_classes=classes_raw["num_classes"],
        label_map=classes_raw["label_map"],
    )

    pp = raw["preprocessing"]
    preprocessing = PreprocessingConfig(
        image_size=tuple(pp["image_size"]),
        channels=pp["channels"],
        mean=pp["mean"],
        std=pp["std"],
        clip_percentile=pp["clip_percentile"],
    )

    aug = raw["augmentation"]
    augmentation = AugmentationConfig(
        enabled=aug["enabled"],
        horizontal_flip=aug["horizontal_flip"],
        vertical_flip=aug["vertical_flip"],
        rotation_limit=aug["rotation_limit"],
        shift_limit=aug["shift_limit"],
        scale_limit=aug["scale_limit"],
        brightness_limit=aug["brightness_limit"],
        contrast_limit=aug["contrast_limit"],
        blur_limit=aug["blur_limit"],
        gaussian_noise_variance=aug["gaussian_noise_variance"],
        elastic_transform=aug["elastic_transform"],
        elastic_alpha=aug["elastic_alpha"],
        elastic_sigma=aug["elastic_sigma"],
        grid_distortion=aug["grid_distortion"],
        augmentation_probability=aug["augmentation_probability"],
    )

    sp = raw["split"]
    split = SplitConfig(
        train_ratio=sp["train_ratio"],
        val_ratio=sp["val_ratio"],
        test_ratio=sp["test_ratio"],
        random_seed=sp["random_seed"],
        stratified=sp["stratified"],
    )

    tr = raw["training"]
    training = TrainingConfig(
        batch_size=int(os.getenv("BATCH_SIZE", tr["batch_size"])),
        epochs=tr["epochs"],
        initial_learning_rate=tr["initial_learning_rate"],
        fine_tune_learning_rate=tr["fine_tune_learning_rate"],
        warmup_epochs=tr["warmup_epochs"],
        early_stopping_patience=tr["early_stopping_patience"],
        reduce_lr_patience=tr["reduce_lr_patience"],
        reduce_lr_factor=tr["reduce_lr_factor"],
        min_lr=tr["min_lr"],
        label_smoothing=tr["label_smoothing"],
        dropout_rate=tr["dropout_rate"],
        l2_regularization=tr["l2_regularization"],
        mixed_precision=tr["mixed_precision"],
        freeze_base_epochs=tr.get("freeze_base_epochs", 10),
    )

    pl = raw["pipeline"]
    pipeline = PipelineConfig(
        prefetch_buffer=pl["prefetch_buffer"],
        shuffle_buffer=pl["shuffle_buffer"],
        cache_in_memory=pl["cache_in_memory"],
        num_parallel_calls=pl["num_parallel_calls"],
    )

    sg = raw.get("segmentation", {})
    segmentation = SegmentationConfig(
        architecture=sg.get("architecture", "AttentionUNet"),
        input_size=tuple(sg.get("input_size", [256, 256])),
        channels=sg.get("channels", 1),
        filters=sg.get("filters", [64, 128, 256, 512]),
        bottleneck_filters=sg.get("bottleneck_filters", 1024),
        dropout=sg.get("dropout", 0.30),
        batch_norm=sg.get("batch_norm", True),
        batch_size=sg.get("batch_size", 16),
        epochs=sg.get("epochs", 50),
        learning_rate=sg.get("learning_rate", 1e-4),
        min_lr=sg.get("min_lr", 1e-8),
        early_stopping_patience=sg.get("early_stopping_patience", 10),
        reduce_lr_patience=sg.get("reduce_lr_patience", 5),
        reduce_lr_factor=sg.get("reduce_lr_factor", 0.5),
        loss=sg.get("loss", "bce_dice"),
        tversky_alpha=sg.get("tversky_alpha", 0.3),
        tversky_beta=sg.get("tversky_beta", 0.7),
        pseudo_mask_blur=sg.get("pseudo_mask_blur", 5),
        pseudo_mask_morph_size=sg.get("pseudo_mask_morph_size", 7),
    )

    xai_raw = raw.get("xai", {})
    xai = XAIConfig(
        gradcam_layer=xai_raw.get("gradcam_layer", "top_conv"),
        overlay_alpha=xai_raw.get("overlay_alpha", 0.45),
        colormap=xai_raw.get("colormap", "jet"),
    )

    config = AppConfig(
        project_name=raw["project"]["name"],
        version=raw["project"]["version"],
        classes=classes,
        preprocessing=preprocessing,
        augmentation=augmentation,
        split=split,
        training=training,
        pipeline=pipeline,
        paths=paths,
        model=raw.get("model", {}),
        segmentation=segmentation,
        xai=xai,
    )

    logger.info(
        f"Config loaded | project={config.project_name} v{config.version} | "
        f"dataset={dataset_root} | classes={classes.num_classes}"
    )
    return config


def _raise(exc: Exception) -> None:
    """Helper to raise inside lambda/walrus context."""
    raise exc
