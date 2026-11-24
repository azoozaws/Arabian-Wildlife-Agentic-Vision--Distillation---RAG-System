"""Split YOLO training, Champion-vs-Challenger evaluation, and ModelRegistry services."""

from app.src.training.model_registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_PATH,
    ModelRegistry,
)
from app.src.training.yolo_trainer import (
    YOLOTrainer,
    compute_fp_per_hour,
    configure_wandb_env,
    evaluate_champion_vs_challenger,
    evaluate_single_model,
)

__all__ = [
    "DEFAULT_MODELS_DIR",
    "DEFAULT_REGISTRY_PATH",
    "ModelRegistry",
    "YOLOTrainer",
    "configure_wandb_env",
    "compute_fp_per_hour",
    "evaluate_single_model",
    "evaluate_champion_vs_challenger",
]
