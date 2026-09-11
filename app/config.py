"""
Centralized configuration for the Animal Classification application.

All constants, paths, and device selection logic live here so that
other modules never contain magic numbers or hardcoded paths.
"""

from __future__ import annotations

import logging
import pathlib
from typing import List, Tuple

import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL: int = logging.INFO

# ---------------------------------------------------------------------------
# Paths  (resolved relative to *this* file → …/app/config.py)
# ---------------------------------------------------------------------------
_APP_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT: pathlib.Path = _APP_DIR.parent
MODEL_PATH: pathlib.Path = PROJECT_ROOT / "model" / "animal_classification_mobilenetv3.pth"

# ---------------------------------------------------------------------------
# Class labels — ground-truth order, must not be reordered
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = [
    "badger",
    "bison",
    "boar",
    "cheetah",
    "chimpanzee",
    "cougar",
    "giraffe",
    "gorilla",
    "hippopotamus",
    "jaguar",
    "koala",
    "leopard",
    "lion",
    "llama",
    "orangutan",
    "snow_leopard",
    "tiger",
    "weasel",
    "wombat",
]

NUM_CLASSES: int = len(CLASS_NAMES)  # 19

# ---------------------------------------------------------------------------
# Preprocessing — matches ImageNet normalisation used during training
# ---------------------------------------------------------------------------
IMG_SIZE: int = 224
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Device selection (CUDA → MPS → CPU)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE: torch.device = get_device()

# ---------------------------------------------------------------------------
# Upload constraints
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS: Tuple[str, ...] = ("jpg", "jpeg", "png")
MAX_UPLOAD_MB: int = 10
