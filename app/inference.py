"""
Inference pipeline: preprocessing and prediction logic.

All functions are pure (no Streamlit dependency) so they can be
unit-tested in isolation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from config import (
    CLASS_NAMES,
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


# ── Preprocessing ─────────────────────────────────────────────────────────

def get_transform() -> transforms.Compose:
    """Return the inference-time transform pipeline.

    Steps mirror standard ImageNet evaluation:
      1. Resize shortest edge to 256 px
      2. Center-crop to 224 × 224
      3. Convert to tensor ([0, 1] float)
      4. Normalise with ImageNet channel stats
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Prediction ────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: nn.Module, image: Image.Image) -> torch.Tensor:
    """Run inference on a single PIL image.

    Args:
        model: The loaded MobileNetV3-Large in eval mode.
        image: A PIL Image (any mode — will be converted to RGB).

    Returns:
        A 1-D tensor of softmax probabilities with shape ``(NUM_CLASSES,)``.
    """
    image = image.convert("RGB")
    tensor: torch.Tensor = get_transform()(image).unsqueeze(0).to(DEVICE)
    logits: torch.Tensor = model(tensor)
    probabilities: torch.Tensor = torch.softmax(logits, dim=1).squeeze(0).cpu()
    return probabilities


def get_top_k(
    probabilities: torch.Tensor,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Extract the top-*k* predictions from a probability vector.

    Args:
        probabilities: 1-D tensor of shape ``(NUM_CLASSES,)``.
        k: Number of top predictions to return (clamped to NUM_CLASSES).

    Returns:
        List of ``(class_name, confidence)`` tuples sorted by descending
        confidence.  ``confidence`` is a plain Python float in [0, 1].
    """
    k = min(k, NUM_CLASSES)
    top_probs, top_indices = torch.topk(probabilities, k)
    return [
        (CLASS_NAMES[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]


def get_all_probabilities(probabilities: torch.Tensor) -> Dict[str, float]:
    """Map every class name to its probability.

    Args:
        probabilities: 1-D tensor of shape ``(NUM_CLASSES,)``.

    Returns:
        Dict mapping class name → float probability.
    """
    return {
        name: probabilities[i].item()
        for i, name in enumerate(CLASS_NAMES)
    }
