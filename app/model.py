"""
Model definition and checkpoint loading for MobileNetV3-Large (23-class).

This module builds the correct torchvision architecture and provides a
robust loader that handles:
  • Raw state_dict (collections.OrderedDict)
  • Wrapped checkpoint dicts  ({"model_state_dict": ..., ...})
  • Full model objects saved with torch.save(model, ...)

The loader is decorated with @st.cache_resource so the model is loaded
exactly once per Streamlit server lifetime.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Union

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models

from config import DEVICE, MODEL_PATH, NUM_CLASSES

logger = logging.getLogger(__name__)


# ── Architecture ──────────────────────────────────────────────────────────

def build_mobilenetv3_large(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Construct a MobileNetV3-Large with the classifier head modified
    for *num_classes* outputs.

    The default torchvision classifier for MobileNetV3-Large is::

        Sequential(
            0: Linear(960, 1280),
            1: Hardswish(),
            2: Dropout(p=0.2),
            3: Linear(1280, 1000),
        )

    We replace layer 3 with ``Linear(1280, num_classes)``.
    """
    model = models.mobilenet_v3_large(weights=None)
    in_features: int = model.classifier[3].in_features  # 1280
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


# ── Checkpoint loading ────────────────────────────────────────────────────

def _extract_state_dict(checkpoint: object) -> OrderedDict:
    """Normalise whatever ``torch.load`` returns into a plain state_dict.

    Raises:
        TypeError: If the loaded object is not a dict-like or nn.Module.
        KeyError:  If a wrapped dict uses an unrecognised key layout.
    """
    # Case 1: raw state_dict (OrderedDict / dict)
    if isinstance(checkpoint, (dict, OrderedDict)):
        # Check for common wrapper keys
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                logger.info("Detected wrapped checkpoint with key '%s'.", key)
                inner = checkpoint[key]
                if isinstance(inner, nn.Module):
                    return inner.state_dict()
                return OrderedDict(inner)

        # If the dict contains typical parameter tensor keys, treat as raw
        sample_key = next(iter(checkpoint), "")
        if "." in sample_key:  # e.g. "features.0.0.weight"
            logger.info("Loaded raw state_dict (%d keys).", len(checkpoint))
            return OrderedDict(checkpoint)

        raise KeyError(
            f"Checkpoint dict has unrecognised top-level keys: "
            f"{list(checkpoint.keys())[:10]}"
        )

    # Case 2: full model object
    if isinstance(checkpoint, nn.Module):
        logger.info("Loaded full nn.Module object.")
        return checkpoint.state_dict()

    raise TypeError(
        f"Unsupported checkpoint type: {type(checkpoint).__name__}. "
        "Expected state_dict, wrapped dict, or nn.Module."
    )


def _validate_state_dict(state_dict: OrderedDict, model: nn.Module) -> None:
    """Verify that the state_dict is compatible with *model*.

    Raises:
        ValueError: On shape mismatches or missing/unexpected keys.
    """
    model_sd = model.state_dict()

    missing = set(model_sd.keys()) - set(state_dict.keys())
    unexpected = set(state_dict.keys()) - set(model_sd.keys())

    if missing:
        raise ValueError(
            f"Checkpoint is missing {len(missing)} expected key(s). "
            f"First few: {sorted(missing)[:5]}"
        )
    if unexpected:
        logger.warning(
            "Checkpoint contains %d unexpected key(s) (will be ignored): %s",
            len(unexpected),
            sorted(unexpected)[:5],
        )

    # Shape check on matching keys
    for key in model_sd:
        if key in state_dict and model_sd[key].shape != state_dict[key].shape:
            raise ValueError(
                f"Shape mismatch for '{key}': "
                f"model expects {model_sd[key].shape}, "
                f"checkpoint has {state_dict[key].shape}."
            )


@st.cache_resource(show_spinner=False)
def load_model() -> nn.Module:
    """Load the MobileNetV3-Large checkpoint and return the model in eval mode.

    The result is cached by Streamlit so successive calls are free.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError / KeyError / TypeError: On incompatible checkpoints.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Ensure the .pth file is placed correctly."
        )

    logger.info("Loading checkpoint from %s …", MODEL_PATH)
    raw = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    state_dict = _extract_state_dict(raw)

    model = build_mobilenetv3_large(NUM_CLASSES)
    _validate_state_dict(state_dict, model)
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()
    logger.info(
        "Model loaded successfully on %s (%d parameters).",
        DEVICE,
        sum(p.numel() for p in model.parameters()),
    )
    return model
