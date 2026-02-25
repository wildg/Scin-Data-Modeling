"""Backbone factory for feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models, transforms

# Registry of supported backbones: name → (constructor, embed_dim)
_REGISTRY: dict[str, tuple] = {}


def _register(name: str, embed_dim: int):
    """Decorator that registers a backbone builder."""

    def wrapper(fn):
        _REGISTRY[name] = (fn, embed_dim)
        return fn

    return wrapper


# ── Backbone builders ──────────────────────────────────────────────────────────


@_register("resnet50", embed_dim=2048)
def _resnet50(pretrained: bool) -> tuple[nn.Module, transforms.Compose]:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    base = models.resnet50(weights=weights)
    # Remove the final FC layer → output is the 2048-dim avg-pool vector
    model = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
    transform = weights.transforms() if weights else _default_transform()
    return model, transform


@_register("efficientnet_b0", embed_dim=1280)
def _efficientnet_b0(pretrained: bool) -> tuple[nn.Module, transforms.Compose]:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    base = models.efficientnet_b0(weights=weights)
    # Remove the classifier → output is the 1280-dim pooled vector
    model = nn.Sequential(base.features, base.avgpool, nn.Flatten())
    transform = weights.transforms() if weights else _default_transform()
    return model, transform


# ── Public API ─────────────────────────────────────────────────────────────────


def get_backbone(
    name: str = "resnet50",
    pretrained: bool = True,
) -> tuple[nn.Module, transforms.Compose, int]:
    """Return ``(model, transform, embed_dim)`` for the named backbone.

    The model is set to eval mode with all parameters frozen
    (no gradients tracked).  Undo this manually if you need to fine-tune.

    Parameters
    ----------
    name:
        One of the registered backbone names (see :func:`list_backbones`).
    pretrained:
        Whether to load ImageNet-pretrained weights.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown backbone {name!r}. Available: {available}")

    builder, embed_dim = _REGISTRY[name]
    model, transform = builder(pretrained)

    # Freeze by default — caller can unfreeze for fine-tuning
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, transform, embed_dim


def list_backbones() -> list[str]:
    """Return the names of all registered backbones."""
    return sorted(_REGISTRY)


def _default_transform() -> transforms.Compose:
    """Fallback transform when no pretrained weights are used."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
