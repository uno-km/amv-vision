# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
from .manifest import ModelManifest, ArtifactInfo
from .cache import ModelCacheManager, CATALOG
from .result import VLMResult, InferenceMetrics
from .api import load, VLMContext
from .protocol import TVRPRequest, TVRPResponse, ProtocolError
from .supervisor import RuntimePolicy, StderrDrainer


def get_models() -> ModelCacheManager:
    """Get or initialize the ModelCacheManager lazily."""
    return ModelCacheManager()


def __getattr__(name: str):
    if name == "models":
        return ModelCacheManager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelManifest",
    "ArtifactInfo",
    "ModelCacheManager",
    "CATALOG",
    "VLMResult",
    "InferenceMetrics",
    "load",
    "VLMContext",
    "get_models",
    "models",
    "TVRPRequest",
    "TVRPResponse",
    "ProtocolError",
    "RuntimePolicy",
    "StderrDrainer"
]
