# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
termux-vision: Lightweight On-Device Computer Vision & VLM Framework for Android Termux.
Open-Source under Apache License 2.0.
"""

__version__ = "1.8.0"
__author__ = "termux-vision contributors"
__license__ = "Apache-2.0"

from . import errors
from . import io
from . import transforms
from . import cv
from . import detect
from . import models
from . import bridge
from . import csrc
from . import vlm
from . import cli

from .detect.types import BoundingBox, Detection
from .detect.haar import detect_faces
from .models.embedding import Embedding, compute_similarity
from .vlm.api import load
from .hardware import HardwareProfile, detect_hardware, resolve_device
from .downloader import list_models, download_model, resolve_model_path
from . import cluster
from .exceptions import (
    AmevaTermuxError,
    TermuxVisionError,
    ClusterConnectionError,
    ClusterConfigurationError,
)
from .cluster import (
    parse_cluster_rpc_spec,
    verify_rpc_cluster_nodes,
    verify_rpc_cluster_health,
)

__all__ = [
    "io",
    "transforms",
    "cv",
    "detect",
    "models",
    "bridge",
    "csrc",
    "vlm",
    "cluster",
    "errors",
    "cli",
    "BoundingBox",
    "Detection",
    "detect_faces",
    "Embedding",
    "compute_similarity",
    "load",
    "HardwareProfile",
    "detect_hardware",
    "resolve_device",
    "list_models",
    "download_model",
    "resolve_model_path",
    "AmevaTermuxError",
    "TermuxVisionError",
    "ClusterConnectionError",
    "ClusterConfigurationError",
    "parse_cluster_rpc_spec",
    "verify_rpc_cluster_nodes",
    "verify_rpc_cluster_health",
    "__version__"
]

