"""
Aspect-Ratio Preserving Smart Scaling and Quality Preset Module.
Open-Source under Apache License 2.0.
"""

import os
import tempfile
from enum import Enum
from typing import Union, Tuple, Optional
import numpy as np

from ..io.loader import load_image, save_image
from .functional import resize

class ImageQuality(str, Enum):
    ORIGINAL = "original"
    HIGH = "high"
    OPTIMAL = "optimal"
    FAST = "fast"

QUALITY_MAX_DIMS = {
    ImageQuality.ORIGINAL.value: None,
    ImageQuality.HIGH.value: 1280,
    ImageQuality.OPTIMAL.value: 768,
    ImageQuality.FAST.value: 384
}

def resolve_target_dimensions(
    orig_w: int,
    orig_h: int,
    mode: Union[str, ImageQuality] = ImageQuality.OPTIMAL,
    max_dim: Optional[int] = None
) -> Optional[Tuple[int, int]]:
    """
    Computes aspect-ratio preserving dimensions based on quality preset or custom max_dim.
    Returns None if no resizing is required (original mode or already smaller).
    """
    mode_str = mode.value if isinstance(mode, ImageQuality) else str(mode).lower().strip()
    
    target_limit = max_dim
    if target_limit is None:
        target_limit = QUALITY_MAX_DIMS.get(mode_str, 768)

    if target_limit is None:
        return None  # Original mode

    current_max = max(orig_w, orig_h)
    if current_max <= target_limit:
        return None  # Image already within bounds

    scale_ratio = float(target_limit) / float(current_max)
    new_w = max(16, int(round(orig_w * scale_ratio)))
    new_h = max(16, int(round(orig_h * scale_ratio)))
    return (new_w, new_h)

def prepare_image_for_inference(
    image: Union[str, np.ndarray],
    quality: Union[str, ImageQuality] = ImageQuality.OPTIMAL,
    max_dim: Optional[int] = None
) -> Tuple[str, bool]:
    """
    Prepares an image file path for VLM inference based on the requested quality preset.
    Returns (resolved_file_path, is_temporary).
    If is_temporary is True, the caller is responsible for deleting the file after inference.
    """
    mode_str = quality.value if isinstance(quality, ImageQuality) else str(quality).lower().strip()

    if isinstance(image, str):
        expanded = os.path.abspath(os.path.expanduser(image))
        if not os.path.exists(expanded):
            raise FileNotFoundError(f"Input image not found: '{expanded}'")

        if mode_str == ImageQuality.ORIGINAL.value and max_dim is None:
            return expanded, False

        raw_img = load_image(expanded)
    elif isinstance(image, np.ndarray):
        raw_img = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    h, w = raw_img.shape[:2]
    new_dims = resolve_target_dimensions(w, h, mode=mode_str, max_dim=max_dim)

    if new_dims is None:
        if isinstance(image, str):
            return expanded, False
        # Save ndarray to temp
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        save_image(raw_img, tmp_path, quality=92)
        return tmp_path, True

    # Scale with aspect ratio preserved
    scaled = resize(raw_img, new_dims)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    save_image(scaled, tmp_path, quality=92)
    return tmp_path, True
