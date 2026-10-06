import numpy as np
from ..detect.types import BoundingBox

def crop(image: np.ndarray, bbox: BoundingBox, copy: bool = False) -> np.ndarray:
    """
    Crops an HWC/HW ndarray using BoundingBox(left, top, right, bottom) coordinates.
    Guarantees boundary clamping and optional zero-copy slice view.
    """
    h, w = image.shape[:2]
    l = max(0, min(w, bbox.left))
    r = max(0, min(w, bbox.right))
    t = max(0, min(h, bbox.top))
    b = max(0, min(h, bbox.bottom))

    if l >= r or t >= b:
        # Empty slice
        if image.ndim == 3:
            return np.empty((0, 0, image.shape[2]), dtype=image.dtype)
        return np.empty((0, 0), dtype=image.dtype)

    sliced = image[t:b, l:r]
    return sliced.copy() if copy else sliced
