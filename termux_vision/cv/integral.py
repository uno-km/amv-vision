import numpy as np
from ..transforms.functional import to_grayscale

def compute_integral_image(image: np.ndarray) -> np.ndarray:
    """
    Compute 2D Integral Image (Summed Area Table) with leading zero padding (H+1, W+1).
    Uses compiled C kernel when available, falls back to NumPy cumsum.
    """
    gray = to_grayscale(image)

    # 1. Accelerated C Path
    try:
        from ..csrc.backend import has_c_backend, c_compute_integral
        if has_c_backend():
            return c_compute_integral(gray)
    except Exception as e:
        import logging
        logging.getLogger("termux_vision.cv.integral").debug(
            "[termux-vision] C compute_integral failed, falling back to NumPy: %s", e
        )

    # 2. Pure NumPy Path
    gray_f = gray.astype(np.float64)
    h, w = gray_f.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(gray_f, axis=0), axis=1)
    return integral

def box_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """
    O(1) calculation of the sum of pixel values in bounding box [y1:y2, x1:x2].
    """
    return (
        integral[y2, x2]
        - integral[y1, x2]
        - integral[y2, x1]
        + integral[y1, x1]
    )

def box_filter(image: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    O(1) per-pixel box blur filter using integral image representation.
    """
    gray = to_grayscale(image).astype(np.float64)
    h, w = gray.shape
    integral = compute_integral_image(gray)
    out = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)
        for x in range(w):
            x1 = max(0, x - radius)
            x2 = min(w, x + radius + 1)
            area = (y2 - y1) * (x2 - x1)
            out[y, x] = box_sum(integral, x1, y1, x2, y2) / area

    if np.issubdtype(image.dtype, np.integer):
        return np.clip(out, 0, 255).astype(np.uint8)
    return out
