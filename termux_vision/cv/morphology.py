import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def structuring_element(shape: str = "rect", size: int = 3) -> np.ndarray:
    if shape == "rect":
        return np.ones((size, size), dtype=bool)
    elif shape == "cross":
        k = np.zeros((size, size), dtype=bool)
        center = size // 2
        k[center, :] = True
        k[:, center] = True
        return k
    return np.ones((size, size), dtype=bool)

def dilate(binary_image: np.ndarray, kernel: np.ndarray = None, iterations: int = 1) -> np.ndarray:
    """Morphological Dilation with C backend fast path."""
    if kernel is None and iterations == 1:
        try:
            from ..csrc.backend import has_c_backend, c_morphology
            if has_c_backend():
                return c_morphology(binary_image, is_dilate=True)
        except Exception as e:
            import logging
            logging.getLogger("termux_vision.cv.morphology").debug(
                "[termux-vision] C dilate failed, falling back to NumPy: %s", e
            )

    if kernel is None:
        kernel = structuring_element("rect", 3)
        
    img = (binary_image > 0).astype(bool)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    for _ in range(iterations):
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=False)
        windows = sliding_window_view(padded, (kh, kw))
        img = np.any(windows & kernel, axis=(-2, -1))

    return (img * 255).astype(np.uint8)

def erode(binary_image: np.ndarray, kernel: np.ndarray = None, iterations: int = 1) -> np.ndarray:
    """Morphological Erosion with C backend fast path."""
    if kernel is None and iterations == 1:
        try:
            from ..csrc.backend import has_c_backend, c_morphology
            if has_c_backend():
                return c_morphology(binary_image, is_dilate=False)
        except Exception as e:
            import logging
            logging.getLogger("termux_vision.cv.morphology").debug(
                "[termux-vision] C erode failed, falling back to NumPy: %s", e
            )

    if kernel is None:
        kernel = structuring_element("rect", 3)
        
    img = (binary_image > 0).astype(bool)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    for _ in range(iterations):
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=True)
        windows = sliding_window_view(padded, (kh, kw))
        matches = np.all(windows[:, :] | (~kernel), axis=(-2, -1))
        img = matches

    return (img * 255).astype(np.uint8)

def morph_open(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    return dilate(erode(image, kernel), kernel)

def morph_close(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    return erode(dilate(image, kernel), kernel)
