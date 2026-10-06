import numpy as np
from PIL import Image

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB or BGR image (H, W, 3) to single channel Grayscale (H, W) or (H, W, 1).
    Uses standard luminance weighting: 0.2989 * R + 0.5870 * G + 0.1140 * B
    """
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 1:
        return image[:, :, 0].copy()
    
    # Fast vectorized dot product
    gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    if np.issubdtype(image.dtype, np.integer):
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

def to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert Grayscale (H, W) or (H, W, 1) or RGBA (H, W, 4) to standard 3-channel RGB (H, W, 3).
    """
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] == 4:
        return image[..., :3].copy()
    return image.copy()

def to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image (H, W, 3) to BGR format for compatibility with OpenCV algorithms.
    """
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    return image[..., [2, 1, 0]].copy()

def resize(image: np.ndarray, size: tuple, interpolation: str = "bilinear") -> np.ndarray:
    """
    Resize image array to (width, height) using PIL acceleration.
    size: (width, height) or integer for square resize.
    """
    if not isinstance(image, np.ndarray) or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"Cannot resize empty or zero-sized image array (shape: {getattr(image, 'shape', None)}).")

    if isinstance(size, int):
        target_w, target_h = size, size
    else:
        target_w, target_h = size

    resample_map = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS
    }
    resample = resample_map.get(interpolation.lower(), Image.Resampling.BILINEAR)

    is_float = np.issubdtype(image.dtype, np.floating)
    if is_float and image.max() <= 1.0:
        pil_img = Image.fromarray((image * 255.0).astype(np.uint8))
        resized = pil_img.resize((target_w, target_h), resample=resample)
        return np.array(resized, dtype=np.float32) / 255.0
    else:
        pil_img = Image.fromarray(image.astype(np.uint8))
        resized = pil_img.resize((target_w, target_h), resample=resample)
        return np.array(resized, dtype=image.dtype)

def crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Crop bounding box region [y:y+h, x:x+w] with safe boundary clipping.
    """
    img_h, img_w = image.shape[:2]
    x1 = max(0, min(x, img_w))
    y1 = max(0, min(y, img_h))
    x2 = max(x1, min(x + w, img_w))
    y2 = max(y1, min(y + h, img_h))
    return image[y1:y2, x1:x2].copy()

def center_crop(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    Crop central region of given (width, height).
    """
    if isinstance(size, int):
        target_w, target_h = size, size
    else:
        target_w, target_h = size

    img_h, img_w = image.shape[:2]
    x = max(0, (img_w - target_w) // 2)
    y = max(0, (img_h - target_h) // 2)
    return crop(image, x, y, target_w, target_h)

def normalize(image: np.ndarray, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Normalize image tensor (H, W, C) or (C, H, W) to float32 with mean and std.
    """
    arr = image.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    
    if arr.ndim == 3 and arr.shape[2] == len(mean):
        return (arr - mean_arr) / std_arr
    elif arr.ndim == 3 and arr.shape[0] == len(mean):
        return (arr - mean_arr[:, None, None]) / std_arr[:, None, None]
    else:
        # 1-channel grayscale
        return (arr - np.mean(mean_arr)) / np.mean(std_arr)
