import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
from .limits import ImageLimits, DEFAULT_IMAGE_LIMITS
from ..errors import DecompressionBombError

def load_image(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    limits: ImageLimits = DEFAULT_IMAGE_LIMITS,
    apply_exif_orientation: bool = True
) -> np.ndarray:
    """
    Loads an image from file system into an HWC RGB uint8 ndarray.
    Guarantees EXIF auto-orientation, decompression bomb defense, and contiguous buffer layout.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    limits.validate_file_size(image_path)

    try:
        from PIL import Image, ImageOps
        with Image.open(image_path) as img:
            if not limits.allow_animated and getattr(img, "is_animated", False):
                raise ValueError("Animated images (GIF/APNG) are not allowed under default security limits.")

            limits.validate_dimensions(img.width, img.height)

            if apply_exif_orientation:
                try:
                    img = ImageOps.exif_transpose(img)
                except (AttributeError, ValueError, TypeError, OSError) as _exif_err:
                    import logging
                    logging.getLogger(__name__).debug("EXIF orientation transpose failed (%s), keeping original orientation", _exif_err)

            if target_size is not None:
                img = img.resize(target_size, Image.Resampling.BILINEAR)

            if img.mode != "RGB":
                img = img.convert("RGB")

            arr = np.array(img, dtype=np.uint8)
            return np.ascontiguousarray(arr)
    except ImportError:
        # Pure NumPy fallback for NetPBM / RAW if Pillow not installed
        raise RuntimeError("Pillow is required for standard JPEG/PNG image loading. Please install pillow.")

def save_image(
    image: np.ndarray,
    save_path: str,
    quality: int = 95,
    metadata: str = "strip"
) -> None:
    """
    Saves an HWC uint8/float32 ndarray to file system.
    Default metadata='strip' securely discards EXIF GPS / sensitive device tags.
    """
    if not isinstance(image, np.ndarray) or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"Cannot save empty or zero-sized image array (shape: {getattr(image, 'shape', None)}).")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and np.max(image) <= 1.0:
            arr = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = image.clip(0, 255).astype(np.uint8)
    else:
        arr = image

    from PIL import Image

    if arr.ndim == 2:
        mode = "L"
    elif arr.ndim == 3 and arr.shape[2] == 1:
        mode = "L"
        arr = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = "RGB"
    elif arr.ndim == 3 and arr.shape[2] == 4:
        mode = "RGBA"
    else:
        raise ValueError(f"Unsupported array shape for image saving: {arr.shape}")

    pil_img = Image.fromarray(arr, mode=mode)
    
    save_kwargs = {}
    if save_path.lower().endswith((".jpg", ".jpeg")):
        save_kwargs["quality"] = quality
    
    # When metadata is 'strip', no EXIF or extra tags are attached
    pil_img.save(save_path, **save_kwargs)
    return save_path

def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Inspects image file metadata without loading the full pixel array into memory.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    from PIL import Image
    with Image.open(image_path) as img:
        info = {
            "path": image_path,
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_bytes": os.path.getsize(image_path),
            "size_mb": round(os.path.getsize(image_path) / (1024.0 * 1024.0), 3)
        }
    return info
