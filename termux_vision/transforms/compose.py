from typing import Callable, List, Tuple
import numpy as np
from . import functional as F

class Compose:
    """
    Composes several transforms together into a single pipeline.
    """
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            img = t(img)
        return img

class Resize:
    def __init__(self, size: Tuple[int, int], interpolation: str = "bilinear"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return F.resize(img, self.size, self.interpolation)

class CenterCrop:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return F.center_crop(img, self.size)

class ToGrayscale:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return F.to_grayscale(img)

class ToRGB:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return F.to_rgb(img)

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return F.normalize(img, self.mean, self.std)

class ToChannelFirst:
    """
    Convert (H, W, C) -> (C, H, W) for neural network layers.
    """
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] in (1, 3, 4):
            return np.transpose(img, (2, 0, 1))
        elif img.ndim == 2:
            return img[np.newaxis, ...]
        return img
