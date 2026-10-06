import os
import tempfile
import numpy as np
import pytest
from termux_vision import io, transforms

def test_image_io_and_info():
    # 1. Synthetic image creation
    arr = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        saved_path = io.save_image(arr, tmp_path)
        assert os.path.exists(saved_path)

        # 2. Get info
        info = io.get_image_info(tmp_path)
        assert info["width"] == 160
        assert info["height"] == 120
        assert info["format"] == "PNG"

        # 3. Load back
        loaded = io.load_image(tmp_path)
        assert loaded.shape == (120, 160, 3)
        assert np.allclose(loaded, arr)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_transforms_functional():
    arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Grayscale
    gray = transforms.to_grayscale(arr)
    assert gray.shape == (100, 100)

    # To RGB
    rgb = transforms.to_rgb(gray)
    assert rgb.shape == (100, 100, 3)

    # Resize
    resized = transforms.resize(arr, (50, 60))
    assert resized.shape == (60, 50, 3)

    # Crop
    cropped = transforms.crop(arr, 10, 10, 30, 40)
    assert cropped.shape == (40, 30, 3)

    # Center crop
    ccropped = transforms.center_crop(arr, (40, 40))
    assert ccropped.shape == (40, 40, 3)

    # Normalize
    norm = transforms.normalize(arr)
    assert norm.shape == (100, 100, 3)
    assert norm.dtype == np.float32

def test_transforms_compose():
    pipeline = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToGrayscale(),
        transforms.Normalize()
    ])
    arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    out = pipeline(arr)
    assert out.shape == (64, 64)
