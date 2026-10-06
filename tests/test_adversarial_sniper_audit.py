import os
import tempfile
import struct
import numpy as np
import pytest

import termux_vision as tv
from termux_vision import io, cv, detect, models, errors
from termux_vision.csrc.backend import _ensure_2d_uint8, c_compute_integral, c_sobel
from termux_vision.io.safetensors import load_safetensors, MAX_SAFETENSORS_HEADER_BYTES

def test_safetensors_malicious_huge_header_rejected():
    """Validates that SafeTensors with malicious oversized header is rejected without OOM."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        # Write header_len = 1 GB (malicious payload)
        tmp.write(struct.pack("<Q", 1024 * 1024 * 1024))
        tmp.write(b"{}")

    try:
        with pytest.raises(ValueError) as exc_info:
            load_safetensors(tmp_path)
        assert "exceeds" in str(exc_info.value)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_save_image_zero_size_empty_array_rejected():
    """Validates that save_image safely rejects empty 0-sized arrays without np.max crash."""
    empty_arr = np.empty((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError) as exc_info:
        io.save_image(empty_arr, "dummy.png")
    assert "empty" in str(exc_info.value).lower()

def test_backend_3d_and_float_array_auto_normalized():
    """Validates _ensure_2d_uint8 safely handles 3D RGB, float32 [0..1], and strided arrays."""
    # 3D RGB array
    rgb = np.random.randint(0, 256, (50, 60, 3), dtype=np.uint8)
    norm = _ensure_2d_uint8(rgb)
    assert norm.shape == (50, 60)
    assert norm.dtype == np.uint8

    # Float32 normalized array
    f_arr = (np.random.rand(40, 40)).astype(np.float32)
    norm_f = _ensure_2d_uint8(f_arr)
    assert norm_f.shape == (40, 40)
    assert norm_f.dtype == np.uint8

    # 4D array rejected
    arr_4d = np.zeros((2, 3, 10, 10))
    with pytest.raises(ValueError):
        _ensure_2d_uint8(arr_4d)

def test_conv_block_deterministic_weights_and_set_weights():
    """Validates DepthwiseSeparableConv2D uses deterministic initialization and supports explicit weight loading."""
    conv1 = models.conv_block.DepthwiseSeparableConv2D(in_channels=3, out_channels=16, kernel_size=3)
    conv2 = models.conv_block.DepthwiseSeparableConv2D(in_channels=3, out_channels=16, kernel_size=3)
    # Deterministic init produces identical weights
    assert np.allclose(conv1.dw_weights, conv2.dw_weights)
    assert np.allclose(conv1.pw_weights, conv2.pw_weights)

    # Explicit set_weights
    custom_dw = np.ones((3, 3, 3), dtype=np.float32)
    custom_pw = np.ones((16, 3), dtype=np.float32)
    conv1.set_weights(custom_dw, custom_pw)
    assert np.allclose(conv1.dw_weights, custom_dw)
    assert np.allclose(conv1.pw_weights, custom_pw)

def test_mobilenet_and_yolo_load_weights_pipeline():
    """Validates MobileNet and YOLO models support explicit weight loading and track is_trained status."""
    mobilenet = models.mobilenet.MobileNetV3FeatureExtractor(in_channels=3, feature_dim=64)
    assert mobilenet.is_trained is False
    x = np.random.randn(3, 32, 32).astype(np.float32)
    feat1 = mobilenet(x)
    assert feat1.shape == (64,)

    mobilenet.load_weights({"block1": {"dw": np.zeros((3,3,3)), "pw": np.zeros((32,3))}})
    assert mobilenet.is_trained is True

    yolo = models.yolo_detector.TinyYOLONanoDetector(num_classes=10)
    assert yolo.is_trained is False
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    dets = yolo.detect(img)
    assert isinstance(dets, list)

    yolo.load_weights({"conv1": {"dw": np.zeros((3,3,3)), "pw": np.zeros((16,3))}})
    assert yolo.is_trained is True
