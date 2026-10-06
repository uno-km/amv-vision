import os
import tempfile
import numpy as np
import pytest
from termux_vision import io, models

def test_safetensors_io():
    weights = {
        "conv.weight": np.random.randn(16, 3, 3, 3).astype(np.float32),
        "conv.bias": np.random.randn(16).astype(np.float32),
        "int_param": np.array([1, 2, 3, 4], dtype=np.int32)
    }

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        io.save_safetensors(weights, tmp_path)
        assert os.path.exists(tmp_path)

        loaded = io.load_safetensors(tmp_path)
        assert "conv.weight" in loaded
        assert "conv.bias" in loaded
        assert "int_param" in loaded
        assert np.allclose(weights["conv.weight"], loaded["conv.weight"])
        assert np.allclose(weights["conv.bias"], loaded["conv.bias"])
        assert np.array_equal(weights["int_param"], loaded["int_param"])
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_mobilenet_extractor():
    extractor = models.MobileNetV3FeatureExtractor(in_channels=3, feature_dim=256)
    img = np.random.randn(3, 128, 128).astype(np.float32)
    embedding = extractor(img)
    assert embedding.shape == (256,)

def test_tiny_yolo_nano_detector():
    detector = models.TinyYOLONanoDetector(class_names=["person", "car"])
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    detections = detector.detect(img)
    assert isinstance(detections, list)
