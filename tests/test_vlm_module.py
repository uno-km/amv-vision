import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import termux_vision as tv
from termux_vision import errors, detect, models, vlm, io

def test_bounding_box_exclusive_geometry():
    bbox = detect.BoundingBox(left=10, top=20, right=50, bottom=80)
    assert bbox.width == 40
    assert bbox.height == 60
    assert bbox.area == 2400
    assert bbox.to_xywh() == (10, 20, 40, 60)

    b2 = detect.BoundingBox.from_xywh(10, 20, 40, 60)
    assert b2 == bbox

def test_zero_copy_crop():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape((100, 100, 3))
    bbox = detect.BoundingBox(10, 20, 30, 40)
    cropped = tv.cv.crop(img, bbox, copy=False)
    assert cropped.shape == (20, 20, 3)

def test_image_limits_decompression_bomb():
    limits = io.limits.ImageLimits(max_pixels=1000, max_dimension=100)
    with pytest.raises(errors.DecompressionBombError):
        limits.validate_dimensions(500, 500)

def test_typed_embedding_similarity():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    e1 = models.Embedding(v1, model_id="mobilenetv3", dimension=3)
    e2 = models.Embedding(v2, model_id="mobilenetv3", dimension=3)
    e3 = models.Embedding(v3, model_id="mobilenetv3", dimension=3)

    assert np.isclose(models.compute_similarity(e1, e2), 1.0)
    assert np.isclose(models.compute_similarity(e1, e3), 0.0)

    e_other = models.Embedding(v1, model_id="resnet50", dimension=3)
    with pytest.raises(errors.IncompatibleEmbeddingError):
        models.compute_similarity(e1, e_other)

def test_vlm_manifest_and_admission_control():
    assert "smolvlm-500m-q4" in vlm.CATALOG
    assert "qwen2-vl-2b-q4" in vlm.CATALOG

    # Memory admission raises InsufficientMemoryError when strict and budget is exceeded
    est = vlm.memory.MemoryEstimate(400, 200, 100, 30, 20, 750, "estimated")
    with pytest.raises(errors.InsufficientMemoryError):
        vlm.memory.check_memory_admission(est, user_budget_mb=200, memory_policy="strict")
