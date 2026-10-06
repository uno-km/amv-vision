import numpy as np
import pytest
from termux_vision import models, bridge

def test_vit_patches():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    patches = models.extract_patches(img, patch_size=16)
    # (64/16)*(64/16) = 16 patches, patch_dim = 16*16*3 = 768
    assert patches.shape == (16, 768)

    reconstructed = models.reconstruct_from_patches(patches, (64, 64, 3), patch_size=16)
    assert reconstructed.shape == (64, 64, 3)
    assert np.allclose(img, reconstructed)

def test_depthwise_separable_conv():
    conv = models.DepthwiseSeparableConv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1)
    x = np.random.randn(3, 32, 32).astype(np.float32)
    out = conv(x)
    assert out.shape == (8, 32, 32)

def test_yolo_decoder():
    decoder = models.YOLODecoder(class_names=["person", "cat", "dog"], conf_threshold=0.5)
    # 2 dummy predictions: (cx, cy, w, h, score_p, score_c, score_d)
    preds = np.array([
        [0.5, 0.5, 0.2, 0.3, 0.85, 0.10, 0.05],  # Person (score 0.85)
        [0.1, 0.1, 0.05, 0.05, 0.05, 0.95, 0.05] # Cat (score 0.95)
    ], dtype=np.float32)
    detections = decoder.decode(preds, orig_image_size=(640, 480))
    assert len(detections) == 2
    # Highest score first
    assert detections[0]["class_name"] == "cat"
    assert detections[0]["score"] == 0.95
    assert detections[1]["class_name"] == "person"
    assert detections[1]["score"] == 0.85

def test_termux_train_bridge_dataset():
    patches = np.random.randn(100, 64).astype(np.float32)
    labels = np.random.randint(0, 2, (100, 1)).astype(np.float32)

    dataset = bridge.VisionPatchDataset(patches, labels, batch_size=16)
    assert len(dataset) == 7

    for batch_x, batch_y in dataset:
        assert batch_x.shape[0] <= 16
        assert batch_y.shape[0] <= 16
        break
