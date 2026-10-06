import numpy as np
import pytest
from termux_vision import detect

def test_nms_and_iou():
    box1 = (10, 10, 50, 50)
    box2 = (10, 10, 50, 50)
    assert detect.box_iou(box1, box2) == 1.0

    box3 = (100, 100, 50, 50)
    assert detect.box_iou(box1, box3) == 0.0

    boxes = [(10, 10, 50, 50), (12, 12, 50, 50), (100, 100, 50, 50)]
    scores = [0.9, 0.85, 0.7]
    keep = detect.non_maximum_suppression(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 2
    assert 0 in keep
    assert 2 in keep

def test_haar_detector():
    detector = detect.HaarCascadeDetector()
    assert len(detector.stages) >= 3

    # Synthetic image with face-like contrast pattern
    img = np.ones((100, 100), dtype=np.uint8) * 128
    # Dark eyes
    img[20:30, 20:80] = 30
    # Bright cheeks
    img[35:55, 20:80] = 220
    # Mouth line
    img[70:78, 30:70] = 40

    boxes = detector.detect_multiscale(img, scale_factor=1.2, min_size=(24, 24))
    assert isinstance(boxes, list)
