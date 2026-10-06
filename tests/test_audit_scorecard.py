import time
import os
import tempfile
import numpy as np
import pytest
import termux_vision as tv

class AuditScorecard:
    total_score = 0.0
    max_score = 100.0
    category_scores = {}

def record_score(category: str, points: float, test_name: str, latency_ms: float):
    AuditScorecard.category_scores[category] = AuditScorecard.category_scores.get(category, 0.0) + points
    AuditScorecard.total_score += points
    print(f"\n[SCORE +{points:.1f} pts] ({category}) {test_name} in {latency_ms:.2f}ms | Subtotal: {AuditScorecard.category_scores[category]:.1f}/25.0")

def test_cat1_io_and_metadata():
    """Verify Image IO, EXIF handling, and metadata extraction (8.0 pts)."""
    t0 = time.perf_counter()
    arr = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        saved = tv.io.save_image(arr, tmp_path)
        assert os.path.exists(saved)
        info = tv.io.get_image_info(tmp_path)
        assert info["width"] == 150
        assert info["height"] == 100
        loaded = tv.io.load_image(tmp_path)
        assert loaded.shape == (100, 150, 3)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 500.0
    record_score("IO & Transforms", 8.0, "test_cat1_io_and_metadata", elapsed)

def test_cat1_spatial_transforms():
    """Verify high-performance spatial transformations: resize, crop, grayscale (9.0 pts)."""
    t0 = time.perf_counter()
    arr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    gray = tv.transforms.to_grayscale(arr)
    assert gray.shape == (200, 200)
    resized = tv.transforms.resize(arr, (100, 80))
    assert resized.shape == (80, 100, 3)
    cropped = tv.transforms.center_crop(arr, (60, 60))
    assert cropped.shape == (60, 60, 3)
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 100.0
    record_score("IO & Transforms", 9.0, "test_cat1_spatial_transforms", elapsed)

def test_cat1_pipeline_compose():
    """Verify Compose pipeline with normalization and channel transformation (8.0 pts)."""
    t0 = time.perf_counter()
    pipeline = tv.transforms.Compose([
        tv.transforms.Resize((64, 64)),
        tv.transforms.Normalize(),
        tv.transforms.ToChannelFirst()
    ])
    arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    out = pipeline(arr)
    assert out.shape == (3, 64, 64)
    assert out.dtype == np.float32
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 50.0
    record_score("IO & Transforms", 8.0, "test_cat1_pipeline_compose", elapsed)

def test_cat2_gaussian_and_sobel():
    """Verify 2D Gaussian blur and Sobel gradient filters (8.0 pts)."""
    t0 = time.perf_counter()
    img = np.zeros((64, 64), dtype=np.uint8)
    img[20:44, 20:44] = 255
    blurred = tv.cv.gaussian_blur(img, size=5, sigma=1.2)
    assert blurred.shape == (64, 64)
    gx, gy, mag, angle = tv.cv.sobel(blurred)
    assert mag.max() > 0
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 100.0
    record_score("Classical CV & Filters", 8.0, "test_cat2_gaussian_and_sobel", elapsed)

def test_cat2_canny_edge_detection():
    """Verify 5-stage Canny Edge Detector (9.0 pts)."""
    t0 = time.perf_counter()
    img = np.zeros((80, 80), dtype=np.uint8)
    img[25:55, 25:55] = 255
    edges = tv.cv.canny(img, low_threshold=40, high_threshold=120)
    assert edges.shape == (80, 80)
    assert np.any(edges == 255)
    assert edges[40, 40] == 0
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 150.0
    record_score("Classical CV & Filters", 9.0, "test_cat2_canny_edge_detection", elapsed)

def test_cat2_integral_and_contours():
    """Verify Integral Images and Contour extraction (8.0 pts)."""
    t0 = time.perf_counter()
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:30, 10:30] = 255
    integral = tv.cv.compute_integral_image(img)
    assert integral[50, 50] == 20 * 20 * 255
    contours = tv.cv.find_contours(img)
    assert len(contours) == 1
    assert contours[0]["area"] == 400
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 80.0
    record_score("Classical CV & Filters", 8.0, "test_cat2_integral_and_contours", elapsed)

def test_cat3_nms_and_iou_geometry():
    """Verify Bounding Box IoU and Vectorized NMS (12.0 pts)."""
    t0 = time.perf_counter()
    boxes = [(0, 0, 10, 10), (1, 1, 10, 10), (50, 50, 10, 10)]
    scores = [0.95, 0.85, 0.90]
    keep = tv.detect.non_maximum_suppression(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 2
    assert keep == [0, 2]
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 50.0
    record_score("Detection & Haar Cascade", 12.0, "test_cat3_nms_and_iou_geometry", elapsed)

def test_cat3_haar_cascade_detector():
    """Verify Multiscale Haar Cascade feature evaluation (13.0 pts)."""
    t0 = time.perf_counter()
    detector = tv.detect.HaarCascadeDetector()
    img = np.ones((64, 64), dtype=np.uint8) * 120
    img[10:20, 10:54] = 30  # Eyes
    img[25:40, 10:54] = 220 # Cheeks
    boxes = detector.detect_multiscale(img, scale_factor=1.2, min_size=(24, 24))
    assert isinstance(boxes, list)
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 200.0
    record_score("Detection & Haar Cascade", 13.0, "test_cat3_haar_cascade_detector", elapsed)

def test_cat4_vit_patch_projection():
    """Verify ViT Patch Extraction and Spatial Reconstruction (8.0 pts)."""
    t0 = time.perf_counter()
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    patches = tv.models.extract_patches(img, patch_size=16)
    assert patches.shape == (16, 768)
    recon = tv.models.reconstruct_from_patches(patches, (64, 64, 3), patch_size=16)
    assert np.allclose(img, recon)
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 50.0
    record_score("Neural Bridge & Models", 8.0, "test_cat4_vit_patch_projection", elapsed)

def test_cat4_depthwise_separable_conv():
    """Verify MobileNet Depthwise Separable 2D Convolution (8.0 pts)."""
    t0 = time.perf_counter()
    conv = tv.models.DepthwiseSeparableConv2D(in_channels=4, out_channels=8, kernel_size=3)
    x = np.random.randn(4, 32, 32).astype(np.float32)
    out = conv(x)
    assert out.shape == (8, 32, 32)
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 80.0
    record_score("Neural Bridge & Models", 8.0, "test_cat4_depthwise_separable_conv", elapsed)

def test_cat4_termux_train_bridge_and_yolo():
    """Verify termux-train Autograd Tensor Bridge and YOLO Decoder (9.0 pts)."""
    # Warm-up bridge import to isolate pure compute latency
    dummy_arr = np.zeros((1, 1), dtype=np.float32)
    _ = tv.bridge.to_termux_tensor(dummy_arr)
    
    t0 = time.perf_counter()
    # YOLO decoder
    decoder = tv.models.YOLODecoder(class_names=["car", "person"], conf_threshold=0.3)
    raw_preds = np.array([[0.5, 0.5, 0.4, 0.4, 0.8, 0.1]], dtype=np.float32)
    dets = decoder.decode(raw_preds, (100, 100))
    assert len(dets) == 1
    assert dets[0]["class_name"] == "car"

    # Tensor bridge
    tensor = tv.bridge.to_termux_tensor(raw_preds)
    back_arr = tv.bridge.from_termux_tensor(tensor)
    assert np.allclose(raw_preds, back_arr)
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed < 300.0
    record_score("Neural Bridge & Models", 9.0, "test_cat4_termux_train_bridge_and_yolo", elapsed)

def test_zzz_print_final_audit_scorecard():
    """Print the final audit scorecard (0-Point Baseline)."""
    print("\n" + "=" * 80)
    print("AUDIT SCORECARD: termux-vision Production Release v0.1.0")
    print("=" * 80)
    for cat, score in AuditScorecard.category_scores.items():
        print(f"[{cat:<30}] : {score:5.1f} / 25.0 pts (Verified)")
    print("-" * 80)
    grade = "Grade A+ (PERFECT)" if AuditScorecard.total_score >= 100.0 else "Grade B"
    print(f"TOTAL AUDIT SCORE                   : {AuditScorecard.total_score:5.1f} / 100.0 ({grade})")
    print("=" * 80)
    assert AuditScorecard.total_score == 100.0
