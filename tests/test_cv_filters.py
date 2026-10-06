import numpy as np
import pytest
from termux_vision import cv

def test_gaussian_blur_and_sobel():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[20:30, 20:30] = 255  # White square in black field

    blurred = cv.gaussian_blur(img, size=5, sigma=1.0)
    assert blurred.shape == (50, 50)
    assert blurred.max() > 0

    gx, gy, mag, angle = cv.sobel(blurred)
    assert gx.shape == (50, 50)
    assert gy.shape == (50, 50)
    assert mag.shape == (50, 50)
    assert angle.shape == (50, 50)
    assert mag.max() > 0

def test_canny_edge_detector():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 255  # Box

    edges = cv.canny(img, low_threshold=30, high_threshold=100)
    assert edges.shape == (64, 64)
    assert np.any(edges == 255)
    # The center of the box should be black (0), only boundary is white (255)
    assert edges[32, 32] == 0
    assert edges[16, 32] == 255 or edges[17, 32] == 255

def test_integral_image_and_box_filter():
    img = np.ones((10, 10), dtype=np.uint8) * 10
    integral = cv.compute_integral_image(img)
    assert integral.shape == (11, 11)
    assert integral[10, 10] == 1000  # 10 * 10 * 10

    # Box sum of 4x4 area = 16 * 10 = 160
    s = cv.box_sum(integral, 2, 2, 6, 6)
    assert s == 160.0

    filtered = cv.box_filter(img, radius=1)
    assert filtered.shape == (10, 10)

def test_morphology_and_contours():
    img = np.zeros((40, 40), dtype=np.uint8)
    img[10:30, 10:30] = 255

    dilated = cv.dilate(img, iterations=1)
    assert np.sum(dilated > 0) > np.sum(img > 0)

    eroded = cv.erode(img, iterations=1)
    assert np.sum(eroded > 0) < np.sum(img > 0)

    contours = cv.find_contours(img, min_area=5)
    assert len(contours) == 1
    assert contours[0]["box"] == (10, 10, 20, 20)
    assert contours[0]["area"] == 400

    hist = cv.color_histogram(img, bins=8)
    assert len(hist) == 8
    assert abs(hist.sum() - 1.0) < 1e-4
