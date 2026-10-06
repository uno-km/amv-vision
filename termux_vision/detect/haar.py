from typing import List, Tuple, Optional
import numpy as np
from ..cv.integral import compute_integral_image, box_sum
from ..transforms.functional import to_grayscale
from .nms import non_maximum_suppression
from .types import BoundingBox, Detection

class HaarFeature:
    """Heuristic intensity rectangle comparison feature."""
    def __init__(self, rects: List[Tuple[int, int, int, int, float]], threshold: float, left_val: float, right_val: float):
        self.rects = rects
        self.threshold = threshold
        self.left_val = left_val
        self.right_val = right_val

    def evaluate(self, integral: np.ndarray, x: int, y: int, scale: float) -> float:
        val = 0.0
        for rx, ry, rw, rh, weight in self.rects:
            sx = int(x + rx * scale)
            sy = int(y + ry * scale)
            sw = max(1, int(rw * scale))
            sh = max(1, int(rh * scale))
            s = box_sum(integral, sx, sy, sx + sw, sy + sh)
            val += s * weight

        return self.left_val if val < self.threshold * (scale**2) else self.right_val

class CascadeStage:
    def __init__(self, stage_threshold: float, features: List[HaarFeature]):
        self.stage_threshold = stage_threshold
        self.features = features

    def evaluate(self, integral: np.ndarray, x: int, y: int, scale: float) -> bool:
        stage_sum = 0.0
        for feat in self.features:
            stage_sum += feat.evaluate(integral, x, y, scale)
        return stage_sum >= self.stage_threshold

class HaarCascadeDetector:
    """
    Heuristic 3-Stage Face-Like Region Scanner.
    Scans integral image using hand-tuned multi-rectangle brightness contrast features.
    """
    def __init__(self, base_window_size: Tuple[int, int] = (24, 24), stages: List[CascadeStage] = None):
        self.base_w, self.base_h = base_window_size
        self.stages = stages if stages is not None else self._build_default_face_stages()

    def _build_default_face_stages(self) -> List[CascadeStage]:
        stages = []
        f1 = HaarFeature([(2, 6, 20, 6, -1.0), (2, 12, 20, 6, 1.0)], threshold=-10.0, left_val=-1.0, right_val=1.2)
        stages.append(CascadeStage(stage_threshold=0.2, features=[f1]))

        f2 = HaarFeature([(6, 6, 4, 12, -1.0), (10, 6, 4, 12, 2.0), (14, 6, 4, 12, -1.0)], threshold=-5.0, left_val=-0.8, right_val=1.5)
        stages.append(CascadeStage(stage_threshold=0.3, features=[f2]))

        f3 = HaarFeature([(4, 16, 16, 4, -1.0), (4, 12, 16, 4, 1.0)], threshold=-2.0, left_val=-0.5, right_val=1.1)
        stages.append(CascadeStage(stage_threshold=0.2, features=[f3]))
        return stages

    def detect_multiscale(
        self,
        image: np.ndarray,
        scale_factor: float = 1.2,
        min_neighbors: int = 2,
        min_size: Tuple[int, int] = (24, 24),
        max_size: Tuple[int, int] = None,
        max_results: int = 20
    ) -> List[BoundingBox]:
        gray = to_grayscale(image)
        img_h, img_w = gray.shape
        integral = compute_integral_image(gray)

        if max_size is None:
            max_size = (img_w, img_h)

        # 1. C-Native Path if available
        try:
            from ..csrc.backend import has_c_backend, c_haar_detect
            if has_c_backend():
                c_boxes, c_scores = c_haar_detect(
                    integral, img_w, img_h,
                    scale_factor=scale_factor,
                    min_size=min_size[0],
                    max_size=min(max_size[0], max_size[1]),
                    max_boxes=1000
                )
                if c_boxes:
                    keep = non_maximum_suppression(c_boxes, c_scores, iou_threshold=0.3, score_threshold=0.0)
                    bboxes = [BoundingBox.from_xywh(c_boxes[i][0], c_boxes[i][1], c_boxes[i][2], c_boxes[i][3]) for i in keep]
                    return bboxes[:max_results]
                return []
        except Exception as e:
            import logging
            logging.getLogger("termux_vision.detect.haar").debug(
                "[termux-vision] C haar_detect failed, falling back to NumPy: %s", e
            )

        # 2. Pure NumPy Path
        candidates = []
        scores = []
        scale = min_size[0] / 24.0

        while True:
            win_w = int(self.base_w * scale)
            win_h = int(self.base_h * scale)

            if win_w > img_w or win_h > img_h or win_w > max_size[0] or win_h > max_size[1]:
                break

            if win_w >= min_size[0] and win_h >= min_size[1]:
                step = max(2, int(4 * scale))
                for y in range(0, img_h - win_h, step):
                    for x in range(0, img_w - win_w, step):
                        passed = True
                        for stage in self.stages:
                            if not stage.evaluate(integral, x, y, scale):
                                passed = False
                                break
                        if passed:
                            candidates.append((x, y, win_w, win_h))
                            mean_val = box_sum(integral, x, y, x + win_w, y + win_h) / (win_w * win_h)
                            scores.append(float(mean_val))

            scale *= scale_factor

        if not candidates:
            return []

        keep_indices = non_maximum_suppression(candidates, scores, iou_threshold=0.3, score_threshold=0.0)
        bboxes = [BoundingBox.from_xywh(candidates[i][0], candidates[i][1], candidates[i][2], candidates[i][3]) for i in keep_indices]
        return bboxes[:max_results]

def detect_faces(
    image: np.ndarray,
    scale_factor: float = 1.2,
    min_size: Tuple[int, int] = (32, 32),
    max_results: int = 10,
    score_threshold: float = 0.70,
    backend: str = "auto"
) -> List[Detection]:
    """
    Detects faces in the given image.
    Prioritizes the production UltraFace SSD Deep Learning engine via ONNX Runtime.
    Falls back to legacy Haar cascade only if ONNX runtime is unavailable.
    """
    if backend in ("auto", "neural", "onnx"):
        try:
            from .neural import NeuralFaceDetector
            detector = NeuralFaceDetector()
            detections = detector.detect(
                image,
                score_threshold=score_threshold,
                max_results=max_results
            )
            if detections or backend == "neural":
                return detections
        except Exception as exc:
            import logging
            logging.getLogger("termux_vision.detect").debug(
                "[termux-vision] NeuralFaceDetector failed or ONNX unavailable: %s. Using Haar fallback.", exc
            )

    # Legacy Haar Cascade Fallback
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    if max_dim > 640:
        ratio = 640.0 / max_dim
        new_w = max(32, int(w * ratio))
        new_h = max(32, int(h * ratio))
        from ..transforms.functional import resize
        scaled_img = resize(image, (new_w, new_h))
        scaled_min_size = (max(16, int(min_size[0] * ratio)), max(16, int(min_size[1] * ratio)))
        
        detector = HaarCascadeDetector()
        bboxes = detector.detect_multiscale(scaled_img, scale_factor=scale_factor, min_size=scaled_min_size, max_results=max_results)
        
        inv_ratio = 1.0 / ratio
        restored = []
        for b in bboxes:
            orig_bbox = BoundingBox(
                left=int(round(b.left * inv_ratio)),
                top=int(round(b.top * inv_ratio)),
                right=min(w, int(round(b.right * inv_ratio))),
                bottom=min(h, int(round(b.bottom * inv_ratio)))
            )
            restored.append(Detection(bbox=orig_bbox, score=None, class_name="face_candidate"))
        return restored

    detector = HaarCascadeDetector()
    bboxes = detector.detect_multiscale(image, scale_factor=scale_factor, min_size=min_size, max_results=max_results)
    return [Detection(bbox=b, score=None, class_name="face_candidate") for b in bboxes]
