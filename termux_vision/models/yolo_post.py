from typing import List, Dict, Tuple
import numpy as np
from ..detect.nms import non_maximum_suppression

class YOLODecoder:
    """
    Lightweight Anchor-free & Anchor-based YOLO prediction post-processor.
    Decodes raw model output tensor [B, Num_Boxes, 4 + Num_Classes] or [Num_Boxes, 4 + Num_Classes]
    into validated bounding boxes with class labels and confidence scores.
    """
    def __init__(self, class_names: List[str] = None, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.class_names = class_names or ["object"]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def decode(self, raw_predictions: np.ndarray, orig_image_size: Tuple[int, int]) -> List[Dict]:
        """
        raw_predictions shape: (Num_Boxes, 4 + Num_Classes) in (cx, cy, w, h, class_scores...) format
        orig_image_size: (img_width, img_height)
        Returns: list of dict {"box": (x, y, w, h), "score": float, "class_id": int, "class_name": str}
        """
        if raw_predictions.ndim == 3:
            raw_predictions = raw_predictions[0]

        img_w, img_h = orig_image_size
        num_classes = len(self.class_names)

        boxes_cxcywh = raw_predictions[:, :4]
        class_scores = raw_predictions[:, 4:4 + num_classes]

        # Calculate max class scores and ids
        class_ids = np.argmax(class_scores, axis=1)
        max_scores = np.max(class_scores, axis=1)

        # Filter by confidence threshold
        mask = max_scores >= self.conf_threshold
        if not np.any(mask):
            return []

        filtered_boxes = boxes_cxcywh[mask]
        filtered_scores = max_scores[mask]
        filtered_class_ids = class_ids[mask]

        # Convert (cx, cy, w, h) normalized to (x, y, w, h) pixel coordinates
        boxes_xywh = []
        for b in filtered_boxes:
            cx, cy, bw, bh = b
            # If coordinates are normalized in [0, 1]
            if cx <= 1.0 and cy <= 1.0 and bw <= 1.0 and bh <= 1.0:
                cx *= img_w
                cy *= img_h
                bw *= img_w
                bh *= img_h
            
            x = max(0, int(cx - bw / 2.0))
            y = max(0, int(cy - bh / 2.0))
            w = int(bw)
            h = int(bh)
            boxes_xywh.append((x, y, w, h))

        # Perform NMS per class
        final_detections = []
        unique_classes = np.unique(filtered_class_ids)

        for cid in unique_classes:
            c_indices = np.where(filtered_class_ids == cid)[0]
            c_boxes = [boxes_xywh[idx] for idx in c_indices]
            c_scores = [float(filtered_scores[idx]) for idx in c_indices]

            keep = non_maximum_suppression(c_boxes, c_scores, iou_threshold=self.iou_threshold, score_threshold=self.conf_threshold)
            
            for k in keep:
                orig_idx = c_indices[k]
                final_detections.append({
                    "box": boxes_xywh[orig_idx],
                    "score": round(float(filtered_scores[orig_idx]), 4),
                    "class_id": int(cid),
                    "class_name": self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
                })

        # Sort all detections by score descending
        final_detections.sort(key=lambda d: d["score"], reverse=True)
        return final_detections
