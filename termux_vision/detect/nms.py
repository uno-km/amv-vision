from typing import List, Tuple
import numpy as np

def box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union (IoU) of two boxes in (x, y, w, h) format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)

def non_maximum_suppression(
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    iou_threshold: float = 0.45,
    score_threshold: float = 0.25
) -> List[int]:
    """
    Perform Non-Maximum Suppression on bounding boxes with zero divide-by-zero warnings.
    """
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    valid_mask = scores_arr >= score_threshold
    if not np.any(valid_mask):
        return []

    indices = np.where(valid_mask)[0]
    boxes_filtered = boxes_arr[indices]
    scores_filtered = scores_arr[indices]

    order = scores_filtered.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(indices[i])

        if order.size == 1:
            break

        current_box = boxes_filtered[i]
        other_boxes = boxes_filtered[order[1:]]

        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[0] + current_box[2], other_boxes[:, 0] + other_boxes[:, 2])
        y2 = np.minimum(current_box[1] + current_box[3], other_boxes[:, 1] + other_boxes[:, 3])

        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
        inter = w * h

        area_curr = current_box[2] * current_box[3]
        area_others = other_boxes[:, 2] * other_boxes[:, 3]
        union = area_curr + area_others - inter

        with np.errstate(divide='ignore', invalid='ignore'):
            ious = np.where(union > 0, inter / np.maximum(union, 1e-7), 0.0)
            
        remaining = np.where(ious <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep
