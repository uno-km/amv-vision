import warnings
from typing import List, Dict, Tuple
import numpy as np
from .conv_block import DepthwiseSeparableConv2D
from .yolo_post import YOLODecoder

COCO_80_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class TinyYOLONanoDetector:
    """
    Ultra-lightweight On-Device Object Detector (YOLO-Nano architecture) with Depthwise Separable Convolutions.
    """
    def __init__(self, num_classes: int = 80, class_names: List[str] = None, weights: dict = None):
        self.class_names = list(class_names) if class_names is not None else list(COCO_80_CLASSES[:num_classes])
        self.num_classes = len(self.class_names)
        # Lightweight Mobile Backbone
        self.conv1 = DepthwiseSeparableConv2D(3, 16, kernel_size=3, stride=2, weights=weights.get("conv1") if weights else None)
        self.conv2 = DepthwiseSeparableConv2D(16, 32, kernel_size=3, stride=2, weights=weights.get("conv2") if weights else None)
        self.conv3 = DepthwiseSeparableConv2D(32, 64, kernel_size=3, stride=2, weights=weights.get("conv3") if weights else None)
        
        # Detection Head: (cx, cy, w, h) + num_classes
        self.head = DepthwiseSeparableConv2D(64, 4 + self.num_classes, kernel_size=1, stride=1, weights=weights.get("head") if weights else None)
        self.decoder = YOLODecoder(class_names=self.class_names, conf_threshold=0.25, iou_threshold=0.45)

        if weights is not None:
            self._weights_loaded = True
        else:
            self._weights_loaded = False
            warnings.warn(
                "TinyYOLONanoDetector initialized with random weights (Kaiming Normal, Seed 42). "
                "Load trained weights via `.load_weights()` for real object detection.",
                UserWarning,
                stacklevel=2
            )

    @property
    def is_trained(self) -> bool:
        """Returns True if weights were explicitly loaded from a trained checkpoint."""
        return self._weights_loaded

    def load_weights(self, weights: dict):
        """Loads trained weights dictionary for backbone and detection head."""
        if "conv1" in weights: self.conv1 = DepthwiseSeparableConv2D(3, 16, 3, 2, weights=weights["conv1"])
        if "conv2" in weights: self.conv2 = DepthwiseSeparableConv2D(16, 32, 3, 2, weights=weights["conv2"])
        if "conv3" in weights: self.conv3 = DepthwiseSeparableConv2D(32, 64, 3, 2, weights=weights["conv3"])
        if "head" in weights: self.head = DepthwiseSeparableConv2D(64, 4 + len(self.class_names), 1, 1, weights=weights["head"])
        self._weights_loaded = True

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Input image: (H, W, 3) uint8 or float.
        Returns: list of detections [{"box": (x,y,w,h), "score": float, "class_name": str}]
        """
        orig_h, orig_w = image.shape[:2]
        
        # Resize to standard 256x256 input
        from ..transforms.functional import resize
        img_res = resize(image, (256, 256))
        
        # Channel-first float32
        x = np.transpose(img_res.astype(np.float32) / 255.0, (2, 0, 1))

        # Forward pass
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        out = self.head(h3) # (4 + num_classes, 32, 32)

        # Reshape to (1024, 4 + num_classes)
        c, grid_h, grid_w = out.shape
        out_trans = np.transpose(out, (1, 2, 0)).reshape(grid_h * grid_w, c)

        # Softmax on class scores
        class_logits = out_trans[:, 4:]
        exp_logits = np.exp(class_logits - np.max(class_logits, axis=-1, keepdims=True))
        out_trans[:, 4:] = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Decode bounding boxes
        detections = self.decoder.decode(out_trans, (orig_w, orig_h))
        return detections
