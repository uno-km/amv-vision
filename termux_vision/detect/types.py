from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

@dataclass(frozen=True)
class BoundingBox:
    """
    Axis-aligned 2D bounding box using exclusive right/bottom coordinate space.
    Coordinates: [left, top, right, bottom)
    """
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self):
        if self.right < self.left:
            raise ValueError(f"Invalid BoundingBox: right ({self.right}) < left ({self.left})")
        if self.bottom < self.top:
            raise ValueError(f"Invalid BoundingBox: bottom ({self.bottom}) < top ({self.top})")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> 'BoundingBox':
        return cls(left=int(x), top=int(y), right=int(x + w), bottom=int(y + h))

@dataclass
class Detection:
    """
    Detected visual object or candidate region.
    """
    bbox: BoundingBox
    score: Optional[float] = None
    class_id: int = 0
    class_name: str = "object"
    metadata: Optional[Dict[str, Any]] = None
