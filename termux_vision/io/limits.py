import os
from dataclasses import dataclass
from ..errors import DecompressionBombError

@dataclass(frozen=True)
class ImageLimits:
    max_pixels: int = 16_000_000 # 16 Megapixels (e.g. 4000x4000)
    max_dimension: int = 8192
    max_file_size_mb: float = 32.0
    allow_animated: bool = False

    def validate_file_size(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file does not exist: {file_path}")
        
        size_mb = os.path.getsize(file_path) / (1024.0 * 1024.0)
        if size_mb > self.max_file_size_mb:
            raise DecompressionBombError(
                f"File size ({size_mb:.2f} MB) exceeds maximum allowed size ({self.max_file_size_mb:.2f} MB)."
            )

    def validate_dimensions(self, width: int, height: int):
        pixels = width * height
        if pixels > self.max_pixels:
            raise DecompressionBombError(
                f"Total image pixels ({pixels:,}) exceeds safe maximum limit ({self.max_pixels:,})."
            )
        if width > self.max_dimension or height > self.max_dimension:
            raise DecompressionBombError(
                f"Image dimension ({width}x{height}) exceeds maximum dimension limit ({self.max_dimension}px)."
            )

DEFAULT_IMAGE_LIMITS = ImageLimits()
