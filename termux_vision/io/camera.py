import subprocess
import os
import tempfile
import numpy as np
from typing import Optional
from .loader import load_image
from ..errors import CameraPermissionError, TermuxAPIUnavailableError

class CameraCapture:
    """
    Termux Camera API wrapper using termux-camera-photo.
    """
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id

    def capture_frame(self, target_size: Optional[tuple] = None) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = ["termux-camera-photo", "-c", str(self.camera_id), tmp_path]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode != 0:
                err = res.stderr.lower()
                if "permission" in err:
                    raise CameraPermissionError(f"Camera permission denied on Android: {res.stderr}")
                raise TermuxAPIUnavailableError(f"termux-camera-photo failed (returncode {res.returncode}): {res.stderr}")

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                raise TermuxAPIUnavailableError("termux-camera-photo produced an empty capture file.")

            # Correct signature: load_image does not accept mode parameter
            arr = load_image(tmp_path, target_size=target_size)
            return arr
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
