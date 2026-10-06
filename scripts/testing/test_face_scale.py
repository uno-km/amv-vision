from pathlib import Path
import numpy as np
from PIL import Image
from termux_vision.detect.haar import HaarCascadeDetector

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "test_images"

img = np.array(Image.open(ASSETS_DIR / "lena.jpg").convert("RGB"))
det = HaarCascadeDetector()

for min_sz in [50, 80, 100, 120, 150]:
    boxes = det.detect_multiscale(img, scale_factor=1.1, min_size=(min_sz, min_sz), max_size=(300, 300))
    print(f"min_size={min_sz}: found {len(boxes)} boxes")
    for b in boxes:
        print(f"  -> left={b.left}, top={b.top}, w={b.width}, h={b.height}")

# Check Grace Hopper
print("\nGrace Hopper:")
gh_img = np.array(Image.open(ASSETS_DIR / "grace_hopper.jpg").convert("RGB"))
for min_sz in [60, 100, 140, 180]:
    boxes = det.detect_multiscale(gh_img, scale_factor=1.1, min_size=(min_sz, min_sz), max_size=(350, 350))
    print(f"min_size={min_sz}: found {len(boxes)} boxes")
    for b in boxes:
        print(f"  -> left={b.left}, top={b.top}, w={b.width}, h={b.height}")

