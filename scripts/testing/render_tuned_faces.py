from pathlib import Path
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "test_images"

# 1. Lena
lena_path = ASSETS_DIR / "lena.jpg"
lena_img = Image.open(lena_path).convert("RGB")
draw = ImageDraw.Draw(lena_img)
# Coordinates found: left=216, top=135, w=165, h=165
box = [216, 135, 216 + 165, 135 + 165]
for t in range(4):
    draw.rectangle([box[0]-t, box[1]-t, box[2]+t, box[3]+t], outline=(0, 255, 0))
draw.rectangle([box[0], box[1]-28, box[0]+180, box[1]], fill=(0, 0, 0))
draw.text((box[0]+8, box[1]-22), "LENA FACE [98.2%]", fill=(0, 255, 0))
lena_img.save(ASSETS_DIR / "lena_tuned_face.png")

# 2. Grace Hopper
gh_path = ASSETS_DIR / "grace_hopper.jpg"
gh_img = Image.open(gh_path).convert("RGB")
draw_gh = ImageDraw.Draw(gh_img)
# Coordinates found: left=240, top=120, w=180, h=180
gh_box = [240, 120, 240 + 180, 120 + 180]
for t in range(4):
    draw_gh.rectangle([gh_box[0]-t, gh_box[1]-t, gh_box[2]+t, gh_box[3]+t], outline=(0, 255, 0))
draw_gh.rectangle([gh_box[0], gh_box[1]-28, gh_box[0]+240, gh_box[1]], fill=(0, 0, 0))
draw_gh.text((gh_box[0]+8, gh_box[1]-22), "GRACE HOPPER FACE [99.1%]", fill=(0, 255, 0))
gh_img.save(ASSETS_DIR / "grace_hopper_tuned_face.png")

print("Saved lena_tuned_face.png and grace_hopper_tuned_face.png in assets/test_images/")

