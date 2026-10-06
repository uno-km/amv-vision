import time
import json
import numpy as np
from PIL import Image, ImageDraw
import termux_vision as tv
from termux_vision.detect import detect_faces
from termux_vision.cv.integral import compute_integral_image, box_sum
from termux_vision.transforms.functional import to_grayscale

def run_face_test(image_path, output_annotated_path):
    raw_img = Image.open(image_path).convert('RGB')
    np_img = np.array(raw_img)
    gray = to_grayscale(np_img)
    integral = compute_integral_image(gray)
    h, w = gray.shape

    t0 = time.perf_counter()
    # Execute detect_faces
    detections = detect_faces(np_img, scale_factor=1.15, min_size=(30, 30), max_results=10)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    annotated = raw_img.copy()
    draw = ImageDraw.Draw(annotated)

    results = []
    for idx, det in enumerate(detections):
        box = det.bbox
        bx, by, bw, bh = box.left, box.top, box.width, box.height
        
        # Calculate regional brightness / contrast score
        mean_val = float(box_sum(integral, max(0, bx), max(0, by), min(w, bx + bw), min(h, by + bh)) / max(1, bw * bh))
        score = round(mean_val / 255.0 * 100.0, 1)

        # Draw box (Neon Cyan / Lime)
        color = (0, 255, 128) if idx == 0 else (0, 200, 255)
        for t in range(3): # border thickness 3
            draw.rectangle([bx - t, by - t, bx + bw + t, by + bh + t], outline=color)

        # Draw label badge
        label = f"Face #{idx+1} ({score}%)"
        draw.rectangle([bx, max(0, by - 22), bx + 120, by], fill=(0, 0, 0))
        draw.text((bx + 4, max(0, by - 20)), label, fill=color)

        results.append({
            "index": idx + 1,
            "box": {"x": bx, "y": by, "w": bw, "h": bh},
            "contrast_score": score,
            "area_px": bw * bh
        })

    # Overlay metadata banner on top-left
    banner = f"termux-vision v1.5.0 Face Detection | Latency: {latency_ms:.2f}ms | Count: {len(detections)}"
    draw.rectangle([0, 0, min(w, 520), 26], fill=(0, 0, 0))
    draw.text((8, 6), banner, fill=(255, 255, 255))

    annotated.save(output_annotated_path)
    return {
        "file": image_path,
        "resolution": f"{w}x{h}",
        "detected_count": len(detections),
        "latency_ms": round(latency_ms, 2),
        "faces": results,
        "output_saved": output_annotated_path
    }

if __name__ == "__main__":
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    ASSETS_DIR = REPO_ROOT / "assets" / "test_images"

    print("=== STARTING FACE DETECTION SUITE (Galaxy A53) ===")
    res_lena = run_face_test(str(ASSETS_DIR / "lena.jpg"), str(ASSETS_DIR / "lena_annotated.png"))
    res_grace = run_face_test(str(ASSETS_DIR / "grace_hopper.jpg"), str(ASSETS_DIR / "grace_hopper_annotated.png"))
    res_person = run_face_test(str(ASSETS_DIR / "person.jpg"), str(ASSETS_DIR / "person_annotated.png"))

    report = [res_lena, res_grace, res_person]
    print(json.dumps(report, indent=2))
    report_path = ASSETS_DIR / "face_detection_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

