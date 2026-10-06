from typing import List, Tuple
import numpy as np

def find_contours(binary_image: np.ndarray, min_area: int = 10) -> List[dict]:
    """
    Find connected components and their bounding boxes and areas in a binary image.
    Returns list of dicts: {"box": (x, y, w, h), "area": int, "centroid": (cx, cy)}
    """
    img = (binary_image > 0).astype(np.int32)
    h, w = img.shape
    visited = np.zeros((h, w), dtype=bool)
    contours = []

    # 8-connectivity offsets
    dx = [-1, -1, -1,  0, 0,  1, 1, 1]
    dy = [-1,  0,  1, -1, 1, -1, 0, 1]

    for y in range(h):
        for x in range(w):
            if img[y, x] and not visited[y, x]:
                # BFS to trace component
                queue = [(y, x)]
                visited[y, x] = True
                pixels = []

                min_x, max_x = x, x
                min_y, max_y = y, y

                while queue:
                    cy, cx = queue.pop(0)
                    pixels.append((cy, cx))
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)

                    for k in range(8):
                        ny, nx = cy + dy[k], cx + dx[k]
                        if 0 <= ny < h and 0 <= nx < w:
                            if img[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                area = len(pixels)
                if area >= min_area:
                    bw = max_x - min_x + 1
                    bh = max_y - min_y + 1
                    mean_y = sum(p[0] for p in pixels) / area
                    mean_x = sum(p[1] for p in pixels) / area
                    contours.append({
                        "box": (min_x, min_y, bw, bh),
                        "area": area,
                        "centroid": (round(mean_x, 2), round(mean_y, 2)),
                        "pixel_count": area
                    })

    # Sort by area descending
    contours.sort(key=lambda c: c["area"], reverse=True)
    return contours

def color_histogram(image: np.ndarray, bins: int = 16, range_val: Tuple[float, float] = (0, 255)) -> np.ndarray:
    """
    Compute normalized 1D or multi-channel color histograms.
    Returns concatenated 1D normalized histogram array.
    """
    if image.ndim == 2:
        hist, _ = np.histogram(image, bins=bins, range=range_val)
        return hist.astype(np.float32) / max(1, hist.sum())
    
    channels = image.shape[2]
    hists = []
    for c in range(channels):
        h, _ = np.histogram(image[:, :, c], bins=bins, range=range_val)
        h_norm = h.astype(np.float32) / max(1, h.sum())
        hists.append(h_norm)

    return np.concatenate(hists)
