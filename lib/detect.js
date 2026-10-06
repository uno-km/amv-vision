/**
 * BoundingBox Geometry & Non-Maximum Suppression (NMS) for Node.js.
 * Open-Source under Apache License 2.0.
 */

'use strict';

function computeIoU(boxA, boxB) {
  const x1 = Math.max(boxA.x, boxB.x);
  const y1 = Math.max(boxA.y, boxB.y);
  const x2 = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
  const y2 = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);

  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = boxA.width * boxA.height;
  const areaB = boxB.width * boxB.height;
  const unionArea = areaA + areaB - interArea;

  return unionArea <= 0 ? 0 : interArea / unionArea;
}

function nms(boxes, iouThreshold = 0.45) {
  if (!boxes || boxes.length === 0) return [];
  const sorted = [...boxes].sort((a, b) => (b.score || 0) - (a.score || 0));
  const keep = [];

  for (let i = 0; i < sorted.length; i++) {
    const current = sorted[i];
    let suppressed = false;
    for (let j = 0; j < keep.length; j++) {
      if (computeIoU(current, keep[j]) > iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      keep.push(current);
    }
  }

  return keep;
}

function computeIntegralImage(grayPixels, width, height) {
  const stride = width + 1;
  const integral = new Float64Array((width + 1) * (height + 1));

  for (let y = 0; y < height; y++) {
    let rowSum = 0.0;
    const dstRow = (y + 1) * stride;
    const srcRow = y * width;

    for (let x = 0; x < width; x++) {
      rowSum += grayPixels[srcRow + x];
      integral[dstRow + x + 1] = integral[dstRow - stride + x + 1] + rowSum;
    }
  }
  return integral;
}

function boxSum(integral, stride, x1, y1, x2, y2) {
  return (
    integral[y2 * stride + x2] -
    integral[y1 * stride + x2] -
    integral[y2 * stride + x1] +
    integral[y1 * stride + x1]
  );
}

function detectFaces(grayPixels, width, height, options = {}) {
  const scaleFactor = options.scaleFactor || 1.2;
  const minSize = options.minSize || 24;
  const maxSize = options.maxSize || Math.min(width, height);
  const integral = computeIntegralImage(grayPixels, width, height);
  const stride = width + 1;

  const candidates = [];
  let scale = minSize / 24.0;

  while (true) {
    const winW = Math.floor(24 * scale);
    const winH = Math.floor(24 * scale);
    if (winW > width || winH > height || winW > maxSize || winH > maxSize) break;

    const step = Math.max(2, Math.floor(4 * scale));
    const s2 = scale * scale;

    for (let y = 0; y <= height - winH; y += step) {
      for (let x = 0; x <= width - winW; x += step) {
        // Stage 1: Eye darkness vs cheek
        const eX = x + Math.floor(2 * scale), eY = y + Math.floor(6 * scale), eW = Math.floor(20 * scale), eH = Math.floor(6 * scale);
        const eyes = boxSum(integral, stride, eX, eY, eX + eW, eY + eH);
        const cX = x + Math.floor(2 * scale), cY = y + Math.floor(12 * scale), cW = Math.floor(20 * scale), cH = Math.floor(6 * scale);
        const cheeks = boxSum(integral, stride, cX, cY, cX + cW, cY + cH);
        const val1 = -1.0 * eyes + 1.0 * cheeks;
        if (val1 < -10.0 * s2 ? -1.0 : 1.2 < 0.2) continue;

        // Stage 2: Nose bridge
        const nX = x + Math.floor(10 * scale), nY = y + Math.floor(6 * scale), nW = Math.floor(4 * scale), nH = Math.floor(12 * scale);
        const nose = boxSum(integral, stride, nX, nY, nX + nW, nY + nH);
        const lX = x + Math.floor(6 * scale), rX = x + Math.floor(14 * scale);
        const lCheek = boxSum(integral, stride, lX, nY, lX + nW, nY + nH);
        const rCheek = boxSum(integral, stride, rX, nY, rX + nW, nY + nH);
        const val2 = -1.0 * lCheek + 2.0 * nose - 1.0 * rCheek;
        if (val2 < -5.0 * s2 ? -0.8 : 1.5 < 0.3) continue;

        const area = winW * winH;
        const meanScore = boxSum(integral, stride, x, y, x + winW, y + winH) / (area || 1);
        candidates.push({
          x,
          y,
          width: winW,
          height: winH,
          score: meanScore,
          className: 'face_candidate'
        });
      }
    }
    scale *= scaleFactor;
  }

  return nms(candidates, 0.3);
}

module.exports = {
  computeIoU,
  nms,
  computeIntegralImage,
  boxSum,
  detectFaces
};
