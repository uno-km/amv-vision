/**
 * Fast Pure-JS Spatial Filters & Classical Computer Vision for Node.js.
 * Open-Source under Apache License 2.0.
 */

'use strict';

function gaussianBlur(grayPixels, width, height) {
  const output = new Float32Array(width * height);
  // 5x5 normalized Gaussian kernel with sigma=1.4
  const kernel = [
    2/159,  4/159,  5/159,  4/159, 2/159,
    4/159,  9/159, 12/159,  9/159, 4/159,
    5/159, 12/159, 15/159, 12/159, 5/159,
    4/159,  9/159, 12/159,  9/159, 4/159,
    2/159,  4/159,  5/159,  4/159, 2/159
  ];

  for (let y = 2; y < height - 2; y++) {
    for (let x = 2; x < width - 2; x++) {
      let sum = 0;
      let k = 0;
      for (let ky = -2; ky <= 2; ky++) {
        for (let kx = -2; kx <= 2; kx++) {
          sum += grayPixels[(y + ky) * width + (x + kx)] * kernel[k++];
        }
      }
      output[y * width + x] = sum;
    }
  }
  return output;
}

function sobel(grayPixels, width, height) {
  const gradX = new Float32Array(width * height);
  const gradY = new Float32Array(width * height);
  const magnitude = new Float32Array(width * height);
  const angleDeg = new Float32Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const gx =
        -grayPixels[(y - 1) * width + (x - 1)] + grayPixels[(y - 1) * width + (x + 1)] +
        -2 * grayPixels[y * width + (x - 1)] + 2 * grayPixels[y * width + (x + 1)] +
        -grayPixels[(y + 1) * width + (x - 1)] + grayPixels[(y + 1) * width + (x + 1)];

      const gy =
        -grayPixels[(y - 1) * width + (x - 1)] - 2 * grayPixels[(y - 1) * width + x] - grayPixels[(y - 1) * width + (x + 1)] +
        grayPixels[(y + 1) * width + (x - 1)] + 2 * grayPixels[(y + 1) * width + x] + grayPixels[(y + 1) * width + (x + 1)];

      gradX[idx] = gx;
      gradY[idx] = gy;
      magnitude[idx] = Math.sqrt(gx * gx + gy * gy);
      let deg = Math.atan2(gy, gx) * (180.0 / Math.PI);
      if (deg < 0) deg += 180.0;
      angleDeg[idx] = deg;
    }
  }

  return { gradX, gradY, magnitude, angleDeg };
}

function canny(grayPixels, width, height, lowThreshold = 40.0, highThreshold = 120.0) {
  const blurred = gaussianBlur(grayPixels, width, height);
  const { magnitude, angleDeg } = sobel(blurred, width, height);
  const dst = new Uint8Array(width * height);
  const queue = [];

  // 1. NMS & Double Thresholding
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const c = magnitude[idx];
      if (c < lowThreshold) continue;

      const deg = angleDeg[idx];
      let p1 = 0.0, p2 = 0.0;

      if ((deg >= 0 && deg < 22.5) || (deg >= 157.5 && deg <= 180)) {
        p1 = magnitude[y * width + (x - 1)];
        p2 = magnitude[y * width + (x + 1)];
      } else if (deg >= 22.5 && deg < 67.5) {
        p1 = magnitude[(y - 1) * width + (x + 1)];
        p2 = magnitude[(y + 1) * width + (x - 1)];
      } else if (deg >= 67.5 && deg < 112.5) {
        p1 = magnitude[(y - 1) * width + x];
        p2 = magnitude[(y + 1) * width + x];
      } else {
        p1 = magnitude[(y - 1) * width + (x - 1)];
        p2 = magnitude[(y + 1) * width + (x + 1)];
      }

      if (c >= p1 && c >= p2) {
        if (c >= highThreshold) {
          dst[idx] = 255;
          queue.push(idx);
        } else {
          dst[idx] = 75; // Weak edge candidate
        }
      }
    }
  }

  // 2. 8-connected BFS Hysteresis Tracking
  const dx = [-1,  0,  1, -1, 1, -1, 0, 1];
  const dy = [-1, -1, -1,  0, 0,  1, 1, 1];
  let qHead = 0;

  while (qHead < queue.length) {
    const curr = queue[qHead++];
    const cx = curr % width;
    const cy = Math.floor(curr / width);

    for (let k = 0; k < 8; k++) {
      const nx = cx + dx[k];
      const ny = cy + dy[k];
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        const nIdx = ny * width + nx;
        if (dst[nIdx] === 75) {
          dst[nIdx] = 255;
          queue.push(nIdx);
        }
      }
    }
  }

  // 3. Suppress remaining unconnected weak edges
  for (let i = 0; i < dst.length; i++) {
    if (dst[i] === 75) {
      dst[i] = 0;
    }
  }

  return dst;
}

module.exports = {
  gaussianBlur,
  sobel,
  canny
};
