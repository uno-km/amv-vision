/**
 * Node.js Smoke and Feature Test Suite for termux-vision.
 * Zero-Hype & Truthful Diagnostics under Apache License 2.0.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const assert = require('assert');
const { spawnSync } = require('child_process');

const tv = require('../index');

console.log('================================================================');
console.log('RUNNING termux-vision NODE.JS ENGINE VERIFICATION SUITE');
console.log('================================================================');

// Test 1: Module structure and exports
console.log('[Test 1] Verifying Node.js exports and version...');
assert.strictEqual(typeof tv.version, 'string');
assert.strictEqual(typeof tv.load, 'function');
assert.strictEqual(typeof tv.doctor, 'function');
assert.strictEqual(typeof tv.cv.canny, 'function');
assert.strictEqual(typeof tv.detect.nms, 'function');
console.log('  -> PASS: All module exports verified.');

// Test 2: Doctor report
console.log('[Test 2] Verifying Diagnostic Doctor output...');
const docRep = tv.doctor(true);
assert.strictEqual(docRep.schemaVersion, 1);
assert.strictEqual(typeof docRep.hardware.cpuCores, 'number');
assert.strictEqual(typeof docRep.vulkan.loaderDetected, 'boolean');
console.log(`  -> PASS: Doctor diagnostic OK (Platform: ${docRep.platform.system}, Cores: ${docRep.hardware.cpuCores})`);

// Test 3: Classical CV Canny & Sobel
console.log('[Test 3] Verifying Pure-JS Canny and Sobel filters...');
const width = 64;
const height = 64;
const dummyGray = new Uint8Array(width * height);
for (let y = 0; y < height; y++) {
  for (let x = 0; x < width; x++) {
    dummyGray[y * width + x] = (x > 32) ? 255 : 0;
  }
}
const edges = tv.cv.canny(dummyGray, width, height, 30, 100);
assert.strictEqual(edges.length, width * height);
// There should be detected edges around column 32
let edgeCount = 0;
for (let i = 0; i < edges.length; i++) {
  if (edges[i] === 255) edgeCount++;
}
assert(edgeCount > 0, 'Edge detection should identify vertical step edge');
console.log(`  -> PASS: Canny edge detection computed ${edgeCount} edge pixels.`);

// Test 4: BoundingBox and NMS
console.log('[Test 4] Verifying NMS and IoU geometry calculations...');
const boxes = [
  { x: 10, y: 10, width: 50, height: 50, score: 0.95 },
  { x: 12, y: 12, width: 48, height: 48, score: 0.85 }, // High overlap with first
  { x: 100, y: 100, width: 40, height: 40, score: 0.90 } // Disjoint
];
const filtered = tv.detect.nms(boxes, 0.4);
assert.strictEqual(filtered.length, 2);
assert.strictEqual(filtered[0].score, 0.95);
assert.strictEqual(filtered[1].score, 0.90);
console.log('  -> PASS: NMS successfully suppressed overlapping candidate.');

// Test 5: Cache Manager & Custom Model Discovery
console.log('[Test 5] Verifying Cache Manager custom model discovery...');
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tv_node_test_'));
try {
  const modelsDir = path.join(tmpDir, 'models');
  const customModelDir = path.join(modelsDir, 'custom-vlm-node');
  fs.mkdirSync(customModelDir, { recursive: true });

  fs.writeFileSync(path.join(customModelDir, 'model.gguf'), Buffer.from('GGUF_TEXT'));
  fs.writeFileSync(path.join(customModelDir, 'mmproj.gguf'), Buffer.from('GGUF_VISION'));

  const cache = new tv.cache.ModelCacheManager(tmpDir);
  const installed = cache.listInstalled();
  assert.strictEqual(installed.length, 1);
  assert.strictEqual(installed[0].modelId, 'custom-vlm-node');

  const resolved = cache.requireInstalledModel('custom-vlm-node');
  assert.strictEqual(resolved.manifest.modelId, 'custom-vlm-node');
  assert.strictEqual(resolved.textModelPath, path.join(customModelDir, 'model.gguf'));
  assert.strictEqual(resolved.visionModelPath, path.join(customModelDir, 'mmproj.gguf'));
  console.log('  -> PASS: Custom (싸제) model discovered and resolved.');
} finally {
  fs.rmSync(tmpDir, { recursive: true, force: true });
}

// Test 6: Empty cache guidance error contract
console.log('[Test 6] Verifying NoInstalledModelsError guidance contract...');
const emptyDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tv_empty_test_'));
try {
  const emptyCache = new tv.cache.ModelCacheManager(emptyDir);
  let caught = false;
  try {
    emptyCache.requireInstalledModel('nonexistent');
  } catch (err) {
    caught = true;
    assert(err instanceof tv.errors.NoInstalledModelsError);
    assert(err.message.includes('smolvlm-500m-q4'));
    assert(err.message.includes('싸제') || err.message.includes('custom'));
  }
  assert(caught, 'Should throw NoInstalledModelsError on empty cache');
  console.log('  -> PASS: NoInstalledModelsError provides dynamic guidance.');
} finally {
  fs.rmSync(emptyDir, { recursive: true, force: true });
}

// Test 7: CLI Binary Smoke Check
console.log('[Test 7] Verifying bin/cli.js execution via node...');
const cliPath = path.resolve(__dirname, '..', 'bin', 'cli.js');
const cliRes = spawnSync(process.execPath, [cliPath, 'doctor', '--json'], { encoding: 'utf-8' });
assert.strictEqual(cliRes.status, 0);
const cliJson = JSON.parse(cliRes.stdout);
assert.strictEqual(cliJson.schemaVersion || cliJson.schema_version, 1);
console.log('  -> PASS: bin/cli.js doctor --json returned valid schema.');

console.log('================================================================');
console.log('ALL NODE.JS VERIFICATION TESTS PASSED (7 / 7 PASSED)');
console.log('================================================================');
