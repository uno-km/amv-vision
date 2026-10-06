/**
 * termux-vision: Native On-Device Computer Vision & VLM Multimodal Inference Framework.
 * Dual-Engine (Python & Node.js/TypeScript) Native Module for Android Termux & ARM64.
 * Open-Source under Apache License 2.0.
 */

'use strict';

const errors = require('./lib/errors');
const cache = require('./lib/cache');
const vlm = require('./lib/vlm');
const doctor = require('./lib/doctor');
const cv = require('./lib/cv');
const detect = require('./lib/detect');

const packageJson = require('./package.json');
const version = packageJson.version;
const cacheManager = new cache.ModelCacheManager();
cacheManager.ModelCacheManager = cache.ModelCacheManager;

module.exports = {
  version,
  __version__: version,
  errors,
  cache: cacheManager,
  ModelCacheManager: cache.ModelCacheManager,
  vlm,
  load: vlm.load,
  doctor: doctor.runDoctor,
  cv,
  detect
};
