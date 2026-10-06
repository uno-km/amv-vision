/**
 * Hardware, Memory, Vulkan, and Environment Diagnostic Doctor for Node.js.
 * Open-Source under Apache License 2.0.
 */

'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { ModelCacheManager } = require('./cache');

function runDoctor(probeVulkan = false) {
  const cache = new ModelCacheManager();
  const isAndroid = fs.existsSync('/system/build.prop') || Boolean(process.env.ANDROID_ROOT);

  let totalRamMb = null;
  let availableRamMb = null;

  try {
    if (fs.existsSync('/proc/meminfo')) {
      const content = fs.readFileSync('/proc/meminfo', 'utf-8');
      for (const line of content.split('\n')) {
        if (line.startsWith('MemTotal:')) {
          totalRamMb = Math.round(parseInt(line.split(/\s+/)[1], 10) / 1024);
        } else if (line.startsWith('MemAvailable:')) {
          availableRamMb = Math.round(parseInt(line.split(/\s+/)[1], 10) / 1024);
        }
      }
    }
  } catch (e) {}

  if (!totalRamMb) {
    totalRamMb = Math.round(os.totalmem() / (1024 * 1024));
    availableRamMb = Math.round(os.freemem() / (1024 * 1024));
  }

  const loaderDetected = fs.existsSync('/system/lib64/libvulkan.so') || fs.existsSync('/system/lib/libvulkan.so');
  const driverDetected = fs.existsSync('/vendor/lib64/hw/vulkan.adreno.so') || fs.existsSync('/vendor/lib64/hw/vulkan.mali.so');

  let vulkanStatus = 'unverified';
  let amevaRuntimeDetected = false;
  try {
    const avr = require('@ameva/runtime');
    amevaRuntimeDetected = true;
    if (probeVulkan) {
      vulkanStatus = (typeof avr.isAvailable === 'function' && avr.isAvailable()) ? 'driver_detected_experimental' : 'disabled';
    }
  } catch (err) {
    if (probeVulkan) {
      vulkanStatus = (loaderDetected && driverDetected) ? 'driver_detected_experimental' : 'disabled';
    }
  }

  const packageJson = require('../package.json');
  const installed = cache.listInstalled();

  return {
    schemaVersion: 1,
    clientVersion: packageJson.version,
    runtimeVersion: packageJson.version,
    platform: {
      system: os.type(),
      machine: os.arch(),
      nodeVersion: process.version,
      isAndroid: isAndroid
    },
    hardware: {
      cpuCores: os.cpus().length,
      totalRamMb: totalRamMb,
      availableRamMb: availableRamMb
    },
    vulkan: {
      loaderDetected: loaderDetected,
      driverDetected: driverDetected,
      status: vulkanStatus
    },
    vlmRuntime: {
      installedModelsCount: installed.length,
      cacheDir: cache.cacheRoot
    },
    recommendedPreset: 'Tier M (smolvlm-500m / 4-Threads CPU Reference)'
  };
}

module.exports = {
  runDoctor
};
