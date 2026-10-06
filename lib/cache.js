/**
 * Model Cache Management and Artifact Discovery for Node.js.
 * Open-Source under Apache License 2.0.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const https = require('https');
const http = require('http');
const crypto = require('crypto');

const {
  ModelNotFoundError,
  NoInstalledModelsError,
  ModelDownloadError
} = require('./errors');

const CATALOG = {
  'smolvlm-500m-q4': {
    modelId: 'smolvlm-500m-q4',
    adapter: 'smolvlm',
    tier: 'M',
    estimatedMemoryMb: 750,
    contextLimit: 2048,
    preferredResolution: 384,
    artifacts: [
      {
        role: 'language_model',
        filename: 'smolvlm-500m-instruct-q4_k_m.gguf',
        sizeBytes: 350000000,
        downloadUrl: 'https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct-GGUF/resolve/main/smolvlm-500m-instruct-q4_k_m.gguf'
      },
      {
        role: 'vision_projector',
        filename: 'mmproj-smolvlm-500m-instruct-f16.gguf',
        sizeBytes: 200000000,
        downloadUrl: 'https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-smolvlm-500m-instruct-f16.gguf'
      }
    ]
  },
  'qwen2-vl-2b-q4': {
    modelId: 'qwen2-vl-2b-q4',
    adapter: 'qwen2vl',
    tier: 'L',
    estimatedMemoryMb: 2100,
    contextLimit: 8192,
    preferredResolution: 384,
    artifacts: [
      {
        role: 'language_model',
        filename: 'Qwen2-VL-2B-Instruct-Q4_K_M.gguf',
        sizeBytes: 986046944,
        downloadUrl: 'https://huggingface.co/second-state/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf'
      },
      {
        role: 'vision_projector',
        filename: 'Qwen2-VL-2B-Instruct-vision-encoder.gguf',
        sizeBytes: 2600000000,
        downloadUrl: 'https://huggingface.co/second-state/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-vision-encoder.gguf'
      }
    ]
  }
};

class ModelCacheManager {
  constructor(customCacheRoot = null) {
    this.cacheRoot = customCacheRoot || path.join(os.homedir(), '.cache', 'termux-vision');
    this.legacyCache = path.join(os.homedir(), '.cache', 'vlm_models');
    this.modelsDir = path.join(this.cacheRoot, 'models');
    this.downloadsDir = path.join(this.cacheRoot, 'downloads');

    fs.mkdirSync(this.modelsDir, { recursive: true });
    fs.mkdirSync(this.downloadsDir, { recursive: true });
  }

  getAvailableCatalogModels() {
    return Object.keys(CATALOG).sort();
  }

  isModelInstalled(modelId) {
    if (CATALOG[modelId] && fs.existsSync(this.legacyCache)) {
      const cat = CATALOG[modelId];
      const allExist = cat.artifacts.every(a => {
        const fp = path.join(this.legacyCache, a.filename);
        return fs.existsSync(fp) && fs.statSync(fp).size > 10000000;
      });
      if (allExist) return true;
    }

    const mdir = path.join(this.modelsDir, modelId);
    if (!fs.existsSync(mdir)) return false;

    // Check ready marker or custom gguf + mmproj
    const readyMarker = path.join(mdir, 'READY');
    if (fs.existsSync(readyMarker)) return true;

    if (fs.statSync(mdir).isDirectory()) {
      const files = fs.readdirSync(mdir);
      const ggufs = files.filter(f => f.endsWith('.gguf') && fs.statSync(path.join(mdir, f)).size > 0);
      const visionGgufs = ggufs.filter(f => /mmproj|encoder|projector/i.test(f));
      const textGgufs = ggufs.filter(f => !/mmproj|encoder|projector/i.test(f));
      if ((textGgufs.length > 0 && visionGgufs.length > 0) || ggufs.length >= 2) {
        return true;
      }
    }
    return false;
  }

  listInstalled() {
    const results = [];
    if (fs.existsSync(this.modelsDir)) {
      const entries = fs.readdirSync(this.modelsDir);
      for (const name of entries) {
        const mdir = path.join(this.modelsDir, name);
        if (fs.statSync(mdir).isDirectory() && this.isModelInstalled(name)) {
          let totalBytes = 0;
          const files = fs.readdirSync(mdir);
          for (const f of files) {
            try {
              totalBytes += fs.statSync(path.join(mdir, f)).size;
            } catch (e) {}
          }
          const tier = CATALOG[name] ? CATALOG[name].tier : 'CUSTOM';
          results.push({
            modelId: name,
            tier: tier,
            state: 'READY',
            sizeMb: Math.round(totalBytes / (1024 * 1024)),
            path: mdir
          });
        }
      }
    }

    if (fs.existsSync(this.legacyCache)) {
      for (const [key, val] of Object.entries(CATALOG)) {
        if (!results.find(r => r.modelId === key)) {
          const allFound = val.artifacts.every(a => {
            const fpath = path.join(this.legacyCache, a.filename);
            return fs.existsSync(fpath) && fs.statSync(fpath).size > 10000000;
          });
          if (allFound) {
            results.push({
              modelId: key,
              tier: val.tier,
              state: 'READY',
              sizeMb: Math.round(val.artifacts.reduce((acc, a) => acc + (a.sizeBytes || 0), 0) / (1024 * 1024)),
              path: this.legacyCache
            });
          }
        }
      }
    }
    return results;
  }

  requireInstalledModel(modelId, mmprojPath = null) {
    // 1. Direct file path check (Custom 싸제 GGUF)
    const expanded = path.resolve(modelId.replace(/^~(?=$|\/|\\)/, os.homedir()));
    if (fs.existsSync(expanded) && fs.statSync(expanded).isFile()) {
      const dir = path.dirname(expanded);
      const filename = path.basename(expanded);
      let visionPath = mmprojPath ? path.resolve(mmprojPath.replace(/^~(?=$|\/|\\)/, os.homedir())) : null;
      if (!visionPath) {
        const files = fs.readdirSync(dir);
        const candidate = files.find(f => f.endsWith('.gguf') && /mmproj|encoder|projector/i.test(f));
        if (candidate) visionPath = path.join(dir, candidate);
      }
      return {
        manifest: {
          modelId: path.parse(filename).name,
          adapter: /smol/i.test(filename) ? 'smolvlm' : 'qwen2vl',
          tier: 'CUSTOM',
          contextLimit: 2048
        },
        textModelPath: expanded,
        visionModelPath: visionPath
      };
    }

    // 2. Installed model search
    const installed = this.listInstalled();
    const installedIds = installed.map(m => m.modelId);

    if (installedIds.length === 0) {
      throw new NoInstalledModelsError(this.getAvailableCatalogModels());
    }

    if (!installedIds.includes(modelId)) {
      throw new ModelNotFoundError(modelId, installedIds, this.getAvailableCatalogModels());
    }

    const mdir = path.join(this.modelsDir, modelId);
    let textPath = null;
    let visionPath = mmprojPath ? path.resolve(mmprojPath) : null;

    if (CATALOG[modelId]) {
      const cat = CATALOG[modelId];
      const searchDirs = [mdir, this.legacyCache];
      for (const sDir of searchDirs) {
        if (!fs.existsSync(sDir)) continue;
        for (const a of cat.artifacts) {
          const p = path.join(sDir, a.filename);
          if (fs.existsSync(p)) {
            if (a.role === 'language_model') textPath = p;
            if (a.role === 'vision_projector') visionPath = p;
          }
        }
        if (textPath && visionPath) break;
      }
      return {
        manifest: cat,
        textModelPath: textPath,
        visionModelPath: visionPath
      };
    }

    // Custom directory
    const files = fs.readdirSync(mdir);
    const ggufs = files.filter(f => f.endsWith('.gguf'));
    const visionFiles = ggufs.filter(f => /mmproj|encoder|projector/i.test(f));
    const textFiles = ggufs.filter(f => !/mmproj|encoder|projector/i.test(f));

    textPath = textFiles.length > 0 ? path.join(mdir, textFiles[0]) : (ggufs.length > 0 ? path.join(mdir, ggufs[0]) : null);
    if (!visionPath) {
      visionPath = visionFiles.length > 0 ? path.join(mdir, visionFiles[0]) : (ggufs.length > 1 ? path.join(mdir, ggufs[1]) : null);
    }

    return {
      manifest: {
        modelId: modelId,
        adapter: /smol/i.test(modelId) ? 'smolvlm' : 'qwen2vl',
        tier: 'CUSTOM',
        contextLimit: 2048
      },
      textModelPath: textPath,
      visionModelPath: visionPath
    };
  }

  async downloadFile(url, destPath, onProgress = null) {
    const partialPath = destPath + '.partial';
    if (fs.existsSync(partialPath)) fs.unlinkSync(partialPath);

    return new Promise((resolve, reject) => {
      const getFollow = (currentUrl) => {
        const client = currentUrl.startsWith('https') ? https : http;
        const req = client.get(currentUrl, { headers: { 'User-Agent': 'Mozilla/5.0 (termux-vision node)' } }, (res) => {
          if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            return getFollow(res.headers.location);
          }
          if (res.statusCode !== 200) {
            return reject(new Error(`HTTP ${res.statusCode}: ${res.statusMessage}`));
          }
          const total = parseInt(res.headers['content-length'] || '0', 10);
          let downloaded = 0;
          const out = fs.createWriteStream(partialPath);

          res.on('data', (chunk) => {
            downloaded += chunk.length;
            if (onProgress) onProgress(path.basename(destPath), downloaded, total);
          });

          res.pipe(out);
          out.on('finish', () => {
            out.close(() => {
              if (fs.existsSync(destPath)) fs.unlinkSync(destPath);
              fs.renameSync(partialPath, destPath);
              resolve();
            });
          });
        });
        req.on('error', (err) => {
          if (fs.existsSync(partialPath)) try { fs.unlinkSync(partialPath); } catch (e) {}
          reject(err);
        });
      };
      getFollow(url);
    });
  }

  async install(modelId, onProgress = null) {
    if (!CATALOG[modelId]) {
      const installed = this.listInstalled().map(m => m.modelId);
      throw new ModelDownloadError(modelId, `Model '${modelId}' is not in catalog.`, installed);
    }
    const manifest = CATALOG[modelId];
    const targetDir = path.join(this.modelsDir, modelId);
    fs.mkdirSync(targetDir, { recursive: true });

    for (const art of manifest.artifacts) {
      const dest = path.join(targetDir, art.filename);
      if (fs.existsSync(dest) && fs.statSync(dest).size > 0) continue;
      await this.downloadFile(art.downloadUrl, dest, onProgress);
    }

    fs.writeFileSync(path.join(targetDir, 'READY'), 'OK\n', 'utf-8');
    fs.writeFileSync(path.join(targetDir, 'manifest.json'), JSON.stringify(manifest, null, 2), 'utf-8');
  }

  remove(modelId) {
    const mdir = path.join(this.modelsDir, modelId);
    if (fs.existsSync(mdir)) {
      fs.rmSync(mdir, { recursive: true, force: true });
      return true;
    }
    return false;
  }
}

module.exports = {
  CATALOG,
  ModelCacheManager
};
