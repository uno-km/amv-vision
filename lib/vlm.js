/**
 * Node.js VLM Engine & Supervised llama-cli Subprocess Bridge.
 * Open-Source under Apache License 2.0.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');

const {
  VulkanNotAvailableError,
  RuntimeNotFoundError,
  TermuxVisionError
} = require('./errors');
const { ModelCacheManager } = require('./cache');

function isVulkanFailure(text) {
  const lower = String(text).toLowerCase();
  const markers = [
    'vulkan',
    'vk_error',
    'device lost',
    'failed to initialize gpu',
    'no vulkan device',
    'gpu backend',
    'vkcreateinstance',
    'vkcreatedevice',
    'ggml_vulkan'
  ];
  return markers.some(m => lower.includes(m));
}

function resolveLlamaCli(explicitPath = null) {
  if (explicitPath) {
    const p = path.resolve(explicitPath);
    if (fs.existsSync(p)) return p;
    throw new RuntimeNotFoundError([p]);
  }

  if (process.env.TERMUX_VISION_LLAMA_CLI && fs.existsSync(process.env.TERMUX_VISION_LLAMA_CLI)) {
    return process.env.TERMUX_VISION_LLAMA_CLI;
  }

  const prefix = process.env.PREFIX || '/data/data/com.termux/files/usr';
  const xdgCache = process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache');
  const candidates = [
    path.join(xdgCache, 'termux-llamacpp', 'bin', 'llama-cli'),
    path.join(os.homedir(), '.local', 'bin', 'llama-cli'),
    path.join(prefix, 'bin', 'llama-cli'),
    path.join(prefix, 'bin', 'termux-llama-cli'),
    path.join(prefix, 'bin', 'llama-mtmd-cli'),
    path.join(os.homedir(), '.termux-llamacpp', 'current', 'bin', 'llama-cli'),
    path.join(os.homedir(), '.termux-llama', 'current', 'bin', 'llama-cli'),
    path.join(os.homedir(), 'bin', 'llama-cli'),
    'llama-cli'
  ];

  const pathDirs = (process.env.PATH || '').split(path.delimiter);
  for (const dir of pathDirs) {
    const full = path.join(dir, 'llama-cli' + (process.platform === 'win32' ? '.exe' : ''));
    if (fs.existsSync(full)) return full;
  }

  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }

  throw new RuntimeNotFoundError(candidates);
}

class NodeVLMContext {
  constructor(options) {
    this.manifest = options.manifest;
    this.textModelPath = options.textModelPath;
    this.visionModelPath = options.visionModelPath;
    this.executable = options.executable;
    this.threads = options.threads || 4;
    this.backend = options.backend || 'auto';
    this.fallback = options.fallback !== false;
    this.contextLimit = options.contextLimit || options.manifest.contextLimit || 1024;
    this.customNgl = options.ngl;
    this.closed = false;
  }

  async _executeOnce(imagePath, prompt, options, targetBackend) {
    const nglVal = this.customNgl !== undefined && this.customNgl !== null ? String(this.customNgl) : (['vulkan', 'auto', 'gpu'].includes(targetBackend) ? '99' : '0');
    const tmpPrompt = path.join(os.tmpdir(), `tv_prompt_${Date.now()}_${Math.random().toString(36).substring(7)}.txt`);
    
    const formattedPrompt = options.systemPrompt ? 
      `System: ${options.systemPrompt}\n<image>\nUser: ${prompt}\nAssistant:` : 
      `<image>\nUser: ${prompt}\nAssistant:`;
    fs.writeFileSync(tmpPrompt, formattedPrompt, 'utf-8');

    const cliArgs = [
      '-m', this.textModelPath,
      '--mmproj', this.visionModelPath,
      '--image', imagePath,
      '-f', tmpPrompt,
      '-t', String(this.threads),
      '-c', String(this.contextLimit),
      '-n', String(options.maxTokens || 150),
      '--temp', String(options.temperature !== undefined ? options.temperature : 0.2),
      '-ngl', nglVal,
      '--single-turn',
      '--simple-io'
    ];

    if (['auto', 'vulkan', 'gpu'].includes(targetBackend)) {
      cliArgs.push('--device', targetBackend === 'auto' ? 'auto' : 'vulkan');
    }

    if (options.repeatPenalty !== undefined) cliArgs.push('--repeat-penalty', String(options.repeatPenalty));
    if (options.topP !== undefined) cliArgs.push('--top-p', String(options.topP));
    if (options.topK !== undefined) cliArgs.push('--top-k', String(options.topK));
    if (options.seed !== undefined) cliArgs.push('-s', String(options.seed));

    return new Promise((resolve, reject) => {
      const t0 = Date.now();
      const proc = spawn(this.executable, cliArgs, {
        stdio: ['ignore', 'pipe', 'pipe']
      });

      let stdoutData = '';
      let stderrData = '';

      proc.stdout.on('data', (d) => { stdoutData += d.toString('utf-8'); });
      proc.stderr.on('data', (d) => { stderrData += d.toString('utf-8'); });

      proc.on('close', (code, signal) => {
        try { if (fs.existsSync(tmpPrompt)) fs.unlinkSync(tmpPrompt); } catch (e) {}
        const totalMs = Date.now() - t0;

        if (code !== 0) {
          const errText = (stderrData.trim() || stdoutData.trim());
          if (code === 137 || signal === 'SIGKILL' || code === -9) {
            return reject(new TermuxVisionError(
              `VLM inference process was terminated by system (OOM / LowMemoryKiller / SIGKILL).\n` +
              `[Action Recommendation] Use a smaller model (e.g. smolvlm-500m-q4), reduce threads (-t 2), or close background apps.`
            ));
          }
          return reject(new TermuxVisionError(`llama-cli exited with code ${code}: ${errText}`));
        }

        const lines = stdoutData.split('\n');
        const contentLines = [];
        let startCapture = false;
        let tps = null;

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('>')) {
            startCapture = true;
            continue;
          }
          if (startCapture) {
            if (line.includes('Generation:') && line.includes('t/s')) {
              try {
                const match = line.match(/Generation:\s*([0-9.]+)\s*t\/s/);
                if (match) tps = parseFloat(match[1]);
              } catch (e) {}
              continue;
            }
            if (line.includes('Exiting') || line.includes('main: image')) continue;
            const cleaned = line.replace(/[|\-\/\\]/g, '').trim();
            if (cleaned) contentLines.push(cleaned);
          }
        }

        const textOutput = contentLines.length > 0 ? contentLines.join(' ') : stdoutData.trim();
        resolve({
          text: textOutput,
          finishReason: 'stop',
          wordCount: textOutput.split(/\s+/).filter(Boolean).length,
          metrics: {
            backend: targetBackend,
            modelId: this.manifest.modelId,
            decodeMs: totalMs,
            tokensPerSecond: tps
          },
          warnings: []
        });
      });

      proc.on('error', (err) => {
        try { if (fs.existsSync(tmpPrompt)) fs.unlinkSync(tmpPrompt); } catch (e) {}
        reject(err);
      });
    });
  }

  async describe(imagePath, options = {}) {
    if (this.closed) throw new Error('Cannot describe with closed VLMContext.');
    if (!imagePath || typeof imagePath !== 'string' || !imagePath.trim()) {
      throw new Error("Parameter 'imagePath' cannot be null or empty.");
    }
    const resolvedImg = path.resolve(imagePath.replace(/^~(?=$|\/|\\)/, os.homedir()));
    if (!fs.existsSync(resolvedImg)) {
      throw new Error(`Input image file not found: '${resolvedImg}'`);
    }

    const prompt = options.prompt || '이 사진 속 인물의 표정, 옷차림, 자세, 그리고 배경 환경을 한국어로 간결하게 요약 설명해줘.';
    if (!prompt || !prompt.trim()) {
      throw new Error("Parameter 'prompt' cannot be empty.");
    }

    // Strict validation
    if (options.maxTokens !== undefined && options.maxTokens <= 0) {
      throw new Error(`Parameter 'maxTokens' must be > 0. Received: ${options.maxTokens}`);
    }
    if (options.temperature !== undefined && options.temperature < 0) {
      throw new Error(`Parameter 'temperature' must be >= 0. Received: ${options.temperature}`);
    }
    if (options.topP !== undefined && (options.topP <= 0 || options.topP > 1.0)) {
      throw new Error(`Parameter 'topP' must be between (0.0, 1.0]. Received: ${options.topP}`);
    }

    let targetImg = resolvedImg;
    let tempImg = null;
    const quality = (options.quality || 'optimal').toLowerCase().trim();
    const maxDim = options.maxDim ? parseInt(options.maxDim, 10) : null;

    if (quality !== 'original' || maxDim !== null) {
      const limit = maxDim || (quality === 'high' ? 1280 : (quality === 'fast' ? 384 : 768));
      try {
        const { spawnSync } = require('child_process');
        const tmpTarget = path.join(os.tmpdir(), `tv_scaled_${Date.now()}_${Math.random().toString(36).substring(7)}.jpg`);
        const scalePyCode = 'import sys, termux_vision as tv; img = tv.io.load_image(sys.argv[1]); dims = tv.transforms.resolve_target_dimensions(img.shape[1], img.shape[0], max_dim=int(sys.argv[3])) or (img.shape[1], img.shape[0]); scaled = tv.transforms.resize(img, dims); tv.io.save_image(scaled, sys.argv[2])';
        let scaleRes = spawnSync('python3', [
          '-c',
          scalePyCode,
          resolvedImg,
          tmpTarget,
          String(limit)
        ]);
        if (scaleRes.status !== 0 || !fs.existsSync(tmpTarget)) {
          scaleRes = spawnSync('python', [
            '-c',
            scalePyCode,
            resolvedImg,
            tmpTarget,
            String(limit)
          ]);
        }
        if (scaleRes.status === 0 && fs.existsSync(tmpTarget)) {
          targetImg = tmpTarget;
          tempImg = tmpTarget;
        }
      } catch (e) {}
    }

    try {
      return await this._executeOnce(targetImg, prompt, options, this.backend);
    } catch (err) {
      if (this.backend === 'vulkan') {
        if (this.fallback && isVulkanFailure(err.message)) {
          const fallbackRes = await this._executeOnce(resolvedImg, prompt, options, 'cpu');
          fallbackRes.warnings.push(`Vulkan execution failed; retried on CPU: ${err.message}`);
          return fallbackRes;
        } else {
          throw new VulkanNotAvailableError(err.message);
        }
      }
      throw err;
    } finally {
      if (tempImg && fs.existsSync(tempImg)) {
        try { fs.unlinkSync(tempImg); } catch (e) {}
      }
    }
  }

  async ask(imagePath, question, options = {}) {
    if (!question || !question.trim()) throw new Error("Parameter 'question' cannot be null or empty.");
    const res = await this.describe(imagePath, { ...options, prompt: question });
    return res.text;
  }

  close() {
    this.closed = true;
  }
}

async function load(options = {}) {
  const cache = new ModelCacheManager(options.cacheRoot || null);

  if (options.modelId && options.modelPath) {
    throw new Error("Cannot specify both 'modelId' and 'modelPath'. Use 'modelId' for catalog presets or 'modelPath' for explicit .gguf file paths.");
  }

  let modelInfo;
  if (options.modelPath) {
    const expanded = path.resolve(options.modelPath.replace(/^~(?=$|\/|\\)/, os.homedir()));
    if (!fs.existsSync(expanded) || !fs.statSync(expanded).isFile()) {
      throw new Error(`Model file not found at 'modelPath': ${options.modelPath}`);
    }
    modelInfo = cache.requireInstalledModel(expanded, options.mmprojPath || null);
  } else {
    const modelId = options.modelId || 'smolvlm-500m-q4';
    if (!modelId || !String(modelId).trim()) {
      throw new Error("Parameter 'modelId' cannot be null or empty.");
    }
    if (options.allowDownload && !cache.isModelInstalled(modelId)) {
      await cache.install(modelId);
    }
    modelInfo = cache.requireInstalledModel(modelId, options.mmprojPath || null);
  }
  const executable = resolveLlamaCli(options.runtimePath || null);

  const reqDevice = (options.device || 'auto').toLowerCase().trim();
  let actualBackend = 'cpu';
  let actualFallback = false;

  if (reqDevice === 'auto') {
    actualBackend = 'auto';
    actualFallback = true;
  } else if (['vulkan', 'gpu', 'vulkan-force'].includes(reqDevice)) {
    actualBackend = 'vulkan';
    actualFallback = false;
  } else {
    actualBackend = 'cpu';
    actualFallback = false;
  }

  let threads = 4;
  if (typeof options.threads === 'number') {
    if (options.threads <= 0) throw new Error(`Parameter 'threads' must be > 0. Received: ${options.threads}`);
    threads = Math.max(1, Math.min(128, options.threads));
  }

  return new NodeVLMContext({
    manifest: modelInfo.manifest,
    textModelPath: modelInfo.textModelPath,
    visionModelPath: modelInfo.visionModelPath,
    executable: executable,
    threads: threads,
    backend: actualBackend,
    fallback: actualFallback,
    contextLimit: options.contextLimit,
    ngl: options.ngl
  });
}

module.exports = {
  load,
  resolveLlamaCli,
  NodeVLMContext
};
