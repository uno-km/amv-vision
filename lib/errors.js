/**
 * Structured Exception Classes and Dynamic User Error Contracts for Node.js.
 * Open-Source under Apache License 2.0.
 */

'use strict';

class TermuxVisionError extends Error {
  constructor(message) {
    super(message);
    this.name = this.constructor.name;
  }
}

class ModelNotFoundError extends TermuxVisionError {
  constructor(modelId, availableLocal = [], catalogModels = []) {
    let msg = `Model '${modelId}' was not found locally or on remote registry.`;
    if (availableLocal && availableLocal.length > 0) {
      msg += `\n\nCurrently installed local models (${availableLocal.length}):\n` +
        availableLocal.map(m => `  - ${m}`).join('\n') +
        `\n\nTo run with an installed model:\n  termux-vision vlm <IMAGE> --model ${availableLocal[0]}`;
    } else if (catalogModels && catalogModels.length > 0) {
      msg += `\n\nNo models installed locally. Available catalog presets:\n` +
        catalogModels.map(m => `  - ${m}`).join('\n') +
        `\n\nInstall an official model:\n  termux-vision model install ${catalogModels[0]}`;
    }
    super(msg);
    this.modelId = modelId;
    this.availableLocal = availableLocal;
    this.catalogModels = catalogModels;
  }
}

class NoInstalledModelsError extends ModelNotFoundError {
  constructor(catalogModels = []) {
    const catText = catalogModels.length > 0 ? catalogModels.map(m => `  - ${m}`).join('\n') : '  (none)';
    const msg = 
      "No installed VLM models were found in local cache (~/.cache/termux-vision/models).\n\n" +
      "How to use VLM models:\n" +
      "1. Install an official catalog model:\n" +
      `${catText}\n` +
      "   Example: termux-vision model install smolvlm-500m-q4\n\n" +
      "2. Use custom/external ('싸제') models:\n" +
      "   - Place your text model GGUF and vision projector mmproj GGUF in:\n" +
      "     ~/.cache/termux-vision/models/<custom_model_name>/\n" +
      "   - Or pass direct file paths:\n" +
      "     termux-vision vlm <IMAGE> --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf\n" +
      "   (Note: VLM inference requires both a language model .gguf and a vision projector mmproj-*.gguf)";
    super('all', [], catalogModels);
    this.message = msg;
  }
}

class ModelSelectionRequiredError extends ModelNotFoundError {
  constructor(installedModels = []) {
    const modelsText = installedModels.map(m => `  - ${m}`).join('\n');
    const exampleModel = installedModels[0] || 'MODEL_ID';
    const msg =
      `Multiple models are installed. Please specify one with --model:\n` +
      `${modelsText}\n\n` +
      `Example:\n` +
      `  termux-vision vlm <IMAGE> --model ${exampleModel} -p "Describe this image"`;
    super('multiple', installedModels, []);
    this.message = msg;
  }
}

class VulkanNotAvailableError extends TermuxVisionError {
  constructor(reason = '') {
    const detail = reason ? `\nFailure detail: ${reason}` : '';
    const msg =
      `Vulkan GPU acceleration is unavailable or failed on this device.${detail}\n\n` +
      `[Action Required] Explicit GPU mode cannot proceed. Please switch to CPU mode:\n` +
      `  CLI: --device cpu\n` +
      `  Node.js API: device: 'cpu'\n` +
      `Or use automatic detection: --device auto (device: 'auto')`;
    super(msg);
    this.reason = reason;
  }
}

class RuntimeNotFoundError extends TermuxVisionError {
  constructor(searchedPaths = []) {
    let searched = '';
    if (searchedPaths.length > 0) {
      searched = '\n\nSearched paths:\n' + searchedPaths.map(p => `  - ${p}`).join('\n');
    }
    const msg =
      `Required runtime engine 'llama-cli' was not found.${searched}\n\n` +
      `Please install the official termux-llamacpp native runtime:\n` +
      `  - Python:  pip install termux-llamacpp\n` +
      `  - Node.js: npm install -g termux-llamacpp\n` +
      `  - Setup:   termux-llama install (compiles native binary if needed)`;
    super(msg);
    this.searchedPaths = searchedPaths;
  }
}

class ModelDownloadError extends TermuxVisionError {
  constructor(source, reason = '', availableLocal = []) {
    let msg = `Failed to download model from '${source}': ${reason}`;
    if (availableLocal && availableLocal.length > 0) {
      msg += `\n\nInstalled local models available:\n` +
        availableLocal.map(m => `  - ${m}`).join('\n') +
        `\nRun with local model: termux-vision vlm <IMAGE> --model ${availableLocal[0]}`;
    }
    super(msg);
    this.source = source;
    this.reason = reason;
  }
}

module.exports = {
  TermuxVisionError,
  ModelNotFoundError,
  NoInstalledModelsError,
  ModelSelectionRequiredError,
  VulkanNotAvailableError,
  RuntimeNotFoundError,
  ModelDownloadError
};
