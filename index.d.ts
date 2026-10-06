/**
 * Type declarations for termux-vision Node.js SDK
 */

export interface VLMDescribeOptions {
  prompt?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface InferenceMetrics {
  backend: string;
  modelId: string;
  decodeMs: number;
  tokensPerSecond?: number;
}

export interface VLMResult {
  text: string;
  finishReason: string;
  wordCount: number;
  metrics: InferenceMetrics;
  warnings: string[];
}

export interface VLMContext {
  describe(imagePath: string, options?: VLMDescribeOptions): Promise<VLMResult>;
  ask(imagePath: string, question: string): Promise<string>;
  close(): void;
}

export interface LoadOptions {
  modelId?: string;
  device?: 'auto' | 'cpu' | 'vulkan' | 'gpu' | 'vulkan-force';
  threads?: number | 'auto';
  runtimePath?: string;
  mmprojPath?: string;
  allowDownload?: boolean;
}

export interface DiagnosticReport {
  schemaVersion: number;
  platform: {
    system: string;
    machine: string;
    isAndroid: boolean;
  };
  hardware: {
    cpuCores: number;
    totalRamMb: number | null;
    availableRamMb: number | null;
  };
  vulkan: {
    loaderDetected: boolean;
    driverDetected: boolean;
    status: string;
  };
  modelsCount: number;
}

export interface ModelInfo {
  modelId: string;
  tier: string;
  state: string;
  sizeMb: number;
  path: string;
}

export namespace vlm {
  export function load(options?: LoadOptions): Promise<VLMContext>;
}

export namespace cache {
  export function listInstalled(): ModelInfo[];
  export function install(modelId: string, onProgress?: (file: string, downloaded: number, total: number) => void): Promise<void>;
  export function remove(modelId: string): boolean;
}

export namespace doctor {
  export function run(probeVulkan?: boolean): DiagnosticReport;
}

export namespace cv {
  export function canny(grayPixels: Uint8Array, width: number, height: number, low?: number, high?: number): Uint8Array;
  export function sobel(grayPixels: Uint8Array, width: number, height: number): { gradX: Float32Array; gradY: Float32Array; magnitude: Float32Array };
}

export namespace detect {
  export interface BBox {
    x: number;
    y: number;
    width: number;
    height: number;
    score?: number;
  }
  export function nms(boxes: BBox[], iouThreshold?: number): BBox[];
}
