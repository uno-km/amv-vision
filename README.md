# Termux-Vision: On-Device Computer Vision & Multimodal VLM Framework

[![PyPI](https://img.shields.io/pypi/v/termux-vision.svg?style=flat-square&color=0369a1)](https://pypi.org/project/termux-vision/)
[![Python](https://img.shields.io/pypi/pyversions/termux-vision.svg?style=flat-square)](https://pypi.org/project/termux-vision/)
[![npm](https://img.shields.io/npm/v/termux-vision.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/termux-vision)
[![npm downloads](https://img.shields.io/npm/dm/termux-vision.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/termux-vision)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/termux-vision)
[![Hardware Acceleration](https://img.shields.io/badge/Vulkan-1.1%2B%20Compute-orange?style=flat-square&logo=vulkan)](https://www.vulkan.org/)

> **Native On-Device Computer Vision & Multimodal Vision-Language Model (VLM) Runtime for Android Termux via Direct Bionic libc & Vulkan Compute Acceleration.**  
> *Zero PRoot. Zero Virtualization. 100% Native ARMv8.2-A NEON SIMD & Hardware GPU Offloading.*

---

## 📑 Table of Contents

1. [Overview & Key Capabilities](#1-overview--key-capabilities)
2. [Installation Guide & Prebuilt Installer](#2-installation-guide--prebuilt-installer)
3. [Enabling Hardware GPU Acceleration (with ameva-runtime)](#3-enabling-hardware-gpu-acceleration-with-ameva-runtime)
4. [Standardized CLI & Parameter Matrix](#4-standardized-cli--parameter-matrix)
5. [Dual Engine Code Examples (Python & Node.js)](#5-dual-engine-code-examples-python--nodejs)
6. [Production Diagnostics (`termux-vision doctor`)](#6-production-diagnostics-termux-vision-doctor)
7. [Real-World Benchmarks & Hardware Scorecard](#7-real-world-benchmarks--hardware-scorecard)
8. [Memory & VRAM Architecture (Zero CPU-Mapped VRAM)](#8-memory--vram-architecture)
9. [Hardware Requirements & Operational Limits](#9-hardware-requirements--operational-limits)
10. [License & Permissible Use](#10-license--permissible-use)

---

## 1. Overview & Key Capabilities

`termux-vision` is an enterprise-grade, on-device multimodal vision inference and spatial computing framework engineered specifically for mobile Android devices. Operating directly against Android's native Bionic libc ABI and host Vulkan compute drivers, `termux-vision` eliminates heavyweight desktop dependencies (OpenCV, TorchVision) and enables high-throughput visual question answering, OCR image captioning, and classical feature extraction directly on edge hardware.

* **100% Vulkan GPU Compute Canny (`0.23 ms`)**: Chains 3-pass SPIR-V compute shaders (Sobel 3x3, NMS, Hysteresis) entirely within VRAM using `vkCmdPipelineBarrier`, achieving 834x acceleration over Python without CPU memory roundtrips.
* **Ultra-Fast ARM64 NEON C++ Kernel (`3.02 ms`)**: Permanently eliminates trigonometric `atan2f` via tangent ratio bit quantization and 1-byte direction buffers, running Canny filtering in 3.02ms on Snapdragon 865 and 4.36ms on Exynos 1380.
* **Prebuilt-Asset-First Idempotent Installer (`0.005s Skip`)**: Automatically provisions verified precompiled ARM64 native binaries in 2 seconds from official releases, guaranteeing zero-build instant skip if assets already exist.
* **5-Backend Unified CLI Standard**: Enforces `['auto', 'gpu', 'vulkan', 'opencl', 'cpu']` and convenience flags (`--gpu`, `--cpu`, `--opencl`) across all subcommands.
* **Zero-Deception Fail-Fast Gatekeeper**: Strictly rejects defective text-only binaries lacking `--mmproj` (`E015`) and corrupted weights (`E014`), permanently banning silent fallbacks.
* **Full-Layer GPU Offloading (-ngl 99)**: Dispatches all transformer layers and cross-attention vision projections directly to device GPU VRAM (**0.00 MiB CPU mapped VRAM**).

---

## 2. Installation Guide & Prebuilt Installer

`termux-vision` is distributed across both Python (PyPI) and Node.js (npm) ecosystems, with official precompiled ARM64 wheel assets published on GitHub Releases.

### 2.1 Termux System Prerequisites
Launch Termux and install required native compilers, Vulkan drivers, and image libraries:
```bash
pkg update -y
pkg install -y python nodejs clang make cmake git termux-api wget vulkan-loader vulkan-headers vulkan-tools opencl-headers python-numpy libjpeg-turbo
```

### 2.2 Python Package Installation

* **Option A: Install from PyPI (Recommended)**:
  ```bash
  pip install --upgrade pip setuptools wheel
  pip install termux-vision
  ```

* **Option B: Prebuilt Native Engine Provisioning (Idempotent 0.005s)**:
  ```bash
  # Automatically download & unpack verified ARM64 prebuilt assets
  termux-vision install
  
  # Optional maintenance flags:
  # termux-vision install --force        # Force re-downloading prebuilts
  # termux-vision install --from-source   # Force compiling from local C++ source
  # termux-vision install --dry-run       # Check integrity without making changes
  ```

* **Option C: Direct GitHub Releases Wheel Asset**:
  ```bash
  # Download and install the prebuilt v1.5.0 release wheel
  pip install https://github.com/uno-km/termux-vision/releases/download/v1.5.0/termux_vision-1.5.0-py3-none-any.whl
  ```

### 2.3 Node.js / TypeScript CLI Installation
```bash
# Global CLI installation
npm install -g termux-vision

# Local project dependency
npm install termux-vision
```

### 2.4 One-Line Bootstrap Installer
Run the universal bootstrap installer to automatically configure repositories, compile native C/C++ acceleration shims, and verify hardware:
```bash
curl -sL https://raw.githubusercontent.com/uno-km/termux-vision/main/install.sh | bash
```

---

## 3. Enabling Hardware GPU Acceleration (with ameva-runtime)

To unlock mobile GPU acceleration via Vulkan compute shaders and achieve significant speedups over pure CPU execution, install **`termux-vision`** alongside **`ameva-runtime`**:

### 🌟 One-Line Installation
```bash
# Python Environment
pip install termux-vision ameva-runtime termux-llamacpp

# Node.js Environment
npm install -g termux-vision @ameva/runtime
```

### 🔮 Mobile GPU Silicon Architecture Status

| GPU Microarchitecture | Silicon / SoC Reference | Status | Optimization Mechanics |
| :--- | :--- | :--- | :--- |
| **Qualcomm Adreno GPU** | Snapdragon 8 Elite (Adreno 830)<br>Snapdragon 8 Gen 1/2/3 (Adreno 730-750)<br>Snapdragon 865 (Adreno 650) | 🟢 **Production Verified** | Direct Bionic ICD binding, SPIR-V JIT patch (`mul_mat_vec_max_cols = 2`), KGSL Watchdog defense (`GGML_VULKAN_SKIP_CHECKS="999999999"`), micro-batch prefill chunking (`-b 64 -ub 64`). |
| **ARM Mali GPU** | Exynos 2100 (Mali-G78 MP14)<br>Exynos 1380 (Mali-G68 MP5) | 🟢 **Production Verified** | Bionic Vulkan ICD binding, Tile-Based Deferred Rendering (TBDR) memory isolation, MMVQ matrix-vector kernel dispatch (`--tune-mali`). |
| **Samsung Xclipse GPU** | Exynos 2200 / 2400<br>(Xclipse 920 / 940 - AMD RDNA) | 🟡 **In Development (개발 진행 중)** | SPIR-V instruction scheduling and RDNA mobile shader alignment under active engineering. |

Verify GPU driver detection and hardware readiness:
```bash
termux-vision doctor
```

---

## 4. Standardized CLI & Parameter Matrix

`termux-vision` strictly complies with the official `uno-km` family 5-backend CLI standard:

| Parameter | Alias | Default | Description |
| :--- | :--- | :--- | :--- |
| `-b, --backend` | `-d, --device` | `auto` | Compute acceleration backend: `auto`, `gpu`, `vulkan`, `opencl`, `cpu` |
| `--gpu` / `--cpu` / `--opencl` | *N/A* | *None* | Convenience shorthand flags for backend routing |
| `-i, --image` | `--image-path` | *Required* | Path to input image (`.png`, `.jpg`, `.webp`) |
| `-p, --prompt` | *N/A* | `"Describe this image"` | Multimodal text instruction query |
| `-m, --model` | *N/A* | `smolvlm-500m` | GGUF language model path or catalog identifier |
| `--mmproj` | *N/A* | *Auto-paired* | Vision projector GGUF model path (`mmproj-*.gguf`) |
| `-n, --max-tokens` | `--n-predict` | `150` | Maximum number of generated tokens |
| `-c, --ctx-size` | `--ctx` | `2048` | Context window size |
| `-t, --threads` | *N/A* | `auto` | Number of CPU execution threads |
| `--image-size` | *N/A* | *None* | Image resolution preset (e.g. `224x224`, `384x384`) |
| `-q, --quality` | *N/A* | `optimal` | 4-tier resolution preset: `fast` (384px), `optimal` (768px), `high` (1280px), `original` (1:1) |
| `--tune-mali` | *N/A* | `False` | Enable ARM Mali GPU MMVQ tuning (`GGML_VK_FORCE_MMVQ=1`) |
| `--json` | *N/A* | `False` | Emit machine-readable JSON benchmark telemetry |
| `-v, --verbose` | *N/A* | `False` | Print detailed layer offloading and hardware logs |

### Practical CLI Usage Examples

```bash
# 1. 100% Vulkan GPU Canny Edge Detection (0.23 ms on Adreno 830)
termux-vision canny photo.jpg -o edges.png --gpu --low 40 --high 120

# 2. Ultra-Fast NEON C++ Canny Edge Detection (3.02 ms on S20 CPU)
termux-vision canny photo.jpg -o edges.png --cpu

# 3. Multimodal VLM Inference with Automated GPU Routing
termux-vision vlm photo.jpg -p "What objects are visible in this scene?"

# 4. Pure GPU Mode on ARM Mali Silicon (Galaxy S21 / A35)
termux-vision vlm photo.jpg -d gpu --tune-mali -p "Describe the text and layout."

# 5. Prebuilt Native Binary Provisioning (0.005s Idempotent Skip)
termux-vision install
```

---

## 5. Dual Engine Code Examples (Python & Node.js)

### 5.1 Python SDK
```python
import termux_vision as tv

# 1. Hardware-Accelerated Canny Edge Detection (0.23ms Vulkan GPU / 3.02ms NEON CPU)
image = tv.io.load_image("document.jpg")
grayscale = tv.transforms.to_grayscale(image)
edges = tv.cv.canny(grayscale, low_threshold=40, high_threshold=120, backend="auto")
tv.io.save_image(edges, "edges.png")

# 2. On-Device Multimodal VLM Inference (Vulkan GPU Accelerated)
with tv.vlm.load("smolvlm-500m", device="gpu") as engine:
    result = engine.describe(
        "document.jpg",
        prompt="Extract all visible text and summarize key bullet points.",
        quality="optimal",
        max_tokens=200
    )
    print(f"Backend: {result.metrics.backend} | TPS: {result.metrics.tokens_per_second:.2f} tok/s")
    print(f"Response:\n{result.text}")
```

### 5.2 Node.js / TypeScript SDK
```typescript
import tv from 'termux-vision';

// 1. Hardware Diagnostic Probe
const doctor = tv.doctor(true);
console.log(`Vulkan GPU: ${doctor.vulkan.status} | Available RAM: ${doctor.hardware.availableRamMb} MB`);

// 2. Multimodal VLM Inference
const engine = await tv.load({ modelId: 'smolvlm-500m', device: 'gpu' });
const response = await engine.describe('photo.jpg', {
  prompt: 'Identify the geometric shapes and colors.',
  quality: 'optimal',
  maxTokens: 100
});

console.log(`[${response.metrics.backend.toUpperCase()}] ${response.text}`);
engine.close();
```

---

## 6. Production Diagnostics (`termux-vision doctor`)

Termux-Vision features an integrated hardware diagnostics probe to inspect the host environment before launching inference:

```bash
termux-vision doctor
```

**Diagnostic Output Profile:**
```
=== termux-vision Diagnostic Doctor ===
  Platform : Linux (aarch64) | Android: True
  RAM      : Total 7812MB | Available 3450MB
  CPU Cores: 8 (big.LITTLE Affinity Governor active)
  Vulkan   : Loader=True | Driver=/system/lib64/libvulkan.so | Status=READY
  GPU Soc  : ARM Mali-G78 MP14 (Exynos 2100)
  Models   : 2 installed in ~/.cache/termux-vision/models
  Preset   : optimal (768px recommended)
```

---

## 7. Real-World Benchmarks & Hardware Scorecard

### 7.1 Classical Vision Filtering Latency (512x512 Image, Physical Devices)

| Algorithm / Kernel | Architecture / Acceleration | Execution Latency | Memory Overhead | Status / Verified Device |
| :--- | :--- | :--- | :--- | :--- |
| **100% Vulkan GPU Compute Canny** | **3-Pass SPIR-V Compute VRAM Chain** | **0.23 ms (Min 0.18 ms)** | **0.00 MiB CPU VRAM** | **Production (S25 Adreno 830, 834x Speedup)** |
| **ARM64 NEON C++ Canny Engine** | **Tangent-Ratio Bit Quantization (No atan2f)** | **3.02 ms ~ 3.42 ms** | **1.0 MB (uint8 buffer)** | **Production (S20: 3.02ms, S25: 3.42ms)** |
| ARM64 NEON C++ Canny Engine | Exynos 1380 Cortex-A78 NEON | 4.36 ms | 1.0 MB | Production (Galaxy A35) |
| Sobel 3x3 Gradient Convolution | ARM NEON Vectorized | 1.1 ms | 0.5 MB | Production |
| Gaussian Blur 5x5 Kernel | Separable 1D Conv | 1.8 ms | 0.5 MB | Production |
| 2D Integral Image (SAT) | Row/Col Prefix Sum | 1.2 ms | 2.0 MB | Production |
| Haar Cascade Face Detection | Candidate Classifier | 12.5 ms | 2.2 MB | Production |

### 7.2 On-Device Multimodal VLM Benchmark

| Target Device | SoC & GPU Architecture | Model Architecture | Mode | Prompt Processing | Token Generation | Mapped CPU VRAM | Vulkan GPU VRAM | Status / Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Samsung Galaxy S25** | Snapdragon 8 Elite<br>Adreno 830 | **Moondream2 1.8B f16** | **GPU (Vulkan 25/25)** | **19.84 tok/s** | **15.00 tok/s** | **0.00 MiB** | **2,706.00 MiB** | **Production Verified** |
| **Samsung Galaxy S21 5G** | Exynos 2100<br>Mali-G78 MP14 | SmolVLM-500M-Instruct | **GPU (Vulkan)** | **14.28 tok/s** | **12.65 tok/s** | **0.00 MiB** | **1,059.02 MiB** | **+58.9% vs CPU** |
| Samsung Galaxy S21 5G | Exynos 2100<br>8-Core CPU | SmolVLM-500M-Instruct | CPU (NEON) | 8.84 tok/s | 7.96 tok/s | 1,059.02 MiB | 0.00 MiB | Baseline |
| **Samsung Galaxy A35 5G** | Exynos 1380<br>Mali-G68 MP5 | SmolVLM-500M-Instruct | **GPU (Vulkan)** | **5.67 tok/s** | **5.47 tok/s** | **0.00 MiB** | **1,059.02 MiB** | **+55.8% vs CPU** |
| Samsung Galaxy A35 5G | Exynos 1380<br>8-Core CPU | SmolVLM-500M-Instruct | CPU (NEON) | 4.88 tok/s | 3.51 tok/s | 1,059.02 MiB | 0.00 MiB | Baseline |

---

## 8. Memory & VRAM Architecture

```
+---------------------------------------------------------------+
|             Physical Mobile LPDDR4X/LPDDR5 RAM (8 GB)          |
+---------------------------------------------------------------+
   |                                                    |
   v                                                    v
+-------------------------------+       +-------------------------------+
|      Android OS & Framework   |       |       Termux User Space       |
|          (~3.5 - 4.2 GB)      |       |          (~3.8 - 4.5 GB)      |
+-------------------------------+       +-------------------------------+
                                                        |
                                                        v
                                        +-------------------------------+
                                        |    Vulkan Unified Memory      |
                                        |  - Model Weights: 1059.02 MiB |
                                        |  - KV Cache     :  384.00 MiB |
                                        |  - CPU Mapped   :    0.00 MiB |
                                        +-------------------------------+
```

---

## 9. Hardware Requirements & Operational Limits

| Requirement | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Android 10+ (Termux ARM64) | Android 13+ (One UI 5.0+ / Termux Bionic) |
| **Processor (SoC)** | 8-Core ARM64 (Cortex-A55/A76) | Exynos 2100 / Snapdragon 8 Gen 2 or newer |
| **System RAM** | 6 GB LPDDR4X | 8 GB+ LPDDR5 |
| **Vulkan API** | Vulkan 1.1 with SPIR-V Compute | Vulkan 1.2+ with Subgroup 16 arithmetic |
| **Free Storage** | 2.5 GB internal storage | 6.0 GB internal storage |

---

## 10. License & Permissible Use

Licensed under the **Apache License, Version 2.0**.  
Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)) & AMEVA Open-Source Foundation (AOSF).

* [Official Documentation Portal](https://uno-km.vercel.app/lib/vision/)
* [AMEVA Open-Source Foundation](https://uno-km.vercel.app/foundation/index.html)
* [GitHub Repository](https://github.com/uno-km/termux-vision)
* [Issue Tracker](https://github.com/uno-km/termux-vision/issues)
