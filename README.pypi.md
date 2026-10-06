# Termux-Vision: On-Device Computer Vision & Multimodal VLM Framework

[![PyPI](https://img.shields.io/pypi/v/termux-vision.svg?style=flat-square&color=0369a1)](https://pypi.org/project/termux-vision/)
[![Python](https://img.shields.io/pypi/pyversions/termux-vision.svg?style=flat-square)](https://pypi.org/project/termux-vision/)
[![npm](https://img.shields.io/npm/v/termux-vision.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/termux-vision)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/termux-vision)

> **Native On-Device Computer Vision & Multimodal Vision-Language Model (VLM) Runtime for Android Termux via Direct Bionic libc & Vulkan Compute Acceleration.**  
> *Zero PRoot. Zero Virtualization. 100% Native ARMv8.2-A NEON SIMD & Hardware GPU Offloading.*

---

## 1. Overview & Key Capabilities

`termux-vision` is an enterprise-grade, on-device multimodal vision inference and spatial computing framework engineered specifically for mobile Android devices. Operating directly against Android's native Bionic libc ABI and host Vulkan compute drivers, `termux-vision` eliminates heavyweight desktop dependencies (OpenCV, TorchVision) and enables high-throughput visual question answering, OCR image captioning, and classical feature extraction directly on edge hardware.

* **100% Vulkan GPU Compute Canny (`0.23 ms`)**: Chains 3-pass SPIR-V compute shaders entirely within VRAM using `vkCmdPipelineBarrier`, achieving 834x acceleration over Python without CPU memory roundtrips.
* **Ultra-Fast ARM64 NEON C++ Kernel (`3.02 ms`)**: Permanently eliminates trigonometric `atan2f` via tangent ratio bit quantization and 1-byte direction buffers.
* **Prebuilt-Asset-First Idempotent Installer (`0.005s Skip`)**: Automatically provisions verified precompiled ARM64 native binaries in 2 seconds from official releases, guaranteeing zero-build instant skip if assets already exist.
* **5-Backend Unified CLI Standard**: Enforces `['auto', 'gpu', 'vulkan', 'opencl', 'cpu']` and convenience flags (`--gpu`, `--cpu`, `--opencl`) across all subcommands.
* **Zero-Deception Fail-Fast Gatekeeper**: Strictly rejects defective text-only binaries lacking `--mmproj` (`E015`) and corrupted weights (`E014`).
* **Full-Layer GPU Offloading (-ngl 99)**: Dispatches all 99 transformer layers and cross-attention vision projections directly to device GPU VRAM (**0.00 MiB CPU mapped VRAM**).

---

## 2. Installation Guide

### 2.1 Python Package Installation (PyPI)
```bash
pip install --upgrade pip setuptools wheel
pip install termux-vision
```

### 2.2 Prebuilt Native Engine Provisioning (Idempotent 0.005s)
```bash
# Automatically download & unpack verified ARM64 prebuilt assets
termux-vision install
```

### 2.3 Direct GitHub Releases Wheel Asset
```bash
pip install https://github.com/uno-km/termux-vision/releases/download/v1.6.0/termux_vision-1.6.0-py3-none-any.whl
```

### 2.4 One-Line Bootstrap Installer
```bash
curl -sL https://raw.githubusercontent.com/uno-km/termux-vision/main/install.sh | bash
```

---

## 3. GPU Hardware Acceleration (`ameva-runtime`)

```bash
pip install termux-vision ameva-runtime termux-llamacpp
```

### Silicon Architecture Support Status
* **Qualcomm Adreno GPU (Snapdragon 8 Elite / Adreno 830, Adreno 7xx, Adreno 650)**: Production Verified & Supported (Full 25/25 layer GPU offloading, 15.00 tok/s on Moondream2 1.8B f16, SPIR-V JIT patch, KGSL watchdog defense via `GGML_VULKAN_SKIP_CHECKS="999999999"`).
* **ARM Mali GPU (Mali-G78, Mali-G68, etc.)**: Production Verified & Supported (Pure GPU offloading, 0.00 MiB CPU Mapped VRAM, MMVQ tuning via `--tune-mali`).
* **Samsung Xclipse GPU (Xclipse 920 / 940 - AMD RDNA)**: Under Active Engineering (In Progress / 개발 진행 중).

Run hardware diagnostics:
```bash
termux-vision doctor
```

---

## 4. Standardized CLI & Parameter Matrix

| Parameter | Alias | Default | Description |
| :--- | :--- | :--- | :--- |
| `-b, --backend` | `-d, --device` | `auto` | Compute acceleration backend: <code>auto</code>, <code>gpu</code>, <code>vulkan</code>, <code>opencl</code>, <code>cpu</code> |
| `--gpu` / `--cpu` / `--opencl` | *N/A* | *None* | Convenience shorthand flags for backend routing |
| `-i, --image` | `--image-path` | *Required* | Path to input image (`.png`, `.jpg`, `.webp`) |
| `-p, --prompt` | *N/A* | `"Describe this image"` | Multimodal text instruction query |
| `-m, --model` | *N/A* | `smolvlm-500m` | GGUF language model path or catalog identifier |
| `--mmproj` | *N/A* | *Auto-paired* | Vision projector GGUF model path (`mmproj-*.gguf`) |
| `-n, --max-tokens` | `--n-predict` | `150` | Maximum number of generated tokens |
| `-c, --ctx-size` | `--ctx` | `2048` | Context window size |
| `-t, --threads` | *N/A* | `auto` | Number of CPU execution threads |
| `-q, --quality` | *N/A* | `optimal` | 4-tier resolution preset: `fast` (384px), `optimal` (768px), `high` (1280px), `original` (1:1) |
| `--tune-mali` | *N/A* | `False` | Enable ARM Mali GPU MMVQ tuning (`GGML_VK_FORCE_MMVQ=1`) |
| `--json` | *N/A* | `False` | Emit machine-readable JSON benchmark telemetry |

### CLI Example
```bash
# 100% Vulkan GPU Canny Edge Detection (0.23 ms)
termux-vision canny photo.jpg -o edges.png --gpu --low 40 --high 120

# GPU-Accelerated Multimodal VLM Inference
termux-vision vlm photo.jpg -d gpu --tune-mali -p "What objects are visible?"
```

---

## 5. Python SDK Quickstart

```python
import termux_vision as tv

# 1. 100% Vulkan GPU Canny Edge Detection (0.23ms) or NEON CPU (3.02ms)
image = tv.io.load_image("document.jpg")
grayscale = tv.transforms.to_grayscale(image)
edges = tv.cv.canny(grayscale, low_threshold=40, high_threshold=120, backend="auto")
tv.io.save_image(edges, "edges.png")

# 2. On-Device Multimodal VLM Inference
with tv.vlm.load("smolvlm-500m", device="gpu") as engine:
    result = engine.describe("document.jpg", prompt="Summarize this document.", quality="optimal")
    print(f"TPS: {result.metrics.tokens_per_second:.2f} tok/s | Output: {result.text}")
```

---

## 6. Real-World Benchmarks & Hardware Scorecard

| Target Device | SoC & GPU | Algorithm / Model | Mode | Execution / Generation | VRAM Overhead | Status / Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Samsung Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | **100% Vulkan GPU Canny** | **GPU (VRAM Chain)** | **0.23 ms (Min 0.18 ms)** | **0.00 MiB CPU VRAM** | **834x Speedup** |
| **Samsung Galaxy S20** | Snapdragon 865 / Kryo 585 | **ARM64 NEON C++ Canny** | **CPU (NEON SIMD)** | **3.02 ms** | **1.0 MB Buffer** | **Zero atan2f** |
| **Samsung Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | **Moondream2 1.8B f16** | **GPU (Vulkan 25/25)** | **15.00 tok/s** | **2,706.00 MiB VRAM** | **Verified (Full GPU)** |
| **Samsung Galaxy S21 5G** | Exynos 2100 / Mali-G78 | SmolVLM-500M-Instruct | **GPU (Vulkan)** | **12.65 tok/s** | **1,059.02 MiB VRAM** | **+58.9% vs CPU** |
| Samsung Galaxy S21 5G | Exynos 2100 / 8-Core CPU | SmolVLM-500M-Instruct | CPU (NEON) | 7.96 tok/s | 0.00 MiB GPU VRAM | Baseline |
| **Samsung Galaxy A35 5G** | Exynos 1380 / Mali-G68 | SmolVLM-500M-Instruct | **GPU (Vulkan)** | **5.47 tok/s** | **1,059.02 MiB VRAM** | **+55.8% vs CPU** |
| Samsung Galaxy A35 5G | Exynos 1380 / 8-Core CPU | SmolVLM-500M-Instruct | CPU (NEON) | 3.51 tok/s | 0.00 MiB GPU VRAM | Baseline |

---

## 7. License
Licensed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)) & AOSF.
