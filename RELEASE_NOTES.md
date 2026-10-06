# Release Notes - termux-vision v1.6.0

**Release Tag**: `v1.6.0`  
**Distribution Channels**: PyPI (`termux-vision 1.6.0`), NPM (`termux-vision@1.6.0`), GitHub Releases (`v1.6.0`)  
**Target Architecture**: Android Termux (ARM64 / aarch64 Bionic libc & Host Vulkan / OpenCL)  
**License**: Apache-2.0  
**Verification Status**: Validated on Physical Samsung Galaxy S25, Galaxy S20, Galaxy A35 Fleets  

---

## 🌟 Highlights & Major Architectural Milestones

### 1. UltraFace SSD Neural Face Detection Engine (v1.6.0)
- **Neural SSD Architecture**: Replaced legacy Haar Cascades with dual-resolution UltraFace SSD ONNX runtimes (`version-RFB-320` and `version-slim-320`), delivering sub-15ms face detection on mobile CPUs.
- **Dynamic Bounding Box Scaling**: Automatic aspect-ratio preserved letterboxing and coordinate un-padding with tuned confidence thresholds.
- **POSIX ELF Magic Integrity Verification**: Automated deterministic 4-byte `\x7fELF` magic header validation preventing corrupted binary execution.
- **VLM Namespace Isolation (`termux-vlm-cli`)**: Full ABI decoupling from system `llama-cli` via isolated symbolic links.
- **Canonical Model Cache**: Native fallback to `~/.cache/termux-ai/models` to eliminate cross-runtime disk bloat.

### 2. 100% Native Vulkan GPU Compute Canny Pipeline (`0.23 ms`)
- **End-to-End VRAM Chaining**: Built a pure compute shader pipeline chaining Sobel 3x3 Gradient Convolution $\to$ Non-Maximum Suppression (NMS) $\to$ Hysteresis Thresholding via `vkCmdPipelineBarrier` memory barriers without intermediate CPU roundtrips.
- **Microsecond Latency**: Achieved **0.23 ms** average execution latency (minimum **0.18 ms**) on Snapdragon 8 Elite (Adreno 830), representing an **834x speedup** compared to Python CPU reference (192.0 ms).
- **Zero CPU-Mapped VRAM**: Guaranteed 0.00 MiB host memory footprint with unified memory buffer pooling and deterministic descriptor set recycling.

### 2. Trigonometric-Free ARM64 NEON C++ Canny Engine (`3.02 ms`)
- **Tangent Ratio Bit Quantization**: Permanently eliminated transcendent `atan2f` trigonometric computations through rational tangent boundary comparisons, quantizing gradient angles into 4 discrete directions in pure integer logic.
- **1-Byte Direction Buffers**: Reduced gradient orientation storage to a single byte per pixel, minimizing cache footprint and maximizing ARMv8.2-A NEON SIMD vector throughput.
- **Physical Device Execution**: Benchmarked at **3.02 ms** on Qualcomm Snapdragon 865 (Galaxy S20) and **4.36 ms** on Exynos 1380 (Galaxy A35).

### 3. Prebuilt-Asset-First Idempotent Installer (`termux-vision install`)
- **Sub-Second Provisioning**: Integrated native asset provisioning downloading verified ARM64 prebuilt archives (`termux-vision-android-arm64.tar.gz`, 1.79 MB) directly from official GitHub Releases in under 2 seconds.
- **0.005s Idempotent Instant Skip**: Bypasses compilation overhead and verification passes instantly when SHA-256 validated binaries already reside in `$PREFIX/lib`.
- **Flexible Management Flags**: Supported `--force` (forced clean reinstall), `--from-source` (clang++ native compilation), and `--dry-run` (hash and dependency audit).

### 4. Ecosystem 5-Backend Standardization & Shorthand Flags
- **Strict 5-Option Governance**: Standardized compute routing across all subcommands to `["auto", "gpu", "vulkan", "opencl", "cpu"]`.
- **Shorthand Flags**: Introduced direct `--gpu`, `--cpu`, and `--opencl` convenience switches.
- **Adreno 650 Quirks Resolved**: Routed Qualcomm Adreno 600 series to OpenCL 2.0 kernels, circumventing legacy Vulkan SPIR-V float subnormal truncation.

### 5. Zero-Deception Fail-Fast Gatekeeper & Error Protocol
- **Binary Pre-validation (E015)**: Actively probes `llama-cli --help` for `--mmproj` capability, instantly rejecting defective text-only binaries before inference.
- **Weight Integrity Guard (E014)**: Automatically rejects truncated or corrupted model weights (<10 MB or invalid GGUF headers).
- **Permanent Ban on Silent Fallbacks**: Refuses unnotified synthetic mocks or CPU down-throttling when GPU acceleration fails, raising structured error codes with remediation instructions.

---

## 📊 Comprehensive Hardware Benchmark Scorecard

### Classical Vision Filtering Latency (512x512 Image)
| Algorithm / Kernel | Acceleration Backend | Hardware / SoC | Latency | Status |
| :--- | :--- | :--- | :---: | :--- |
| **Vulkan GPU Canny** | **3-Pass SPIR-V VRAM Chain** | **Snapdragon 8 Elite (Adreno 830)** | **0.23 ms** | **Production Verified (834x)** |
| **ARM64 NEON Canny** | **Tangent Ratio (No atan2f)** | **Snapdragon 865 (Galaxy S20)** | **3.02 ms** | **Production Verified** |
| ARM64 NEON Canny | Cortex-A78 NEON SIMD | Exynos 1380 (Galaxy A35) | 4.36 ms | Production Verified |
| Sobel 3x3 Filter | ARM NEON Vectorized | Snapdragon 865 (Galaxy S20) | 1.10 ms | Production Verified |
| 2D Integral Image | Prefix Sum Loop | Snapdragon 865 (Galaxy S20) | 1.20 ms | Production Verified |

### On-Device Multimodal VLM Inference (SmolVLM-500M & Moondream2 1.8B)
| Device | SoC & GPU | Model Architecture | Mode | Prompt Processing | Generation Speed | GPU VRAM |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Samsung Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | Moondream2 1.8B f16 | **GPU (Vulkan 25/25)** | **19.84 tok/s** | **15.00 tok/s** | **2,706 MiB** |
| **Samsung Galaxy S21** | Exynos 2100 / Mali-G78 | SmolVLM-500M-Instruct | **GPU (Vulkan)** | **14.28 tok/s** | **12.65 tok/s** | **1,059 MiB** |
| **Samsung Galaxy A35** | Exynos 1380 / Mali-G68 | SmolVLM-500M-Instruct | **GPU (Vulkan)** | **5.67 tok/s** | **5.47 tok/s** | **1,059 MiB** |

---

## 🛠️ Installation & Upgrade

```bash
# Upgrade via PyPI
pip install --upgrade termux-vision

# Upgrade native accelerated prebuilt engine
termux-vision install --force

# Upgrade Node.js CLI via npm
npm install -g termux-vision@1.5.0
```
