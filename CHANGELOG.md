# Changelog

All notable changes to `termux-vision` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-10-01

### Added
- **UltraFace SSD ONNX Runtime Upgrade**: Replaced legacy Haar cascades with real-time neural face detection models (`version-RFB-320` and `version-slim-320`), supporting multi-face tracking and tuned bounding boxes.
- **POSIX ELF Binary Integrity Verification**: Enforced deterministic 4-byte `\x7fELF` magic byte header audit on all compiled runtime binaries.
- **VLM Namespace Isolation (`termux-vlm-cli`)**: Created dedicated binary symlink namespace to prevent runtime conflicts with system-level `llama-cli`.
- **Shared Model Cache Fallback**: Integrated centralized `~/.cache/termux-ai/models` canonical lookup, reducing cross-runtime disk duplication.
- **3-Tier CLI Mode Governance**: Implemented strict CLI handling with `--force`, `--dedicate`, and skip-if-healthy instant (<0.002s) bypass.

## [1.5.0] - 2026-10-01

### Added
- **100% Native Vulkan GPU Canny Edge Pipeline**: 3-Pass SPIR-V compute shader chain with 0.23ms latency on Snapdragon 8 Elite (Adreno 830).
- **Trigonometric-Free ARM64 NEON Canny Engine**: Eliminated `atan2f` transcendent computations via rational tangent ratio quantization.
- **Prebuilt Asset-First Idempotent Installer**: Instant skip when verified ARM64 prebuilt assets reside in `$PREFIX/lib`.
- **Ecosystem 5-Backend Standardization**: Standardized compute routing across `auto`, `gpu`, `vulkan`, `opencl`, and `cpu`.

## [1.4.5] - 2026-09-18

### Fixed
- **Multimodal Chat Template Collision Resolution**: Stripped conflicting `--chat-template` arguments during multimodal inference, preventing `Failed to tokenize prompt` and marker-count mismatches with GGUF vision models.
- **Vulkan Device Flag Sanitization**: Purged invalid `--device vulkan` argument from standalone execution paths, ensuring clean device selection and compatibility with modern llama.cpp runtime builds.

## [1.4.4] - 2026-09-18

### Fixed
- Enforced `--single-turn` and `--no-conversation` in `SubprocessVLMRuntime` to prevent interactive shell hangs.
- Streamlined prompt formatting to let `llama-cli` apply model-native Jinja chat templates cleanly.

## [1.4.3] - 2026-09-18

### Added
- Dedicated `--model-path` CLI option and SDK parameter (`model_path` in Python, `modelPath` in Node.js) for explicit custom GGUF models.
- Strict mutually exclusive parameter validation between model ID and model path.

### Changed & Fixed
- Eradicated all legacy 1024 context limits; established universal 2048 token baseline across custom discovery and catalog models.
- Integrated `termux-llamacpp` official runtime execution environment (`prepare_env`) into `ZeroFlickerEngine` and `SubprocessVLMRuntime`.
- Updated `RuntimeNotFoundError` to explicitly guide installation via `pip install termux-llamacpp` and `npm install -g termux-llamacpp`.
- Synchronized package manifests across PyPI and npm to `1.4.3`.

## [1.4.2] - 2026-09-18

### Changed & Hardened
- **Zero-Hardcoding Dynamic Latest-First Provisioning Architecture**:
  - Completely purged hardcoded fallback version strings (`1.4.0`) from `termux_vision/installer.py` and `install.sh`.
  - Implemented dynamic package version resolution (`_resolve_package_version`) and GitHub API release querying.
  - Prioritized invariant `releases/latest/download/` canonical endpoints for both release wheels and native engine tarballs.
  - Upgraded PyPI installation command to use `--upgrade` flag.
  - Synchronized versions across `package.json`, `pyproject.toml`, and `termux_vision/__init__.py` to `1.4.2`.

## [1.4.0] - 2026-09-07

### Added
- **Qualcomm Snapdragon 8 Elite (Adreno 830) Full-GPU VLM Acceleration**:
  - Validated 25/25 layer full GPU offloading on Galaxy S25 Adreno 830 (15.00 tokens/s generation, Moondream2 1.8B f16, 2,706 MiB VRAM).
  - Qualcomm SPIR-V JIT shader compiler patch: `mul_mat_vec_max_cols = 2` (fixed register spilling and `VK_ERROR_UNKNOWN` aborts).
  - Android kernel GPU watchdog (`kgsl`) timeout (`ErrorDeviceLost`) defense: `GGML_VULKAN_SKIP_CHECKS="999999999"` injection and micro-batch prefill chunking (`-b 64 -ub 64`).
  - Real-world empirical visual QA validation on photorealistic diffusion and geometric test scenes.
- **Dynamic Parameter & Execution Un-Clamping**:
  - `ZeroFlickerEngine` thread and batch parameter clamping removed, forwarding user options directly to `VisionAdapter`.
- **Installer Hardening**:
  - `install.sh` wheel installation unified to prevent legacy PyPI package overwriting.

---

## [1.3.1] - 2026-09-07

### Added
- Official GitHub Release v1.3.1 asset distribution with verified SHA-256 checksums.
- Dynamic shields badge integration for PyPI and npm.

### Changed
- Bumped version to 1.3.1 across pyproject.toml, package.json, installer.py, install.sh, and documentation portals.

## [1.3.0] - 2026-09-07

### Added
- **Universal Parameter Suite & Sibling Alignment**: Full parameterization (`-d/--device`, `-b/--backend`, `-i`, `-p`, `-m`, `--mmproj`, `-n`, `-c`, `-t`, `-W`, `-H`, `--image-size`, `-q`, `--tune-mali`, `--json`) matching official ecosystem CLI standards.
- **ARM Mali GPU Acceleration (Production Verified)**: Full 99-layer Vulkan SPIR-V offloading on ARM Mali-G78 (Galaxy S21) and Mali-G68 (Galaxy A35), achieving pure GPU execution with 0.00 MiB CPU Mapped VRAM.
- **Mali MMVQ Optimization Kernel**: `--tune-mali` flag enabling `GGML_VK_FORCE_MMVQ=1` for Mali TBDR tile-cache alignment.
- **Real-Device Benchmarks**: Galaxy S21 (12.65 tok/s, +58.9% over CPU baseline) and Galaxy A35 (5.47 tok/s, +55.8% over CPU baseline).
- **Dynamic Installer & Asset Provisioner**: `install.sh` and `termux_vision/installer.py` resolving dynamic GitHub Releases candidate endpoints with SHA-256 integrity verification.
- **Architecture Roadmap**: Qualcomm Adreno and Samsung Xclipse acceleration targets defined as Under Active Development.

---

## [1.2.0] - 2026-09-07

### Added
- Direct integration with `VisionAdapter` from `ameva_runtime.adapters` SSOT.
- Enforced strict E001 Fail-Fast on explicit Vulkan/GPU acceleration requests.
- Full English localization of diagnostic logs, Canny edge detection bridge, and VLM loaders.

---

## [1.1.4] - 2026-09-05

### Changed
- Resolved Node.js doctor `@ameva/runtime` module require preventing runtime lookup errors.
- Aligned backend docstrings, install script dependencies, and subprocess environment bindings.

---

## [1.1.3] - 2026-09-05

### Changed
- Migrated hardware acceleration dependency to unified `ameva-runtime>=2.0.0` and `@ameva/runtime>=2.0.0`.
- Fixed Doctor Vulkan driver discovery keys (`loader_detected`, `driver_file_detected`).
- Standardized VLM and OpenCV Canny acceleration bridge.

---

## [1.1.1] - 2026-09-02

### Added
- **ConvBlock Hardening**: Added explicit warnings.warn for uncalibrated random weights and is_trained property.
- **Multimodal Pipeline**: Depthwise separable convolution layers for memory-constrained edge devices.

### Cleaned
- Purged legacy 1.0.0.tgz tarball, .egg-info, and orphan .pyc files from source tree.

### Verification
- **Unit Tests**: 70 / 70 passed with 100% assertion coverage.
