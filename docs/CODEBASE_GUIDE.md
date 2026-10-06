# termux-vision 전체 소스코드 분석 및 종합 기술 전수 매뉴얼

본 문서는 **termux-vision** 프레임워크의 시스템 아키텍처, 설치 및 빌드 메커니즘, 모듈별 소스코드 구현 원리, 실행 흐름 및 SDK 사용법을 체계적으로 기술한 엔지니어링 전수 문서입니다.

---

## 1. 개요 및 시스템 아키텍처 (System Architecture)

`termux-vision`은 Android Termux 및 ARM64 환경에 최적화된 온디바이스 컴퓨터 비전(Classical CV) 및 대규모 시각 언어 모델(VLM, Vision-Language Model) 멀티모달 추론 프레임워크입니다.

### 1.1 핵심 설계 원칙
1. **Zero PyTorch Dependency**: 고비용 대규모 프레임워크인 PyTorch 없이, 순수 NumPy, C, C++, ctypes, Node.js 내장 모듈만으로 모든 고전 비전 연산 및 텐서 직렬화를 구현합니다.
2. **Dual-Engine Architecture (Python & Node.js/TypeScript)**: 동일한 기능 명세와 CLI 인터페이스를 Python 3.8+ 및 Node.js 16+ 양대 런타임에서 독립적이면서도 동일하게 제공합니다.
3. **Hardware Acceleration & Graceful Fallback**: 
   - 1차: Vulkan GPU 및 컴파일된 네이티브 C 엔진 (`libfast_cv.so`, `libfast_cv_vk.so`)을 통한 고속 가속.
   - 2차: 네이티브 런타임 부재 또는 오류 발생 시 순수 NumPy/JS 알고리즘으로 무중단 자동 폴백(Fallback).
4. **Strict Memory Admission & LowMemoryKiller(LMK) Defense**: `/proc/meminfo`를 기반으로 가용 RAM을 실시간 검사하며, 사용자 정책(`warn`, `strict`, `unrestricted`)에 따른 메모리 승인 제어 및 POSIX 프로세스 그룹(`start_new_session=True`) 격리를 수행합니다.
5. **Universal Model Hub & Custom Model Freedom**: 공식 카탈로그 프리셋(`smolvlm-500m-q4`, `qwen2-vl-2b-q4`)뿐만 아니라, 임의의 Hugging Face GGUF (`hf:owner/repo:file.gguf`) 및 로컬 임의 파일(`--model model.gguf --mmproj mmproj.gguf`)을 완벽 지원합니다.

```
[termux-vision Dual-Engine Architecture]

+------------------------------------------------------------------------------------+
|                                    CLI Layer                                       |
|            Python CLI (`termux-vision`)   |   Node.js CLI (`npx termux-vision`)    |
+------------------------------------------------------------------------------------+
|                                 Public API Layer                                   |
|   Python SDK (`termux_vision.vlm.load`)   |   Node.js SDK (`require('termux-vision')`)|
+------------------------------------------------------------------------------------+
|                                   Core Engines                                     |
|  +-----------------------------+  +---------------------------------------------+  |
|  |     Classical CV Engine     |  |                 VLM Engine                  |  |
|  | - Sobel / Canny (5-stage)   |  | - Supervised llama-cli Subprocess Runner    |  |
|  | - Integral Image / Box Sum  |  | - Model Cache Manager & SHA-256 Verifier    |  |
|  | - Morphology (Dilate/Erode) |  | - RAM Admission Controller (/proc/meminfo)  |  |
|  | - Haar Cascade Face Detect  |  | - Adapters (SmolVLM, Qwen2-VL)              |  |
|  | - MobileNet / YOLO Nano /ViT|  | - TVRP Framing Protocol & Supervisor        |  |
|  +-----------------------------+  +---------------------------------------------+  |
+------------------------------------------------------------------------------------+
|                                Acceleration Layer                                  |
|   C Native (`libfast_cv.so`)   |   Vulkan Native (`libfast_cv_vk.so`) | Pure NumPy |
+------------------------------------------------------------------------------------+
|                              Hardware & OS Target                                  |
|                 Android Termux (Linux ARM64 / bionic / POSIX)                      |
+------------------------------------------------------------------------------------+
```

---

## 2. 설치 및 빌드 시스템 (`install.sh`, `pyproject.toml`, `package.json`)

### 2.1 원터치 무결점 설치 스크립트 (`install.sh`)
`install.sh`는 안드로이드 Termux 환경과 일반 리눅스 환경을 자동으로 판별하여 전체 툴체인을 구성합니다.

```bash
# 전체 설치 흐름:
# 1. 패키지 매니저 감지 (Termux pkg vs Debian/Ubuntu apt-get)
# 2. 필수 바이너리 빌드 도구 설치 (clang, git, python, numpy, libjpeg-turbo, nodejs 등)
# 3. ameva-runtime 및 llama-cli(termux-llamacpp) 프로비저닝
# 4. 네이티브 C & Vulkan 가속 라이브러리 컴파일
#    - clang -O3 -shared -fPIC -o termux_vision/csrc/libfast_cv.so termux_vision/csrc/fast_cv.c -lm
#    - clang++ -O3 -shared -fPIC -o termux_vision/csrc/libfast_cv_vk.so termux_vision/csrc/vk_cv_engine.cpp
# 5. Python SDK 빌드 격리 우회 설치 (pip install --no-build-isolation -e .)
# 6. Node.js 글로벌 CLI 심볼릭 링크 연결 (npm install -g .)
# 7. 대화형 기본 VLM 모델(smolvlm-500m-q4) 다운로드 안내
```

### 2.2 패키지 명세
- **`pyproject.toml`**: `setuptools>=61.0` 기반, 의존성 `numpy>=1.20.0`, `Pillow>=8.0.0`, `ameva-runtime>=2.0.0`, 선택적 의존성 `termux-train>=0.1.0`. 진입점 CLI: `termux-vision = termux_vision.cli.main:main`.
- **`package.json`**: `version: 1.1.3`, 바이너리 실행 파일 `termux-vision`, `tv` (`bin/cli.js`), TypeScript 타입 정의 `index.d.ts`.

---

## 3. Python 서브모듈별 소스코드 전수 분석

### 3.1 예외 및 에러 가이드 시스템 (`termux_vision/errors.py`)
구조화된 예외 계층을 정의하며, 단순 에러 메시지 출력에 그치지 않고 구체적인 해결 가이드라인(Actionable Recommendation)을 제공합니다.

- `TermuxVisionError`: 프레임워크의 최상위 기본 예외.
- `ImageDecodeError` / `ImageEncodeError`: 이미지 파싱 실패 시 지원 포맷(JPEG, PNG, BMP, PPM, WebP) 안내.
- `DecompressionBombError`: 악의적이거나 비정상적인 거대 픽셀 할당 공격 차단.
- `InsufficientMemoryError`: 요청된 VLM 동작 메모리 대비 가용 RAM 부족 시 발생.
- `ModelNotFoundError` / `NoInstalledModelsError` / `ModelSelectionRequiredError` / `ModelArtifactsMissingError`: 로컬 캐시 모델의 누락, 불완전성, 다중 설치 시 지정 안내.
- `RuntimeNotFoundError`: `llama-cli` 바이너리 미발견 시 설치 경로 및 `pkg install termux-llamacpp` 안내.
- `VulkanNotAvailableError`: Vulkan GPU 하드웨어/드라이버 부재 시 명시적 CPU 모드 전환(`--device cpu`) 안내.
- `CameraPermissionError` / `TermuxAPIUnavailableError`: Android Termux API 카메라 권한 누락 감지.

---

### 3.2 입출력 서브모듈 (`termux_vision/io/`)

#### 1) `limits.py` (보안 및 리소스 한계 검증)
- `ImageLimits` 데이터클래스를 통해 최대 픽셀 수(기본 16 메가픽셀, 4000x4000), 단일 축 최대 길이(8192px), 최대 파일 크기(32MB), 애니메이션(GIF/APNG) 허용 여부를 강제 검증합니다.

#### 2) `loader.py` (안전한 이미지 로더 및 세이버)
- `load_image(image_path, target_size=None, limits=DEFAULT_IMAGE_LIMITS, apply_exif_orientation=True) -> np.ndarray`:
  - 파일 크기 및 차원 제한 검증.
  - Pillow의 `ImageOps.exif_transpose`를 통한 스마트 EXIF 회전각 자동 보정.
  - RGB 포맷 강제 변환 후 C-contiguous 메모리 레이아웃(`np.ascontiguousarray`)의 `(H, W, 3)` uint8 ndarray 반환.
- `save_image(image, save_path, quality=95, metadata="strip")`:
  - ndarray의 차원(L, RGB, RGBA) 판별 및 float 정규화 배열의 자동 0~255 클리핑.
  - `metadata="strip"` 옵션으로 사진에 포함된 민감한 GPS 및 기기 EXIF 태그를 원천 제거하여 보안 저장.
- `get_image_info(image_path) -> Dict[str, Any]`:
  - 메모리에 전체 픽셀 버퍼를 로드하지 않고 헤더만 파싱하여 width, height, format, mode, size_mb를 고속 반환.

#### 3) `camera.py` (안드로이드 카메라 연동)
- `CameraCapture(camera_id=0)`:
  - `termux-camera-photo -c <id> <tmp_path>` 서브프로세스를 호출하여 임시 파일로 프레임을 캡처한 뒤 `load_image`를 거쳐 uint8 ndarray로 변환하고 임시 파일을 안전하게 소멸시킵니다.

#### 4) `safetensors.py` (PyTorch 없는 순수 SafeTensors 직렬화)
- `save_safetensors(tensors_dict, file_path)`:
  - NumPy 데이터 타입을 SafeTensors 규격(`F32`, `F64`, `I32`, `I64`, `U8`)으로 매핑.
  - 8바이트 Little-Endian 정수 헤더 길이 + JSON 메타데이터 + 연속 바이너리 버퍼 구조로 디스크에 직접 기록.
- `load_safetensors(file_path) -> dict`:
  - 8바이트 헤더 길이를 읽고 JSON 오프셋 정보를 파싱하여 `np.frombuffer`로 zero-copy에 준하는 고속 텐서 복원.

---

### 3.3 변환 및 전처리 서브모듈 (`termux_vision/transforms/`)

#### 1) `functional.py` (고속 함수형 연산)
- `to_grayscale(image)`: 표준 휘도 가중치 벡터 내적 (`0.2989*R + 0.5870*G + 0.1140*B`).
- `to_rgb(image)`: Grayscale(2D/1-ch) 또는 RGBA(4-ch)를 3-ch RGB로 변환.
- `to_bgr(image)`: OpenCV 호환 BGR 채널 역순 변환.
- `resize(image, size, interpolation="bilinear")`: PIL C-엔진 가속 기반 리사이징.
- `crop(image, x, y, w, h)` / `center_crop(image, size)`: 경계면 오버플로우 안전 클리핑 크롭.
- `normalize(image, mean, std)`: float32 변환 및 표준 정규화 연산.

#### 2) `scale.py` (종횡비 보존 스마트 스케일링)
- `ImageQuality` 열거형 및 품질 프리셋:
  - `ORIGINAL`: 원본 1:1 해상도 유지.
  - `HIGH`: 최대 축 1280px.
  - `OPTIMAL`: 최대 축 768px (VLM 추론 권장 밸런스).
  - `FAST`: 최대 축 384px (고속 처리).
- `resolve_target_dimensions(orig_w, orig_h, mode, max_dim)`: 가로/세로 비율을 완벽히 유지하며 다운스케일링 크기를 계산.
- `prepare_image_for_inference(image, quality, max_dim)`: VLM 추론 직전 임시 최적 해상도 이미지 생성 및 파일 경로 반환.

#### 3) `compose.py` (파이프라인 결합)
- `Compose([Resize(...), CenterCrop(...), Normalize(...), ToChannelFirst()])` 형태의 전처리 체인 구성.

---

### 3.4 고전 컴퓨터 비전 및 네이티브 가속 서브모듈 (`termux_vision/cv/` & `csrc/`)

#### 1) 네이티브 C 엔진 (`csrc/fast_cv.c`)
- `compute_integral_c(src, dst, w, h)`: $O(W \times H)$ 단일 패스로 2차원 적분 영상(Summed Area Table) 계산.
- `sobel_c(src, mag, angle_deg, w, h)`: 3x3 수평/수직 소벨 마스크 합성, 그래디언트 강도($\sqrt{G_x^2 + G_y^2}$) 및 각도($\text{atan2}(G_y, G_x)$) 연산.
- `canny_nms_threshold_c(...)`:
  1. Gradient Direction 기반 비최대 억제(NMS, Non-Maximum Suppression).
  2. Double Thresholding (강한 에지 255, 약한 에지 후보 75).
  3. Hysteresis Edge Tracking (강한 에지와 8-이웃으로 연결된 약한 에지만 보존).
- `morphology_c(src, dst, w, h, is_dilate)`: 3x3 팽창(Dilation, 최댓값 필터) 및 침식(Erosion, 최솟값 필터).
- `haar_detect_multiscale_c(...)`: 24x24 기준 윈도우 기반 3단계 캐스케이드(눈 영역, 코/볼 명암비, 입 영역)를 C 언어 레벨에서 적분 영상의 $O(1)$ 박스 섬으로 스캔하여 실시간 얼굴 후보군 추출.

#### 2) C++ 엔진 (`csrc/vk_cv_engine.cpp`)
- 쓰레드 안전성을 위한 `std::mutex` 및 프레임별 `malloc` 힙 단편화를 방지하는 `thread_local` 스크래치 버퍼 풀 관리.
- 고속 병렬 Canny 및 Bilinear 보간 스케일러 구현.

#### 3) 백엔드 브릿지 (`csrc/backend.py`)
- `libfast_cv.so`, `libfast_cv_vk.so` 라이브러리를 동적으로 검색 및 ctypes 매핑.
- 네이티브 바이너리 부재 시 순수 파이썬 구현체로 자동 폴백.

#### 4) CV 모듈 (`cv/filters.py`, `cv/contours.py`, `cv/morphology.py`, `cv/integral.py`)
- `canny()`: C 백엔드 $\rightarrow$ 순수 NumPy 5단계 파이프라인.
- `find_contours()`: 이진 영상에서 BFS(너비 우선 탐색) 기반 8-방향 연결 컴포넌트 추적 및 바운딩 박스/면적/무게중심 계산.
- `color_histogram()`: 1D/3D 색상 채널별 정규화 히스토그램 산출.

---

### 3.5 객체 검출 서브모듈 (`termux_vision/detect/`)

- `types.py`:
  - `BoundingBox(left, top, right, bottom)`: 불변(frozen) 축 정렬 2D 박스. `width`, `height`, `area`, `to_xywh()`, `from_xywh()` 제공.
  - `Detection(bbox, score, class_id, class_name, metadata)`.
- `nms.py`:
  - `box_iou(box1, box2)`: IoU(Intersection over Union) 교집합/합집합 면적비 계산.
  - `non_maximum_suppression(boxes, scores, iou_threshold=0.45, score_threshold=0.25)`: 분모 0 방어 코드가 적용된 고속 NMS 인덱스 필터링.
- `haar.py`:
  - `HaarCascadeDetector`: 적분 영상을 활용한 3단계 휴리스틱 명암비 검사기.
  - `detect_faces(image, scale_factor=1.2, min_size=(32,32))`: 모바일 CPU 부하를 방지하기 위해 대형 이미지를 내부적으로 640px로 다운스케일링하여 고속 검출 후, 원본 좌표계로 스케일을 정밀 복원하여 반환.

---

### 3.6 신경망 모델 서브모듈 (`termux_vision/models/`)

- `embedding.py`:
  - `Embedding(values, model_id, dimension, normalized=True)`: L2 정규화 임베딩 벡터.
  - `compute_similarity(a, b)`: 동일 모델 공간(`model_id`) 및 동일 차원 검증 후 코사인 유사도(내적) 산출.
- `conv_block.py` (`DepthwiseSeparableConv2D`):
  - 표준 합성곱 대비 연산량을 대폭 절감하는 MobileNet 핵심 연산: Spatial Filtering (Depthwise) + 1x1 Linear Mixing (Pointwise) + ReLU6 활성화.
- `mobilenet.py` (`MobileNetV3FeatureExtractor`):
  - 4개 스테이지의 Depthwise Separable Conv + Global Average Pooling + L2 Normalization 구조의 특징 추출기.
- `vit_patch.py`:
  - `extract_patches(image, patch_size=16)`: 이미지를 $N \times (P^2 \cdot C)$ 크기의 1D 패치 시퀀스로 분할.
  - `reconstruct_from_patches(...)`: 패치 시퀀스를 2D 공간 이미지로 역재구성.
- `yolo_detector.py` & `yolo_post.py`:
  - `TinyYOLONanoDetector`: 경량 Mobile 백본과 $32 \times 32$ 그리드 헤드로 구성된 객체 검출기.
  - `YOLODecoder`: (cx, cy, w, h, class_scores...) 텐서를 디코딩하고 클래스별 NMS를 수행하여 바운딩 박스 목록 반환.

---

### 3.7 온디바이스 VLM 멀티모달 추론 서브모듈 (`termux_vision/vlm/`)

VLM 모듈은 `termux-vision`의 핵심 기능으로, 경량 언어 모델과 비전 프로젝터(mmproj)를 결합하여 단일 턴 질의응답 및 이미지 요약을 수행합니다.

```
[VLM Inference Pipeline Flow]

User Input (Image + Prompt)
           │
           ▼
[1. Memory Admission Check] ──> /proc/meminfo 검사 (RAM floor 부족 시 warn/strict 처리)
           │
           ▼
[2. Cache & Manifest Resolve] ──> ~/.cache/termux-vision/models/<model_id>/
           │                      (Language Model GGUF + Vision mmproj GGUF)
           ▼
[3. Image Preprocessing] ──> 종횡비 보존 다운스케일링 (optimal: 768px, fast: 384px)
           │
           ▼
[4. Adapter Prompt Formatting] ──> ChatML 포맷 변환 (<|im_start|>user\n<image>\n...)
           │
           ▼
[5. Isolated Subprocess Spawn] ──> llama-cli -m model.gguf --mmproj mmproj.gguf ...
           │                      (POSIX Session Group 분리, SIGTERM/SIGKILL 안전 감시)
           ▼
[6. Metrics & Text Parsing] ──> Latency, TPS(t/s), Hardware Diagnostic 파싱
           │
           ▼
VLMResult Object (text, metrics, warnings)
```

#### 1) `api.py` (`load`, `VLMContext`)
- `load(...)`:
  - 모델 캐시 조회, 메모리 어드미션 검증, 런타임(`llama-cli`) 바이너리 해석, 스레드 수 제어, 디바이스(CPU/Vulkan) 설정.
  - `VLMContext` 객체를 반환하며 `with load(...) as engine:` 형태의 컨텍스트 매니저 지원.
- `VLMContext.describe(image, prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, quality, max_dim, ...)`:
  - 엄격한 파라미터 경계값 검증 (0 이하 토큰, 음수 온도 등 즉시 거부).
  - 이미지 전처리 $\rightarrow$ 런타임 추론 $\rightarrow$ 결과 객체(`VLMResult`) 반환.
- `VLMContext.ask(image, question)`: 생성된 텍스트 문자열만 간결하게 반환.

#### 2) `cache.py` (`ModelCacheManager`, `CATALOG`)
- 공식 카탈로그 프리셋:
  - `smolvlm-500m-q4`: SmolVLM 500M Instruct GGUF + mmproj (~550MB, 메모리 요구량 ~750MB, Tier M).
  - `qwen2-vl-2b-q4`: Qwen2-VL 2B Instruct GGUF + vision encoder (~3.4GB, 메모리 요구량 ~2100MB, Tier L).
- 모델 상태 머신 (`ModelState`):
  - `ABSENT`, `DOWNLOADING`, `VERIFYING`, `READY`, `IN_USE`, `QUARANTINED`.
- 보안 파일 검증:
  - Path Traversal(`../`) 원천 차단.
  - 다운로드 시 `.partial` 임시 파일 격리 다운로드 후 원자적 치환(`os.replace`).
  - SHA-256 해시 스트리밍 무결성 검증 (`verify_integrity()`).

#### 3) `memory.py` (메모리 제어 및 LMK 방어)
- `/proc/meminfo`에서 `MemTotal`, `MemAvailable`을 실시간 파싱.
- `check_memory_admission(estimate, user_budget_mb, memory_policy)`:
  - 시스템 필수 보존 메모리($\approx 384\text{MB}$) + 추정 피크 메모리 + 불확실성 마진($15\%$)을 합산한 안전 임계값을 기준으로 검사.
  - `memory_policy="strict"`일 경우 예외 발생, `warn`일 경우 경고 로그 후 진행, `unrestricted`일 경우 제한 없이 진행.

#### 4) `runtime/subprocess.py` (`SubprocessVLMRuntime`)
- `llama-cli`를 단일 턴(`--single-turn`, `--simple-io`) 모드로 안전하게 실행.
- POSIX 환경에서 `start_new_session=True`를 설정하여 메인 프로세스와 자식 프로세스 그룹을 완벽 분리.
- 타임아웃 발생 시 프로세스 그룹 전체에 `SIGTERM` $\rightarrow$ grace period $\rightarrow$ `SIGKILL` 단계적 강제 종료.
- `stdout`/`stderr` 로그를 분석하여 로드 시간(`load_ms`), 비전 인코딩 시간(`vision_ms`), 디코딩 시간(`decode_ms`), 초당 토큰 수(`tokens_per_second`, $t/s$), GPU 하드웨어 정보 실시간 추출.
- Vulkan 모드 실패 시 `--fallback` 정책에 따라 CPU 모드로 자동 재시도.

#### 5) `adapters/` (모델별 프롬프트 어댑터)
- `smolvlm.py`: `<|im_start|>user\n<image>\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n`
- `qwen2vl.py`: `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_prompt}<|im_end|>\n<|im_start|>assistant\n`

---

### 3.8 훈련 브릿지 서브모듈 (`termux_vision/bridge/termux_train_bridge.py`)

- `to_termux_tensor(data, requires_grad=False)`:
  - NumPy ndarray를 `.tolist()` 변환 없이 직접 C-버퍼로 전달하여 **8배 메모리 팽창 및 LMK 킬을 원천 방지**.
  - `termux-train` 프레임워크와 즉시 연동되는 자동 미분 텐서로 변환.
- `VisionPatchDataset`:
  - ViT 및 경량 분류기 온디바이스 학습을 위한 무복사 배치 제너레이터.

---

### 3.9 CLI 및 진단 서브모듈 (`termux_vision/cli/`)

#### 1) `doctor.py` (시스템 및 런타임 진단)
- 기기 OS, Python 버전, CPU 코어 수, 총 RAM 및 가용 RAM 진단.
- Vulkan 로더(`/system/lib64/libvulkan.so`) 및 Adreno/Mali 하드웨어 드라이버 존재 여부 판별.
- `llama-cli` 설치 상태 및 로컬 캐시 모델 무결성(Full SHA-256) 검사.

#### 2) `main.py` (CLI 서브커맨드)
- `doctor`: 하드웨어 및 런타임 상태 진단 (`--json`, `--full`).
- `model`: 모델 관리 (`list`, `install <id>`, `download <url/hf>`, `remove <id>`).
- `info`: 이미지 메타데이터(크기, 포맷, 모드 등) 확인.
- `canny`: Canny 에지 검출 실행 및 저장 (`--low`, `--high`, `--resize`).
- `detect-face`: 얼굴 후보 영역 검출 및 크롭 저장 (`--json`).
- `vlm`: VLM 멀티모달 이미지 질의/요약 (`-m`, `-p`, `--quality`, `--device`, `-t`, `-n`, `--temp` 등).
- `benchmark`: 온디바이스 비전 처리 속도 벤치마크.

---

## 4. Node.js / TypeScript 듀얼 엔진 분석

Node.js 환경에서도 Python SDK와 100% 동일한 기능을 네이티브 JS로 제공합니다.

- **`index.js` / `index.d.ts`**: 전체 API 진입점 및 엄격한 TypeScript 타입 정의.
- **`bin/cli.js`**: `npx termux-vision` 또는 전역 `tv` 명령어로 실행 가능한 Node.js CLI.
- **`lib/vlm.js`**: `NodeVLMContext`, `resolveLlamaCli()`, `spawn` 기반의 비동기 스트림 제어 및 메트릭 파서.
- **`lib/cv.js`**: Float32Array/Uint8Array 기반 순수 JS 소벨 및 Canny 에지 검출.
- **`lib/detect.js`**: 순수 JS BoundingBox IoU 및 NMS 알고리즘.
- **`lib/doctor.js`**: `os`, `child_process` 기반 시스템 진단.
- **`lib/cache.js`**: Node.js 파일 시스템 기반 모델 캐시 및 Hugging Face 다운로더.

---

## 5. 실전 사용 가이드 (How-to-Run Manual)

### 5.1 CLI 실행 예시

#### 1) 환경 진단
```bash
# 기본 진단
termux-vision doctor

# Vulkan 드라이버 프로브 및 전체 모델 SHA-256 검사
termux-vision doctor --probe-vulkan --full --json
```

#### 2) VLM 모델 설치 및 관리
```bash
# 공식 권장 경량 모델(SmolVLM 500M) 설치 (~550MB)
termux-vision model install smolvlm-500m-q4

# 설치된 모델 목록 확인
termux-vision model list

# 임의의 Hugging Face 모델 직접 다운로드
termux-vision model download hf:owner/custom-repo:custom-model.gguf
```

#### 3) VLM 이미지 질의응답 및 요약
```bash
# 기본 한국어 요약 설명
termux-vision vlm sample.jpg

# 특정 질문 질의 (고속 프리셋, 4 스레드, CPU 모드)
termux-vision vlm sample.jpg -p "사진 속 텍스트와 주요 객체를 알려줘" --quality fast -t 4 --device cpu

# JSON 결과 출력 (메트릭 포함)
termux-vision vlm sample.jpg -p "Describe this scene" --json
```

#### 4) 고전 비전 (Canny 에지 및 얼굴 검출)
```bash
# Canny 에지 검출
termux-vision canny input.jpg -o edges.png --low 50 --high 150

# 얼굴 후보군 검출 및 대표 얼굴 자동 크롭
termux-vision detect-face photo.jpg -o face.jpg --json
```

---

### 5.2 Python SDK 사용 예시

```python
import termux_vision as tv

# 1. 이미지 로드 및 정보 확인
img = tv.io.load_image("sample.jpg")
info = tv.io.get_image_info("sample.jpg")
print(f"Image Size: {info['width']}x{info['height']}, Format: {info['format']}")

# 2. 고속 Canny 에지 검출
gray = tv.transforms.functional.to_grayscale(img)
edges = tv.cv.filters.canny(gray, low_threshold=40.0, high_threshold=120.0)
tv.io.save_image(edges, "edges_out.png")

# 3. 얼굴 후보군 검출
detections = tv.detect_faces(img)
for d in detections:
    print(f"Face candidate box: {d.bbox.to_xywh()}")

# 4. On-Device VLM 추론
with tv.vlm.load(model_id="smolvlm-500m-q4", device="auto", threads=4) as engine:
    result = engine.describe(
        image=img,
        prompt="이 사진에 보이는 주요 객체와 상황을 설명해줘.",
        max_tokens=150,
        quality="optimal"
    )
    print("VLM Result:", result.text)
    print("Metrics:", result.metrics)
```

---

### 5.3 Node.js / TypeScript SDK 사용 예시

```javascript
const tv = require('termux-vision');

async function run() {
  // 1. 시스템 닥터 진단
  const report = tv.doctor({ probeVulkan: true });
  console.log('System RAM:', report.hardware.total_ram_mb, 'MB');

  // 2. VLM 엔진 로드 및 추론
  const engine = await tv.load({
    modelId: 'smolvlm-500m-q4',
    device: 'cpu',
    threads: 4
  });

  const result = await engine.describe('sample.jpg', {
    prompt: 'What is inside this image?',
    maxTokens: 100
  });

  console.log('Description:', result.text);
  console.log('Latency:', result.metrics.decodeMs, 'ms');
  engine.close();
}

run().catch(console.error);
```

---

## 6. 결론 및 엔지니어링 요약

`termux-vision`은 모바일 ARM64 기기의 제한된 RAM과 연산 자원 하에서 **Zero PyTorch, Zero-Copy Buffer Pass, Process-Group Isolation, 듀얼 엔진(Python/Node.js)**을 통해 높은 안정성과 빠른 반응 속도를 달성한 온디바이스 컴퓨터 비전/VLM 표준 프레임워크입니다.
