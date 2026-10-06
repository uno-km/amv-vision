#!/usr/bin/env bash
# ==============================================================================
# termux-vision: Universal Dynamic One-Line Bootstrap Installer (v1.4.0)
# Open-Source under Apache License 2.0 (AMEVA Foundation)
# Usage: curl -sL https://raw.githubusercontent.com/uno-km/termux-vision/main/install.sh | bash
# ==============================================================================
# Dynamic version resolution: Environment -> GitHub API (latest) -> Dynamic fallback
if [ -n "${TERMUX_VISION_VERSION:-}" ]; then
    VERSION="${TERMUX_VISION_VERSION}"
else
    VERSION=$(curl -sL https://api.github.com/repos/uno-km/termux-vision/releases/latest 2>/dev/null | grep '"tag_name":' | head -n 1 | sed -E 's/.*"v?([^"]+)".*/\1/' || true)
    if [ -z "${VERSION}" ]; then
        VERSION="latest"
    fi
fi
REPO="uno-km/termux-vision"
ARCH="$(uname -m)"

# Parse CLI arguments for auto-yes / non-interactive flags
AUTO_YES=false
for arg in "$@"; do
    case "$arg" in
        -y|-Y|-yes|-YES|--yes|--all|-a)
            AUTO_YES=true
            ;;
    esac
done

echo "================================================================="
echo " [AMEVA Foundation] termux-vision Universal Installer v${VERSION}"
echo "================================================================="
echo "-> Detected Architecture: ${ARCH}"

# 1. Platform Detection
IS_TERMUX=false
if [ -d "/data/data/com.termux" ] || [ -n "${TERMUX_VERSION:-}" ]; then
    IS_TERMUX=true
    BIN_DIR="${PREFIX:-/data/data/com.termux/files/usr}/bin"
    LIB_DIR="${PREFIX:-/data/data/com.termux/files/usr}/lib"
    echo "-> Detected Platform: Android Termux (Bionic libc)"
else
    BIN_DIR="/usr/local/bin"
    LIB_DIR="/usr/local/lib"
    echo "-> Detected Platform: Generic Linux / Host POSIX"
fi

if [ "${ARCH}" != "aarch64" ] && [ "${ARCH}" != "arm64" ]; then
    echo "[WARN] Architecture is ${ARCH}. ARM64 is strongly recommended for mobile NPU/GPU tensor acceleration."
fi

# 2. Storage Setup (Termux only)
if [ "${IS_TERMUX}" = "true" ] && command -v termux-setup-storage >/dev/null 2>&1; then
    if [ ! -d "${HOME}/storage" ]; then
        echo "-> Requesting Android storage permission..."
        termux-setup-storage || true
    fi
fi

# 3. System Package Dependencies
if [ "${IS_TERMUX}" = "true" ] && command -v pkg >/dev/null 2>&1; then
    echo "-> [1/6] Updating Termux package repositories..."
    pkg update -y
    echo "-> [2/6] Installing build toolchains and pure CPU runtime dependencies..."
    pkg install -y \
        python \
        nodejs \
        clang \
        make \
        cmake \
        git \
        termux-api \
        wget \
        python-numpy \
        libjpeg-turbo
elif command -v apt-get >/dev/null 2>&1; then
    echo "-> [1/6] Updating Ubuntu/Debian repositories..."
    apt-get update -y
    echo "-> [2/6] Installing build tools and dependencies..."
    apt-get install -y build-essential git python3 python3-pip python3-numpy libjpeg-dev nodejs npm curl wget
fi

# 4. Pre-provision Core Python Toolchain & Ecosystem Dependencies
echo "-> [3/6] Checking required VLM runtime dependency: termux-llamacpp..."
python -m pip install setuptools wheel >/dev/null 2>&1 || true

INSTALL_LLAMACPP=false
CURRENT_LLAMA_VER=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('termux-llamacpp'))" 2>/dev/null || true)
LATEST_LLAMA_VER=$(curl -sL https://pypi.org/pypi/termux-llamacpp/json 2>/dev/null | grep '"version":' | head -n 1 | sed -E 's/.*"version":\s*"([^"]+)".*/\1/' || true)
if [ -z "${LATEST_LLAMA_VER}" ]; then
    LATEST_LLAMA_VER="latest"
fi

if [ -z "${CURRENT_LLAMA_VER}" ]; then
    echo "  [Notice] 'termux-llamacpp' is currently NOT installed."
    if [ "${AUTO_YES}" = "true" ]; then
        echo "  -> Auto-approval enabled (-y/--all). Installing latest termux-llamacpp (v${LATEST_LLAMA_VER})..."
        INSTALL_LLAMACPP=true
    else
        read -r -p "지금 termux-llamacpp 설치가 필요합니다. 설치하시겠습니까? [y/N]: " USER_CHOICE
        case "${USER_CHOICE}" in
            [yY]|[yY][eE][sS])
                INSTALL_LLAMACPP=true
                ;;
            *)
                echo "  [Warning] termux-llamacpp installation skipped by user. VLM multimodal features will be unavailable."
                ;;
        esac
    fi
else
    echo "  -> Found installed termux-llamacpp (v${CURRENT_LLAMA_VER}). Latest available: v${LATEST_LLAMA_VER}"
    if [ "${LATEST_LLAMA_VER}" != "latest" ] && [ "${CURRENT_LLAMA_VER}" != "${LATEST_LLAMA_VER}" ]; then
        if [ "${AUTO_YES}" = "true" ]; then
            echo "  -> Auto-approval enabled (-y/--all). Updating termux-llamacpp to v${LATEST_LLAMA_VER}..."
            INSTALL_LLAMACPP=true
        else
            read -r -p "현재 termux-llamacpp 버전(v${CURRENT_LLAMA_VER})이 최신 버전(v${LATEST_LLAMA_VER})과 다릅니다. 업데이트하시겠습니까? [y/N]: " USER_CHOICE
            case "${USER_CHOICE}" in
                [yY]|[yY][eE][sS])
                    INSTALL_LLAMACPP=true
                    ;;
                *)
                    echo "  -> Keeping existing version v${CURRENT_LLAMA_VER}."
                    ;;
            esac
        fi
    fi
fi

if [ "${INSTALL_LLAMACPP}" = "true" ]; then
    echo "  -> Installing / Updating termux-llamacpp to latest version..."
    python3 -m pip install --upgrade termux-llamacpp
fi

# 5. Standard Python SDK Installation
echo "-> [4/6] Installing termux-vision Python SDK (v${VERSION})..."
if [ -f "pyproject.toml" ]; then
    echo "   -> Installing from local source repository..."
    python -m pip install --no-build-isolation -e .
else
    echo "   -> Installing latest release from PyPI..."
    python -m pip install --upgrade termux-vision || true
fi

# 6. Compile Native C/C++ Compute Engines directly into $PREFIX/lib SSOT
if [ -f "termux_vision/csrc/fast_cv.c" ] && command -v clang >/dev/null 2>&1; then
    echo "-> [5/6] Compiling Native C & C++ Compute Acceleration Engines into ${LIB_DIR}..."
    mkdir -p "${LIB_DIR}"
    clang -O3 -shared -fPIC -o "${LIB_DIR}/libfast_cv.so" termux_vision/csrc/fast_cv.c -lm 2>/dev/null || true
    chmod 0755 "${LIB_DIR}/libfast_cv.so" 2>/dev/null || true
    if [ -f "termux_vision/csrc/fast_cv_engine.cpp" ]; then
        clang++ -O3 -shared -fPIC -o "${LIB_DIR}/libfast_cv_engine.so" termux_vision/csrc/fast_cv_engine.cpp 2>/dev/null || true
        chmod 0755 "${LIB_DIR}/libfast_cv_engine.so" 2>/dev/null || true
    fi
fi

# 7. Node.js Dual Engine CLI Installation
if command -v npm >/dev/null 2>&1; then
    echo "-> [6/6] Linking Node.js CLI..."
    if [ -f "package.json" ]; then
        npm install -g . || npm link || true
    else
        npm install -g termux-vision || true
    fi
fi

echo "================================================================="
echo "  [SUCCESS] termux-vision v${VERSION} successfully installed!"
echo "================================================================="
echo "  Dual-Engine Verification:"
echo "    * Python CLI:   termux-vision doctor"
echo "    * Node.js CLI:  npx termux-vision doctor"
echo "================================================================="

# 8. Run Hardware Diagnostics Probe
if command -v termux-vision >/dev/null 2>&1; then
    echo "-> Running Hardware Diagnostics Probe (CPU & Environment)..."
    termux-vision doctor || true
fi

# 9. Optional Model Download (Interactive only)
if [ -t 0 ]; then
    echo ""
    echo "-----------------------------------------------------------------"
    echo "  Official Multimodal VLM Model Download"
    echo "  Model       : SmolVLM-500M Instruct (smolvlm-500m)"
    echo "  Size        : ~550 MB (Q4_K_M + Q8_0 mmproj)"
    echo "  Target Path : ~/.cache/termux-vision/models/"
    echo "-----------------------------------------------------------------"
    read -p "Do you want to automatically download this model now? [y/N]: " answer
    case "$answer" in
        [yY]|[yY][eE][sS])
            echo "[*] Downloading smolvlm-500m (~550MB)..."
            if command -v termux-vision >/dev/null 2>&1; then
                termux-vision model install smolvlm-500m || true
            fi
            ;;
        *)
            echo "[*] Model download skipped."
            echo "    You can install anytime via: termux-vision model install smolvlm-500m"
            ;;
    esac
fi

echo ""
echo "termux-vision is ready for on-device computer vision and VLM inference."
