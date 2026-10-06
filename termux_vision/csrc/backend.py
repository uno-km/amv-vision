import os
import sys
import ctypes
import threading
import numpy as np

# Thread-safe synchronization lock
_backend_lock = threading.Lock()

_lib = None
_lib_path = None
_c_backend_load_errors = {}

def _load_c_backend():
    global _lib, _lib_path
    if _lib is not None:
        return _lib

    with _backend_lock:
        if _lib is not None:
            return _lib

        dir_path = os.path.dirname(os.path.abspath(__file__))
        prefix_lib = os.path.join(os.environ.get("PREFIX", "/data/data/com.termux/files/usr"), "lib")
        candidates = [
            os.path.join(prefix_lib, "libfast_cv.so"),
            "/data/data/com.termux/files/usr/lib/libfast_cv.so",
            os.path.join(dir_path, "libfast_cv.so"),
            os.path.join(dir_path, "fast_cv.so"),
            os.path.join(dir_path, "fast_cv.dll"),
            os.path.join(dir_path, "..", "libfast_cv.so"),
        ]

        for p in candidates:
            if not os.path.exists(p):
                _c_backend_load_errors[p] = "File not found"
                continue

            try:
                lib_handle = ctypes.CDLL(p)
                _setup_signatures(lib_handle)
                _lib = lib_handle
                _lib_path = p
                return _lib
            except Exception as e:
                _c_backend_load_errors[p] = f"{type(e).__name__}: {e}"
                import logging
                logging.getLogger("termux_vision.csrc.backend").warning(
                    "[termux-vision] Native C backend dlopen failed at '%s': %s", p, e
                )

        return None

def get_c_backend_load_errors() -> dict:
    """Returns diagnostic dictionary of C backend candidates and load error reasons."""
    _load_c_backend()
    return dict(_c_backend_load_errors)

def get_cpp_backend_load_errors() -> dict:
    """Returns diagnostic dictionary of C++ backend candidates and load error reasons."""
    _load_cpp_backend()
    return dict(_cpp_backend_load_errors)

def _setup_signatures(lib):
    # compute_integral_c
    lib.compute_integral_c.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.compute_integral_c.restype = None

    # sobel_c
    lib.sobel_c.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.sobel_c.restype = None

    # canny_nms_threshold_c
    lib.canny_nms_threshold_c.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float
    ]
    lib.canny_nms_threshold_c.restype = None

    # morphology_c
    lib.morphology_c.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.morphology_c.restype = None

    # haar_detect_multiscale_c
    lib.haar_detect_multiscale_c.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    lib.haar_detect_multiscale_c.restype = ctypes.c_int

def has_c_backend() -> bool:
    return _load_c_backend() is not None

def _ensure_2d_uint8(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.ndim == 3:
        # Standard luminance conversion if RGB/BGR
        if arr.shape[2] in (3, 4):
            arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image array, received array with shape {arr.shape}")
    
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0 and arr.size > 0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr, dtype=np.uint8)

def c_compute_integral(src_uint8: np.ndarray) -> np.ndarray:
    lib = _load_c_backend()
    if lib is None:
        err_msg = "\n".join(f"  - {path}: {err}" for path, err in _c_backend_load_errors.items())
        raise RuntimeError(
            f"Native C backend is not available.\nSearched candidate paths:\n{err_msg}\n"
            f"[Action Recommendation] Please compile libfast_cv.so via clang: clang -O3 -shared -fPIC -o termux_vision/csrc/libfast_cv.so termux_vision/csrc/fast_cv.c -lm"
        )
    
    src_cont = _ensure_2d_uint8(src_uint8)
    h, w = src_cont.shape
    dst = np.zeros((h + 1, w + 1), dtype=np.float64)

    src_ptr = src_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lib.compute_integral_c(src_ptr, dst_ptr, w, h)
    return dst

def c_sobel(src_uint8: np.ndarray):
    lib = _load_c_backend()
    if lib is None:
        err_msg = "\n".join(f"  - {path}: {err}" for path, err in _c_backend_load_errors.items())
        raise RuntimeError(
            f"Native C backend is not available.\nSearched candidate paths:\n{err_msg}\n"
            f"[Action Recommendation] Please compile libfast_cv.so via clang: clang -O3 -shared -fPIC -o termux_vision/csrc/libfast_cv.so termux_vision/csrc/fast_cv.c -lm"
        )

    src_cont = _ensure_2d_uint8(src_uint8)
    h, w = src_cont.shape
    mag = np.zeros((h, w), dtype=np.float32)
    angle = np.zeros((h, w), dtype=np.float32)

    src_ptr = src_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    mag_ptr = mag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    angle_ptr = angle.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.sobel_c(src_ptr, mag_ptr, angle_ptr, w, h)
    return mag, angle

_cpp_lib = None
_cpp_lib_path = None
_cpp_backend_load_errors = {}

def _load_cpp_backend():
    global _cpp_lib, _cpp_lib_path
    if _cpp_lib is not None:
        return _cpp_lib

    with _backend_lock:
        if _cpp_lib is not None:
            return _cpp_lib

        dir_path = os.path.dirname(os.path.abspath(__file__))
        prefix_lib = os.path.join(os.environ.get("PREFIX", "/data/data/com.termux/files/usr"), "lib")
        candidates = [
            os.path.join(prefix_lib, "libfast_cv_engine.so"),
            "/data/data/com.termux/files/usr/lib/libfast_cv_engine.so",
            os.path.join(dir_path, "libfast_cv_engine.so"),
            os.path.join(dir_path, "fast_cv_engine.so"),
            os.path.join(dir_path, "libfast_cv_engine.dll"),
            os.path.join(dir_path, "fast_cv_engine.dll"),
            os.path.join(dir_path, "libfast_cv_engine.dylib"),
            os.path.join(dir_path, "..", "libfast_cv_engine.so"),
            os.path.join(dir_path, "..", "libfast_cv_engine.dll"),
        ]

        for p in candidates:
            if not os.path.exists(p):
                _cpp_backend_load_errors[p] = "File not found"
                continue

            try:
                cpp = ctypes.CDLL(p)
                cpp.fast_canny_cpp.argtypes = [
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.c_float
                ]
                cpp.fast_canny_cpp.restype = ctypes.c_int

                cpp.fast_scale_cpp.argtypes = [
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int
                ]
                cpp.fast_scale_cpp.restype = ctypes.c_int

                if hasattr(cpp, "fast_cv_cleanup_context"):
                    cpp.fast_cv_cleanup_context.argtypes = []
                    cpp.fast_cv_cleanup_context.restype = None

                _cpp_lib = cpp
                _cpp_lib_path = p
                return _cpp_lib
            except Exception as e:
                _cpp_backend_load_errors[p] = f"{type(e).__name__}: {e}"
                import logging
                logging.getLogger("termux_vision.csrc.backend").warning(
                    "[termux-vision] Native C++ backend dlopen failed at '%s': %s", p, e
                )

        return None

def cleanup_native_context() -> None:
    """Explicitly releases native C++ scratch buffers."""
    global _cpp_lib
    if _cpp_lib is not None and hasattr(_cpp_lib, "fast_cv_cleanup_context"):
        try:
            _cpp_lib.fast_cv_cleanup_context()
        except Exception as _clean_err:
            import logging
            logging.getLogger("termux_vision.csrc.backend").debug("fast_cv_cleanup_context error: %s", _clean_err)

import atexit
atexit.register(cleanup_native_context)

_vk_lib = None
_vk_lib_path = None
_vk_backend_load_errors = {}

def _load_vulkan_backend():
    global _vk_lib, _vk_lib_path
    if _vk_lib is not None:
        return _vk_lib

    with _backend_lock:
        if _vk_lib is not None:
            return _vk_lib

        dir_path = os.path.dirname(os.path.abspath(__file__))
        prefix_lib = os.path.join(os.environ.get("PREFIX", "/data/data/com.termux/files/usr"), "lib")
        candidates = [
            os.path.join(prefix_lib, "libvulkan_cv.so"),
            "/data/data/com.termux/files/usr/lib/libvulkan_cv.so",
            os.path.join(dir_path, "libvulkan_cv.so"),
            os.path.join(dir_path, "vulkan_cv.so"),
            os.path.join(dir_path, "libvulkan_cv.dll"),
            os.path.join(dir_path, "vulkan_cv.dll"),
            os.path.join(dir_path, "..", "libvulkan_cv.so"),
        ]

        for p in candidates:
            if not os.path.exists(p):
                _vk_backend_load_errors[p] = "File not found"
                continue

            try:
                vk = ctypes.CDLL(p)
                vk.is_vulkan_cv_available.argtypes = []
                vk.is_vulkan_cv_available.restype = ctypes.c_int
                vk.get_vulkan_device_name.argtypes = []
                vk.get_vulkan_device_name.restype = ctypes.c_char_p
                vk.vulkan_canny.argtypes = [
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.c_float
                ]
                vk.vulkan_canny.restype = ctypes.c_int

                if vk.is_vulkan_cv_available():
                    _vk_lib = vk
                    _vk_lib_path = p
                    return _vk_lib
                else:
                    _vk_backend_load_errors[p] = "is_vulkan_cv_available() returned 0"
            except Exception as e:
                _vk_backend_load_errors[p] = f"{type(e).__name__}: {e}"

        return None

def has_vulkan_backend() -> bool:
    """Inspects native Vulkan Compute availability."""
    lib = _load_vulkan_backend()
    if lib is not None:
        return True
    try:
        from ameva_runtime import vulkan as avr
        return bool(avr.is_available())
    except ImportError:
        return False

def get_vulkan_device_name() -> str:
    """Returns physical Vulkan GPU device name via native backend or ameva-runtime."""
    lib = _load_vulkan_backend()
    if lib is not None and hasattr(lib, "get_vulkan_device_name"):
        try:
            name = lib.get_vulkan_device_name()
            if name:
                return name.decode("utf-8", errors="replace")
        except Exception:
            pass
    try:
        from ameva_runtime import vulkan as avr
        return avr.get_device_name() or "Vulkan GPU Device"
    except ImportError:
        return "None (CPU Pipeline Only)"

def c_canny(
    src_uint8: np.ndarray, 
    low_threshold: float = 40.0, 
    high_threshold: float = 120.0,
    device: str = "auto"
) -> np.ndarray:
    dev = (device or "auto").lower().strip()
    src_cont = _ensure_2d_uint8(src_uint8)
    h, w = src_cont.shape

    # 1. OpenCL: Explicit Fail-Fast contract (Zero-Silent-Fallback)
    if dev == "opencl":
        from ..errors import OpenCLNotImplementedError
        raise OpenCLNotImplementedError(
            reason="OpenCL vision compute pipeline is reserved for ARM Mali/Exynos devices (e.g. Galaxy S20, A53)."
        )

    # 2. 100% Vulkan GPU Path (Anti-Deception: Zero CPU Fallback)
    if dev in ("vulkan", "gpu"):
        vk_lib = _load_vulkan_backend()
        if vk_lib is None:
            err_msg = "\n".join(f"  - {path}: {err}" for path, err in _vk_backend_load_errors.items())
            from ..errors import VulkanNotAvailableError
            raise VulkanNotAvailableError(
                reason=f"Native Vulkan GPU Compute backend (libvulkan_cv.so) could not be loaded.\n"
                       f"Searched paths:\n{err_msg}\n"
                       f"[Action Required] Explicit '--gpu' requires functional Vulkan compute.\n"
                       f"  - Switch to CPU: --cpu or --device cpu\n"
                       f"  - Check Vulkan: termux-vision doctor --probe-vulkan"
            )

        dst = np.zeros((h, w), dtype=np.uint8)
        src_ptr = src_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        ok = vk_lib.vulkan_canny(src_ptr, dst_ptr, w, h, float(low_threshold), float(high_threshold))
        if not ok:
            from ..errors import GpuExecutionError
            raise GpuExecutionError(
                reason="Vulkan GPU Canny execution failed during vkQueueSubmit or fence synchronization."
            )
        return dst

    # 3. Auto Mode: Prefer Vulkan GPU, transparently fallback to CPU if GPU unavailable
    if dev == "auto":
        vk_lib = _load_vulkan_backend()
        if vk_lib is not None:
            dst = np.zeros((h, w), dtype=np.uint8)
            src_ptr = src_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            if vk_lib.vulkan_canny(src_ptr, dst_ptr, w, h, float(low_threshold), float(high_threshold)):
                return dst

    # 4. 100% CPU NEON / C++ Native Engine (Explicit '--cpu' or auto fallback)
    cpp_lib = _load_cpp_backend()
    if cpp_lib is not None:
        dst = np.zeros((h, w), dtype=np.uint8)
        src_ptr = src_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        ok = cpp_lib.fast_canny_cpp(src_ptr, dst_ptr, w, h, float(low_threshold), float(high_threshold))
        if ok:
            return dst

    # 5. Standard High-speed C Kernel
    lib = _load_c_backend()
    if lib is not None:
        mag, angle = c_sobel(src_cont)
        dst = np.zeros((h, w), dtype=np.uint8)
        mag_ptr = mag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        angle_ptr = angle.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        lib.canny_nms_threshold_c(mag_ptr, angle_ptr, dst_ptr, w, h, float(low_threshold), float(high_threshold))
        return dst

    err_msg = "\n".join(f"  - {path}: {err}" for path, err in _c_backend_load_errors.items())
    raise RuntimeError(
        f"Native C/C++ backend is not available.\nSearched candidate paths:\n{err_msg}\n"
        f"[Action Recommendation] Please compile libfast_cv_engine.so or libfast_cv.so via clang."
    )

def c_morphology(src_uint8: np.ndarray, is_dilate: bool = True) -> np.ndarray:
    lib = _load_c_backend()
    if lib is None:
        err_msg = "\n".join(f"  - {path}: {err}" for path, err in _c_backend_load_errors.items())
        raise RuntimeError(
            f"Native C backend is not available.\nSearched candidate paths:\n{err_msg}"
        )

    src_cont = _ensure_2d_uint8(src_uint8)
    h, w = src_cont.shape
    dst = np.zeros((h, w), dtype=np.uint8)

    src_ptr = src_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    lib.morphology_c(src_ptr, dst_ptr, w, h, 1 if is_dilate else 0)
    return dst

def c_haar_detect(integral: np.ndarray, width: int, height: int, scale_factor: float = 1.2, min_size: int = 24, max_size: int = 512, max_boxes: int = 1000):
    lib = _load_c_backend()
    if lib is None:
        err_msg = "\n".join(f"  - {path}: {err}" for path, err in _c_backend_load_errors.items())
        raise RuntimeError(
            f"Native C backend is not available.\nSearched candidate paths:\n{err_msg}"
        )

    out_boxes = np.zeros((max_boxes, 4), dtype=np.int32)
    out_scores = np.zeros(max_boxes, dtype=np.float32)

    int_cont = np.ascontiguousarray(integral, dtype=np.float64)
    int_ptr = int_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    boxes_ptr = out_boxes.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    scores_ptr = out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    num_found = lib.haar_detect_multiscale_c(
        int_ptr,
        width,
        height,
        scale_factor,
        min_size,
        max_size,
        boxes_ptr,
        scores_ptr,
        max_boxes
    )

    boxes = [tuple(out_boxes[i].tolist()) for i in range(num_found)]
    scores = [float(out_scores[i]) for i in range(num_found)]
    return boxes, scores
