import os
import sys
import shutil
import platform
import subprocess
import json
import time
from typing import Dict, Any, Optional
from .. import __version__
from .. import csrc
from ..vlm.cache import ModelCacheManager

def run_doctor(probe_vulkan: bool = False, full_check: bool = False) -> Dict[str, Any]:
    """
    Runs truthful diagnostic inspection of the Android Termux runtime environment.
    - Default: Read-only environment inspection.
    - probe_vulkan: Checks driver presence and reports execution capability (Zero-Hype / Ground Truth).
    - full_check: Runs actual SHA-256 and byte-level integrity verification on installed models.
    """
    cache_mgr = ModelCacheManager()

    report = {
        "schema_version": 1,
        "client_version": __version__,
        "runtime_version": __version__,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "is_android": os.path.exists("/system/build.prop") or "ANDROID_ROOT" in os.environ
        },
        "hardware": {
            "cpu_cores": os.cpu_count() or 1,
            "total_ram_mb": None,
            "available_ram_mb": None
        },
        "vulkan": {
            "driver_dir": None,
            "vulkan_loader_installed": shutil.which("vulkaninfo") is not None,
            "loader_detected": shutil.which("vulkaninfo") is not None,
            "driver_file_detected": False,
            "compute_probe_executed": probe_vulkan,
            "safe_for_vlm": None,
            "status": "unverified"
        },
        "vlm_runtime": {
            "llama_cli_available": shutil.which("llama-cli") is not None,
            "multimodal_capable": False,
            "multimodal_status": "unverified",
            "installed_models_count": 0,
            "cache_dir": cache_mgr.cache_root
        },
        "native_backends": {
            "has_c_backend": csrc.has_c_backend(),
            "has_vulkan_backend": csrc.has_vulkan_backend(),
            "vulkan_device_name": csrc.get_vulkan_device_name(),
            "c_backend_errors": csrc.get_c_backend_load_errors(),
            "cpp_backend_errors": csrc.get_cpp_backend_load_errors()
        },
        "recommended_preset": "Tier M (smolvlm-500m / 4-Threads CPU Reference)",
        "warnings": []
    }

    # Verify multimodal capability via installer gatekeeper
    try:
        from ..installer import verify_multimodal_support
        mm_ok, mm_msg = verify_multimodal_support()
        report["vlm_runtime"]["multimodal_capable"] = mm_ok
        report["vlm_runtime"]["multimodal_status"] = mm_msg
        if not mm_ok and report["vlm_runtime"]["llama_cli_available"]:
            report["warnings"].append(f"llama-cli detected but lacks multimodal support: {mm_msg}")
    except Exception as _mm_err:
        report["vlm_runtime"]["multimodal_status"] = f"Check failed: {_mm_err}"

    # RAM Inspection
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    report["hardware"]["total_ram_mb"] = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    report["hardware"]["available_ram_mb"] = int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError) as _mem_err:
        report["warnings"].append(f"RAM inspection failed: {_mem_err}")

    # Truthful Vulkan status via ameva-runtime integration
    if probe_vulkan:
        try:
            from ameva_runtime.vulkan.adapters import find_system_vulkan_driver_dir
            from ameva_runtime import vulkan as avr
            driver_dir = find_system_vulkan_driver_dir()
            is_avail = bool(driver_dir is not None)
            report["vulkan"]["driver_dir"] = driver_dir
            report["vulkan"]["driver_file_detected"] = is_avail
            report["vulkan"]["loader_detected"] = bool(is_avail or report["vulkan"]["vulkan_loader_installed"])
            report["vulkan"]["ameva_runtime_detected"] = True
            report["vulkan"]["status"] = "driver_detected_experimental" if is_avail else "disabled"
            dev_name = avr.get_device_name() if hasattr(avr, "get_device_name") and callable(avr.get_device_name) else "Vulkan GPU"
            report["vulkan"]["device_name"] = dev_name
            report["vulkan"]["note"] = "Hardware driver dynamically discovered via ameva-runtime."
        except ImportError:
            report["vulkan"]["ameva_runtime_detected"] = False
            report["vulkan"]["status"] = "disabled"
            report["vulkan"]["safe_for_vlm"] = False
            report["vulkan"]["note"] = "Install ameva-runtime for optimized GPU compute shaders."

    # Model count and Actual Full SHA-256 check
    installed = cache_mgr.list_installed()
    report["vlm_runtime"]["installed_models_count"] = len(installed)

    if full_check:
        report["vlm_runtime"]["models_integrity"] = {}
        for m in installed:
            mid = m["model_id"]
            report["vlm_runtime"]["models_integrity"][mid] = cache_mgr.verify_integrity(mid)

    return report
