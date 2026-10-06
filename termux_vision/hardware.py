"""
AMEVA Unified Hardware & Environment Diagnostics for Termux AI Engines.
Component: [VISION]
"""
from __future__ import annotations

import os
import re
import sys
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Any, List, Dict


@dataclass
class HardwareProfile:
    """Standardized Hardware & Capabilities Profile for Mobile Termux Environment."""
    is_termux: bool = False
    is_android: bool = False
    is_arm64: bool = False
    cpu_count: int = 4
    recommended_threads: int = 4
    ram_total_mb: float = 0.0
    ram_available_mb: float = 0.0
    has_neon: bool = False
    has_fp16: bool = False
    has_dotprod: bool = False
    has_vulkan: bool = False
    gpu_name: Optional[str] = None
    soc_model: Optional[str] = None
    features: List[str] = field(default_factory=list)

    @property
    def soc_name(self) -> str:
        return self.soc_model or "Unknown SoC"

    @property
    def cpu_cores(self) -> int:
        return self.cpu_count

    @property
    def threads(self) -> int:
        return self.recommended_threads


def is_termux() -> bool:
    """Authoritative detection for Android Termux environment."""
    if os.environ.get("TERMUX_VERSION") or os.environ.get("TERMUX_APP_PID"):
        return True
    prefix = os.environ.get("PREFIX", "")
    if "/com.termux" in prefix:
        return True
    return Path("/data/data/com.termux").is_dir()


def is_android() -> bool:
    """Check whether running on Android (Termux execution implies Android runtime)."""
    return is_termux()



def _read_cpu_features() -> Tuple[List[str], int]:
    """Read CPU features and core count from /proc/cpuinfo."""
    features: List[str] = []
    cores = 0
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            content = cpuinfo.read_text(encoding="utf-8", errors="ignore")
            for line in content.splitlines():
                if line.startswith("processor"):
                    cores += 1
                elif line.startswith("Features") or line.startswith("flags"):
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        features.extend(parts[1].strip().lower().split())
        except Exception:
            pass
    if cores == 0:
        cores = os.cpu_count() or 4
    return list(set(features)), cores


def _get_memory_info_mb() -> Tuple[float, float]:
    """Read total and available RAM in megabytes."""
    total_mb = 0.0
    avail_mb = 0.0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            content = meminfo.read_text(encoding="utf-8", errors="ignore")
            for line in content.splitlines():
                if line.startswith("MemTotal:"):
                    total_mb = float(re.findall(r"\d+", line)[0]) / 1024.0
                elif line.startswith("MemAvailable:"):
                    avail_mb = float(re.findall(r"\d+", line)[0]) / 1024.0
        except Exception:
            pass
    return round(total_mb, 1), round(avail_mb, 1)


def _probe_vulkan_driver() -> Tuple[bool, Optional[str]]:
    """Probe Vulkan ICD driver existence on Android."""
    vulkan_libs = [
        "/system/lib64/libvulkan.so",
        "/vendor/lib64/hw/vulkan.adreno.so",
        "/vendor/lib64/libvulkan.so",
        "/system/lib/libvulkan.so",
        "/vendor/lib/hw/vulkan.adreno.so",
    ]
    for lib in vulkan_libs:
        if Path(lib).exists():
            return True, "Adreno / Mali Vulkan Driver"
    # Fallback to pkg / ameva-runtime check
    if shutil.which("vulkaninfo") or Path("/data/data/com.termux/files/usr/lib/libvulkan.so").exists():
        return True, "Generic Vulkan Driver"
    return False, None


def detect_hardware() -> HardwareProfile:
    """Analyze current device hardware and return standardized profile."""
    features, cores = _read_cpu_features()
    total_mb, avail_mb = _get_memory_info_mb()
    has_vk, gpu_name = _probe_vulkan_driver()
    
    is_arm = sys.platform.startswith("linux") and ("aarch64" in os.uname().machine.lower() or "arm64" in os.uname().machine.lower()) if hasattr(os, "uname") else False
    if not is_arm:
        import platform
        m = platform.machine().lower()
        is_arm = "arm64" in m or "aarch64" in m

    has_neon = "asimd" in features or "neon" in features or is_arm
    has_fp16 = "fphp" in features or "fp16" in features or "asimdhp" in features
    has_dotprod = "asimddp" in features or "dotprod" in features

    # Optimal threads heuristic: leave 1-2 cores for system/UI responsiveness
    if cores >= 8:
        optimal_threads = 4  # Standard Big-cluster size
    elif cores >= 6:
        optimal_threads = 4
    elif cores >= 4:
        optimal_threads = min(3, cores)
    else:
        optimal_threads = max(1, cores)

    soc = None
    if Path("/system/build.prop").exists():
        try:
            bp = Path("/system/build.prop").read_text(encoding="utf-8", errors="ignore")
            for line in bp.splitlines():
                if "ro.soc.model" in line or "ro.hardware" in line:
                    soc = line.split("=", 1)[-1].strip()
                    break
        except Exception:
            pass

    return HardwareProfile(
        is_termux=is_termux(),
        is_android=is_android(),
        is_arm64=is_arm,
        cpu_count=cores,
        recommended_threads=optimal_threads,
        ram_total_mb=total_mb,
        ram_available_mb=avail_mb,
        has_neon=has_neon,
        has_fp16=has_fp16,
        has_dotprod=has_dotprod,
        has_vulkan=has_vk,
        gpu_name=gpu_name,
        soc_model=soc,
        features=features,
    )


def get_optimal_threads() -> int:
    """Get calculated optimal worker thread count."""
    return detect_hardware().recommended_threads


def _resolve_ameva_runtime() -> Optional[Any]:
    """Dynamically resolve ameva_runtime engine if installed."""
    try:
        import ameva_runtime
        return ameva_runtime
    except ImportError:
        return None


def resolve_device(requested_device: str = "auto") -> Tuple[str, int]:
    """Resolve compute device ('cpu' or 'vulkan') and recommended GPU offload layers."""
    req = str(requested_device).lower().strip()
    if req in ("vulkan", "gpu"):
        rt = _resolve_ameva_runtime()
        if rt is None:
            # Under fail-fast policy, guide user to ameva-runtime
            from .exceptions import HardwareCompatibilityError
            raise HardwareCompatibilityError(
                "Vulkan GPU acceleration requires commercial engine. Run: pip install ameva-runtime",
                code="E003_VULKAN_NOT_PROVISIONED"
            )
        return "vulkan", 99
    
    # Auto resolution
    if req == "auto":
        rt = _resolve_ameva_runtime()
        if rt is not None and detect_hardware().has_vulkan:
            return "vulkan", 99
        return "cpu", 0

    return "cpu", 0


resolve_device_backend = resolve_device


def bind_hardware(engine: Any, requested_device: str = "auto", **kwargs) -> Optional[Any]:
    """Bind engine instance to underlying hardware compute backend."""
    device, ngl = resolve_device(requested_device)
    if hasattr(engine, "device"):
        setattr(engine, "device", device)
    if hasattr(engine, "n_gpu_layers"):
        setattr(engine, "n_gpu_layers", ngl)
    return engine


# Standard Unified Model Search Dirs
def get_unified_model_search_dirs(submodule: str = "vision") -> List[Path]:
    """Standard multi-tier model search directories across Termux ecosystem."""
    dirs: List[Path] = []
    
    # 1. Custom ENV override
    custom_dir = os.environ.get(f"TERMUX_{submodule.upper()}_MODELS_DIR") or os.environ.get("AMEVA_MODELS_DIR")
    if custom_dir:
        dirs.append(Path(custom_dir))

    # 2. Standard XDG Cache Directory (Primary SSOT)
    dirs.append(Path.home() / ".cache" / f"termux-{submodule}" / "models")
    dirs.append(Path.home() / ".cache" / f"termux-{submodule}")

    # 3. Legacy fallbacks
    dirs.append(Path.home() / f".termux-{submodule}" / "models")
    dirs.append(Path.home() / f".termux-{submodule}")
    
    # 4. Termux prefix shared data
    prefix = os.environ.get("PREFIX", "/data/data/com.termux/files/usr")
    dirs.append(Path(prefix) / "share" / f"termux-{submodule}" / "models")

    return [d for d in dirs if d.exists()]
