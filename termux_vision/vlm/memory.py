import os
import threading
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from ..errors import InsufficientMemoryError, TermuxVisionError

logger = logging.getLogger("termux_vision.vlm.memory")

class ConcurrentEngineLimitError(TermuxVisionError):
    """Raised when multiple active VLM engines attempt concurrent instantiation."""
    pass

_ACTIVE_ENGINE_LOCK = threading.Lock()
_ACTIVE_ENGINE_COUNT = 0

@dataclass(frozen=True)
class MemoryEstimate:
    model_weights_mb: int
    vision_encoder_mb: int
    kv_cache_mb: int
    compute_buffers_mb: int
    runtime_overhead_mb: int
    estimated_peak_mb: int
    confidence: str  # "measured" | "calibrated" | "estimated"

def get_system_ram_info() -> Dict[str, int]:
    """Inspects total and available physical RAM in MB from /proc/meminfo or system telemetry."""
    total_mb = 0
    avail_mb = 0

    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_mb = int(line.split()[1]) // 1024
                    elif line.startswith("MemAvailable:"):
                        avail_mb = int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError) as exc:
            logger.debug("Failed reading /proc/meminfo: %s", exc)

    if total_mb == 0:
        try:
            # POSIX sysconf fallback
            if hasattr(os, "sysconf"):
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                total_mb = (pages * page_size) // (1024 * 1024)
                avail_mb = total_mb // 2
        except (AttributeError, ValueError, OSError) as _sc_err:
            logger.debug("POSIX sysconf memory inspection failed: %s", _sc_err)

    if total_mb == 0:
        # Default reasonable baseline if all probes fail
        total_mb = 4096
        avail_mb = 2048

    return {"total_mb": total_mb, "available_mb": avail_mb}

def check_memory_admission(
    estimate: MemoryEstimate,
    user_budget_mb: Optional[int] = None,
    memory_policy: str = "warn"
) -> Tuple[bool, Optional[str]]:
    """
    Evaluates memory admission against system telemetry and user policy.
    - 'warn' (default): Logs warning and allows execution.
    - 'strict': Raises InsufficientMemoryError when floor is exceeded.
    - 'unrestricted': Bypasses limits and attempts execution to device limits.
    """
    ram_info = get_system_ram_info()
    total_ram = ram_info["total_mb"]
    avail_ram = ram_info["available_mb"]

    system_reserve_mb = max(384, int(total_ram * 0.08))
    uncertainty_margin_mb = max(128, int(estimate.estimated_peak_mb * 0.15))
    required_floor_mb = estimate.estimated_peak_mb + system_reserve_mb + uncertainty_margin_mb

    warning_msg = None
    policy = memory_policy.lower().strip()

    if policy == "unrestricted":
        if avail_ram < required_floor_mb:
            warning_msg = f"[Unrestricted Mode] Estimated peak ({estimate.estimated_peak_mb}MB) exceeds safe headroom ({avail_ram}MB available). Proceeding at user request."
        return True, warning_msg

    if user_budget_mb is not None and estimate.estimated_peak_mb > user_budget_mb:
        msg = f"Estimated peak ({estimate.estimated_peak_mb}MB) exceeds user budget ({user_budget_mb}MB)."
        if policy == "strict":
            raise InsufficientMemoryError(estimate.estimated_peak_mb, user_budget_mb, msg)
        else:
            warning_msg = msg

    if avail_ram < required_floor_mb:
        msg = f"Available RAM ({avail_ram}MB) is less than safe floor ({required_floor_mb}MB = peak {estimate.estimated_peak_mb}MB + reserve {system_reserve_mb}MB + margin {uncertainty_margin_mb}MB)."
        if policy == "strict":
            raise InsufficientMemoryError(required_floor_mb, avail_ram, msg)
        else:
            warning_msg = msg

    return True, warning_msg

def acquire_engine_lock():
    global _ACTIVE_ENGINE_COUNT
    with _ACTIVE_ENGINE_LOCK:
        _ACTIVE_ENGINE_COUNT += 1

def release_engine_lock():
    global _ACTIVE_ENGINE_COUNT
    with _ACTIVE_ENGINE_LOCK:
        if _ACTIVE_ENGINE_COUNT > 0:
            _ACTIVE_ENGINE_COUNT -= 1
