"""
termux_vision.control.component
AMEVA Component Protocol v1 — VisionControl

기존 cli/doctor.py + vlm/cache.py ModelCacheManager를 Adapter로 연결.
VLM + CV 이중 구조 추적.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from ameva_component import (
    ActivationLock, ComponentInfo, ComponentStateFile,
    ControlMode, InstanceRegistry, InstanceState, InstanceStatus,
    ModelRegistry, ModelState, ModelNotFound, ModelLoadFailed,
    OperationNotSupported, now_timestamps, log_stderr, PROTOCOL_COMPONENT,
)
from ameva_component.control import ComponentControl


class VisionControl(ComponentControl):

    COMPONENT_ID   = "termux-vision"
    COMPONENT_TYPE = "vision"
    CAPABILITIES   = ("vision.caption", "vision.vlm")

    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "termux-vision" / "models"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir  = cache_dir or self.DEFAULT_CACHE_DIR
        self._state_file = ComponentStateFile(self.COMPONENT_ID)
        self._model_reg  = ModelRegistry(self.COMPONENT_ID)
        self._inst_reg   = InstanceRegistry(self.COMPONENT_ID)
        self._act_lock   = ActivationLock()
        # Phase 4: Heartbeat Writer
        from termux_vision.control.status import VisionStatusWriter
        self._heartbeat = VisionStatusWriter(self)

    def _get_version(self) -> str:
        try:
            from termux_vision import __version__; return __version__
        except Exception: return "1.1.1"

    def component_info(self) -> dict:
        info = ComponentInfo(
            protocol=PROTOCOL_COMPONENT, component_id=self.COMPONENT_ID,
            component_type=self.COMPONENT_TYPE, version=self._get_version(),
            capabilities=self.CAPABILITIES,
        )
        info.validate()
        return info.to_dict()

    def doctor_lite(self) -> dict:
        ts = now_timestamps()
        state_data = self._state_file.read()
        stale = self._state_file.is_stale(threshold_ms=30_000)
        pid_info = self._check_pid()
        pid = pid_info.get("pid")
        pid_alive = pid_info.get("alive")

        instances = self._inst_reg.list_all()
        hot = [i for i in instances if i.state == InstanceState.HOT]

        # import 가능 여부만 (실제 모델 로드 금지)
        vlm_available = False
        try:
            import termux_vision.vlm.api
            vlm_available = True
        except ImportError as _imp_err:
            import logging
            logging.getLogger(__name__).debug("termux-vision: vlm.api not importable: %s", _imp_err)

        ready = vlm_available
        degraded = stale or (pid_alive is not True)

        proc_dict: dict[str, Any] = {
            "running": pid_alive,
            "pid": pid,
            "verified": pid_info.get("verified", False),
        }
        if "inspection_error" in pid_info:
            proc_dict["inspection_error"] = pid_info["inspection_error"]
        if "reason" in pid_info:
            proc_dict["reason"] = pid_info["reason"]

        return {
            "protocol": "ameva-component-status/1",
            "component_id": self.COMPONENT_ID, "component_type": self.COMPONENT_TYPE,
            "version": self._get_version(), "ready": ready, "degraded": degraded,
            **ts,
            "process": proc_dict,
            "capabilities": list(self.CAPABILITIES),
            "active_models": [i.model_id for i in hot],
            "backends": {"vlm": vlm_available, "cv": False},
            "instances": [{"instance_id": i.instance_id, "model_id": i.model_id,
                           "state": i.state.value, "active_jobs": i.active_jobs} for i in instances],
            "errors": [state_data.get("last_error")] if state_data and state_data.get("last_error") else [],
            "state_file": {"path": str(self._state_file.path), "stale": stale,
                           "updated_at": state_data.get("updated_at") if state_data else None},
        }

    def _check_pid(self) -> dict[str, Any]:
        """PID 파일 기반 프로세스 활성 여부 확인.

        BLOCKER 1: PermissionError/OSError를 alive=False로 은폐하지 않고
        alive=None, verified=False, inspection_error 구조로 반환.
        """
        import logging
        _log = logging.getLogger(__name__)
        pid_file = Path.home() / ".local" / "run" / "termux-vision.pid"

        if not pid_file.exists():
            return {
                "pid": None,
                "alive": False,
                "verified": True,
                "reason": "pid_file_missing",
            }

        try:
            raw = pid_file.read_text().strip()
        except OSError as read_err:
            _log.warning("termux-vision: PID file read failed: %s", read_err)
            return {
                "pid": None,
                "alive": None,
                "verified": False,
                "inspection_error": {
                    "code": "PID_FILE_READ_ERROR",
                    "message": str(read_err),
                },
            }

        try:
            pid = int(raw)
        except ValueError as parse_err:
            _log.warning("termux-vision: PID file contains non-integer value: %r", raw)
            return {
                "pid": None,
                "alive": None,
                "verified": False,
                "inspection_error": {
                    "code": "PID_PARSE_ERROR",
                    "message": f"Non-integer PID value: {raw!r}",
                },
            }

        try:
            os.kill(pid, 0)
            return {
                "pid": pid,
                "alive": True,
                "verified": True,
            }
        except ProcessLookupError:
            # 프로세스가 확실히 존재하지 않음 — 정상 종료 또는 이전 실행 잔여
            return {
                "pid": pid,
                "alive": False,
                "verified": True,
                "reason": "process_lookup_failed",
            }
        except PermissionError as perm_err:
            # alive 여부 불확실 — 검사 권한 없음
            _log.warning(
                "termux-vision: PID %d alive check failed (PermissionError): %s. "
                "Process may be alive but unverifiable.",
                pid, perm_err,
            )
            return {
                "pid": pid,
                "alive": None,
                "verified": False,
                "inspection_error": {
                    "code": "PROCESS_INSPECTION_PERMISSION_DENIED",
                    "message": str(perm_err),
                },
            }
        except OSError as os_err:
            _log.warning("termux-vision: PID %d os.kill check failed: %s", pid, os_err)
            return {
                "pid": pid,
                "alive": None,
                "verified": False,
                "inspection_error": {
                    "code": "PROCESS_INSPECTION_OS_ERROR",
                    "message": str(os_err),
                },
            }

    def doctor_full(self) -> dict:
        lite = self.doctor_lite()
        try:
            from termux_vision.cli.doctor import run_doctor
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_doctor(type("a", (), {})())
            lite["doctor_output"] = buf.getvalue()
        except Exception as e:
            lite["doctor_error"] = str(e)
        lite["doctor_level"] = "full"
        return lite

    def list_models(self) -> dict:
        reg_map = {m["model_id"]: m for m in self._model_reg.list_all()}
        try:
            from termux_vision.vlm.cache import ModelCacheManager
            mgr = ModelCacheManager()
            for item in (mgr.list_cached() if hasattr(mgr, "list_cached") else []):
                mid = item if isinstance(item, str) else item.get("model_id", str(item))
                if mid not in reg_map:
                    reg_map[mid] = {"model_id": mid, "state": "unverified",
                                    "note": "In cache but not verified by AMEVA registry"}
        except Exception as e:
            log_stderr(f"[vision] ModelCacheManager list failed: {e}")
        return {"models": list(reg_map.values()), "total": len(reg_map),
                "cache_dir": str(self._cache_dir)}

    def model_status(self, model_id: str | None = None) -> dict:
        if model_id:
            rec = self._model_reg.get(model_id)
            if rec is None: raise ModelNotFound(model_id)
            return {"model": rec}
        return self.list_models()

    def install_model(self, request: dict) -> dict:
        from ameva_component import ModelInstaller
        url = request.get("url", ""); filename = request.get("filename", "")
        sha256 = request.get("sha256", "")
        expected_bytes = int(request.get("expected_bytes", 0))
        model_id = request.get("model_id") or Path(filename).stem
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        installer = ModelInstaller(self.COMPONENT_ID, self._cache_dir, self._model_reg)
        return installer.install(url=url, filename=filename, sha256=sha256,
                                 expected_bytes=expected_bytes, model_id=model_id)

    async def activate_model(self, request: dict) -> dict:
        model_id = request.get("model_id", "")
        rec = self._model_reg.get(model_id)
        if rec is None: raise ModelNotFound(model_id)
        if ModelState.from_str(rec.get("state", "missing")) not in (ModelState.INSTALLED, ModelState.INACTIVE):
            raise ModelLoadFailed(model_id, f"State is '{rec.get('state')}'")
        with self._act_lock.acquire(timeout=60.0):
            self._model_reg.set_state(model_id, ModelState.ACTIVE)
            self._write_state()
        return {"activated": True, "model_id": model_id,
                "rollback": {"attempted": False, "succeeded": False}}

    async def deactivate_model(self, request: dict) -> dict:
        model_id = request.get("model_id", "")
        self._model_reg.set_state(model_id, ModelState.INACTIVE)
        self._write_state()
        return {"deactivated": True, "model_id": model_id}

    def list_instances(self) -> dict:
        instances = self._inst_reg.list_all()
        return {"instances": [i.to_dict() for i in instances], "total": len(instances)}

    async def start_instance(self, request: dict) -> dict:
        model_id = request.get("model_id", "")
        instance_id = request.get("instance_id") or f"vision-worker-{int(time.time())}"
        inst = InstanceStatus(
            instance_id=instance_id, component_id=self.COMPONENT_ID,
            model_id=model_id, state=InstanceState.HOT,
            active_jobs=0, queue_depth=0, max_concurrency=1,
            backend="cpu", started_at=time.time(), last_heartbeat=time.time(),
            last_error=None, control_mode=ControlMode.IN_PROCESS,
        )
        self._inst_reg.register(inst)
        self._write_state()
        # Phase 4: Heartbeat 시작 (Worker 시작 트리거)
        self._heartbeat.start()
        return {"instance_id": instance_id, "state": InstanceState.HOT.value}

    async def drain_instance(self, instance_id: str) -> dict:
        from ameva_component import InstanceNotFound
        if not self._inst_reg.get(instance_id): raise InstanceNotFound(instance_id)
        self._inst_reg.update_state(instance_id, InstanceState.DRAINING)
        return {"instance_id": instance_id, "state": InstanceState.DRAINING.value}

    async def stop_instance(self, instance_id: str) -> dict:
        from ameva_component import InstanceNotFound
        if not self._inst_reg.get(instance_id): raise InstanceNotFound(instance_id)
        self._inst_reg.update_state(instance_id, InstanceState.STOPPED)
        self._inst_reg.remove(instance_id)
        # Phase 4: Heartbeat 중단 (정상 종료 트리거)
        remaining = self._inst_reg.list_all()
        if not remaining:
            self._heartbeat.stop()
        else:
            self._write_state()
        return {"instance_id": instance_id, "state": InstanceState.STOPPED.value}

    def _write_state(self, *, ready: bool | None = None, last_error: str | None = None) -> None:
        ts = now_timestamps()
        hot = [i for i in self._inst_reg.list_all() if i.state == InstanceState.HOT]
        _, pid_alive = self._check_pid()
        _ready = True if ready is None else ready
        self._state_file.write({
            "protocol": "ameva-component-status/1", "component_id": self.COMPONENT_ID,
            "component_type": self.COMPONENT_TYPE, "version": self._get_version(),
            "ready": _ready, "degraded": not _ready, **ts,
            "active_models": [i.model_id for i in hot], "last_error": last_error,
        })

    async def analyze(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute on-device vision analysis (VLM multimodal QA/captioning, Canny, or face detection)."""
        task = request.get("task", "caption")
        image_path = request.get("image_path")
        if not image_path:
            return {"ok": False, "error": {"code": "IMAGE_PATH_MISSING", "message": "image_path is required"}}

        if task in ("caption", "vlm"):
            from termux_vision.vlm.api import load
            model_id = request.get("model_id") or "smolvlm-500m"
            prompt = request.get("prompt") or "Describe this image in detail."
            device = request.get("device") or "auto"
            quality = request.get("quality", "optimal")
            max_tokens = int(request.get("max_tokens", 150))

            try:
                with load(model_id=model_id, device=device) as engine:
                    res = engine.describe(image_path, prompt=prompt, quality=quality, max_tokens=max_tokens)
                    metrics_dict = res.metrics.to_dict() if res.metrics else {}
                    return {
                        "ok": True,
                        "task": task,
                        "result": {
                            "text": res.text,
                            "metrics": metrics_dict,
                            "model_id": model_id,
                        },
                        "fallback_used": False,
                        "requested_backend": device,
                        "executed_backend": res.metrics.backend if res.metrics else "unknown",
                    }
            except Exception as e:
                return {
                    "ok": False,
                    "error": {
                        "code": "VLM_INFERENCE_ERROR",
                        "message": str(e),
                    }
                }
        elif task == "canny":
            from termux_vision.io.loader import load_image, save_image
            from termux_vision.transforms.functional import to_grayscale
            from termux_vision.cv.filters import canny
            try:
                img = load_image(image_path)
                gray = to_grayscale(img)
                low = float(request.get("low", 40.0))
                high = float(request.get("high", 120.0))
                edges = canny(gray, low_threshold=low, high_threshold=high)
                output_path = request.get("output_path")
                if output_path:
                    save_image(edges, output_path)
                return {
                    "ok": True,
                    "task": "canny",
                    "result": {"output_path": output_path, "edge_pixels": int(edges.sum() > 0)},
                    "fallback_used": False,
                    "executed_backend": "cpu_neon",
                }
            except Exception as e:
                return {"ok": False, "error": {"code": "CANNY_ERROR", "message": str(e)}}
        elif task == "detect":
            from termux_vision.io.loader import load_image
            from termux_vision.detect.haar import detect_faces
            try:
                img = load_image(image_path)
                detections = detect_faces(img)
                return {
                    "ok": True,
                    "task": "detect",
                    "result": {
                        "count": len(detections),
                        "detections": [{"bbox": d.bbox.to_xywh(), "score": d.score} for d in detections]
                    },
                    "fallback_used": False,
                    "executed_backend": "cpu_neon",
                }
            except Exception as e:
                return {"ok": False, "error": {"code": "DETECT_ERROR", "message": str(e)}}
        else:
            return {"ok": False, "error": {"code": "UNSUPPORTED_TASK", "message": f"Task '{task}' is not supported"}}
