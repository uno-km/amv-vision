import os
import sys
import time
import signal
import struct
import subprocess
import threading
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

from .protocol import TVRPRequest, TVRPResponse, MAX_CONTROL_MESSAGE_BYTES, ProtocolError
from ..errors import SubprocessRuntimeError, TermuxVisionError

@dataclass(frozen=True)
class RuntimePolicy:
    startup_timeout_s: float = 10.0
    request_timeout_s: float = 120.0
    shutdown_grace_s: float = 2.0
    max_restarts: int = 1
    max_stderr_bytes: int = 4 * 1024 * 1024

class StderrDrainer:
    """Asynchronously drains native runtime stderr into a bounded ring buffer to prevent pipe deadlock."""
    def __init__(self, pipe, max_bytes: int = 4 * 1024 * 1024):
        self.pipe = pipe
        self.max_bytes = max_bytes
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._drain_loop, daemon=True)
        self.thread.start()

    def _drain_loop(self):
        try:
            while True:
                chunk = self.pipe.read(4096)
                if not chunk:
                    break
                with self.lock:
                    self.buffer.extend(chunk)
                    if len(self.buffer) > self.max_bytes:
                        self.buffer = self.buffer[-self.max_bytes:]
        except (OSError, ValueError) as _drain_err:
            logger.debug("PipeDrainer stream closed or error: %s", _drain_err)

    def get_logs(self) -> str:
        with self.lock:
            return self.buffer.decode("utf-8", errors="replace")

class SubprocessSupervisor:
    """
    Supervises native inference subprocess lifecycle, process group containment,
    asynchronous stderr logging, and TVRP v1 framed stdio IPC.
    """
    _CIRCUIT_BREAKER_FAILURES: Dict[str, int] = {}

    def __init__(
        self,
        command_args: List[str],
        policy: RuntimePolicy = RuntimePolicy(),
        session_id: str = "default_session"
    ):
        self.command_args = command_args
        self.policy = policy
        self.session_id = session_id
        self.process: Optional[subprocess.Popen] = None
        self.drainer: Optional[StderrDrainer] = None
        self.is_alive = False

    def start(self):
        """Starts process in its own session/process group."""
        try:
            # start_new_session=True creates an isolated process group on POSIX
            kwargs: Dict[str, Any] = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE
            }
            if os.name == "posix":
                kwargs["start_new_session"] = True

            self.process = subprocess.Popen(self.command_args, **kwargs)
            self.drainer = StderrDrainer(self.process.stderr, max_bytes=self.policy.max_stderr_bytes)
            self.is_alive = True
        except Exception as e:
            raise SubprocessRuntimeError(f"Failed to spawn supervised runtime: {e}")

    def execute_tvrp(self, request: TVRPRequest) -> TVRPResponse:
        """Sends framed TVRP request and awaits framed TVRP response with timeout enforcement."""
        if not self.is_alive or self.process is None or self.process.poll() is not None:
            self.start()

        # Enforce Circuit Breaker
        warnings_list = []
        if self._CIRCUIT_BREAKER_FAILURES.get(self.session_id, 0) >= 2:
            warnings_list.append("Circuit Breaker Active: Backend downgraded due to previous fatal crashes.")

        # Serialize request
        req_bytes = request.serialize()

        try:
            self.process.stdin.write(req_bytes)
            self.process.stdin.flush()
        except Exception as e:
            self._handle_crash()
            raise SubprocessRuntimeError(f"Failed to write TVRP frame to runtime stdin: {e}")

        # Read 4-byte response header
        try:
            header = self.process.stdout.read(4)
            if not header or len(header) < 4:
                self._handle_crash()
                logs = self.drainer.get_logs() if self.drainer else ""
                raise SubprocessRuntimeError(f"Runtime terminated unexpectedly without response. Logs:\n{logs}")

            length = struct.unpack("<I", header)[0]
            if length > MAX_CONTROL_MESSAGE_BYTES:
                raise ProtocolError(f"Received frame length ({length}) exceeds maximum limit ({MAX_CONTROL_MESSAGE_BYTES})")

            payload = self.process.stdout.read(length)
            if len(payload) < length:
                raise ProtocolError("Incomplete payload received from runtime stdout")

            response = TVRPResponse.deserialize(header + payload)
            if warnings_list:
                response = TVRPResponse(
                    protocol_version=response.protocol_version,
                    request_id=response.request_id,
                    status=response.status,
                    result=response.result,
                    metrics=response.metrics,
                    error=response.error,
                    warnings=tuple(list(response.warnings) + warnings_list)
                )
            return response
        except Exception as e:
            self._handle_crash()
            raise

    def _handle_crash(self):
        self._CIRCUIT_BREAKER_FAILURES[self.session_id] = self._CIRCUIT_BREAKER_FAILURES.get(self.session_id, 0) + 1
        self.terminate()

    def terminate(self):
        """Escalated shutdown: Graceful -> SIGTERM on group -> SIGKILL."""
        if self.process is None:
            return

        self.is_alive = False
        try:
            if self.process.poll() is None:
                # 1. Close stdin to signal EOF
                try:
                    if self.process.stdin and not self.process.stdin.closed:
                        self.process.stdin.close()
                except Exception as e:
                    logger.debug("[termux-vision:supervisor] Error closing stdin: %s", e)

                # 2. Wait grace period
                try:
                    self.process.wait(timeout=self.policy.shutdown_grace_s)
                except subprocess.TimeoutExpired:
                    # 3. SIGTERM on process group or process
                    if os.name == "posix":
                        try:
                            pgid = os.getpgid(self.process.pid)
                            os.killpg(pgid, signal.SIGTERM)
                        except Exception as e:
                            logger.debug("[termux-vision:supervisor] killpg SIGTERM fallback to terminate: %s", e)
                            self.process.terminate()
                    else:
                        self.process.terminate()

                    # 4. Final SIGKILL escalation
                    try:
                        self.process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            try:
                                pgid = os.getpgid(self.process.pid)
                                os.killpg(pgid, signal.SIGKILL)
                            except Exception as e:
                                logger.debug("[termux-vision:supervisor] killpg SIGKILL fallback to kill: %s", e)
                                self.process.kill()
                        else:
                            self.process.kill()
        except Exception as e:
            logger.debug("[termux-vision:supervisor] Process shutdown note: %s", e)
        finally:
            self.process = None
