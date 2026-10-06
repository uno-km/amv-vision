import struct
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from ..errors import TermuxVisionError

MAX_CONTROL_MESSAGE_BYTES = 1 * 1024 * 1024  # 1 MB

class ProtocolError(TermuxVisionError):
    """Raised when TVRP framing or payload validation fails."""
    pass

@dataclass(frozen=True)
class TVRPRequest:
    protocol_version: int
    request_id: str
    op: str
    model_id: str
    image_path: str
    prompt: str
    options: Dict[str, Any]

    def serialize(self) -> bytes:
        payload = {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "op": self.op,
            "model_id": self.model_id,
            "image_path": self.image_path,
            "prompt": self.prompt,
            "options": self.options
        }
        json_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        length = len(json_bytes)
        if length > MAX_CONTROL_MESSAGE_BYTES:
            raise ProtocolError(f"Request payload size ({length} bytes) exceeds limit ({MAX_CONTROL_MESSAGE_BYTES} bytes)")
        header = struct.pack("<I", length)
        return header + json_bytes

    @classmethod
    def deserialize(cls, data: bytes) -> "TVRPRequest":
        if len(data) < 4:
            raise ProtocolError("Incomplete TVRP frame header (< 4 bytes)")
        length = struct.unpack("<I", data[:4])[0]
        if length > MAX_CONTROL_MESSAGE_BYTES:
            raise ProtocolError(f"Framed length ({length} bytes) exceeds limit ({MAX_CONTROL_MESSAGE_BYTES} bytes)")
        if len(data) < 4 + length:
            raise ProtocolError("Incomplete TVRP frame payload")
        
        json_bytes = data[4:4 + length]
        try:
            payload = json.loads(json_bytes.decode("utf-8"))
        except Exception as e:
            raise ProtocolError(f"Malformed JSON payload: {e}")

        if payload.get("protocol_version") != 1:
            raise ProtocolError(f"Unsupported protocol version: {payload.get('protocol_version')}")

        return cls(
            protocol_version=payload["protocol_version"],
            request_id=payload["request_id"],
            op=payload["op"],
            model_id=payload["model_id"],
            image_path=payload["image_path"],
            prompt=payload["prompt"],
            options=payload.get("options", {})
        )

@dataclass(frozen=True)
class TVRPResponse:
    protocol_version: int
    request_id: str
    status: str  # "ok" | "error"
    result: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    warnings: Tuple[str, ...] = ()

    def serialize(self) -> bytes:
        payload = {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "status": self.status,
            "result": self.result,
            "metrics": self.metrics,
            "error": self.error,
            "warnings": list(self.warnings)
        }
        json_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        length = len(json_bytes)
        if length > MAX_CONTROL_MESSAGE_BYTES:
            raise ProtocolError(f"Response payload size ({length} bytes) exceeds limit ({MAX_CONTROL_MESSAGE_BYTES} bytes)")
        header = struct.pack("<I", length)
        return header + json_bytes

    @classmethod
    def deserialize(cls, data: bytes) -> "TVRPResponse":
        if len(data) < 4:
            raise ProtocolError("Incomplete TVRP frame header (< 4 bytes)")
        length = struct.unpack("<I", data[:4])[0]
        if length > MAX_CONTROL_MESSAGE_BYTES:
            raise ProtocolError(f"Framed length ({length} bytes) exceeds limit ({MAX_CONTROL_MESSAGE_BYTES} bytes)")
        if len(data) < 4 + length:
            raise ProtocolError("Incomplete TVRP frame payload")
        
        json_bytes = data[4:4 + length]
        try:
            payload = json.loads(json_bytes.decode("utf-8"))
        except Exception as e:
            raise ProtocolError(f"Malformed JSON payload: {e}")

        return cls(
            protocol_version=payload.get("protocol_version", 1),
            request_id=payload.get("request_id", ""),
            status=payload.get("status", "error"),
            result=payload.get("result"),
            metrics=payload.get("metrics"),
            error=payload.get("error"),
            warnings=tuple(payload.get("warnings", []))
        )
