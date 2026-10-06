from dataclasses import dataclass, field
import hashlib
import json
import os
from typing import List, Dict, Any, Optional
from ..errors import ModelCorruptedError

@dataclass(frozen=True)
class ArtifactInfo:
    role: str # e.g. "language_model", "vision_projector"
    filename: str
    size_bytes: int
    sha256: str
    download_url: str

@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_id: str
    adapter: str
    tier: str # "S", "M", "L"
    estimated_memory_mb: int
    context_limit: int
    preferred_resolution: int
    artifacts: List[ArtifactInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "adapter": self.adapter,
            "tier": self.tier,
            "estimated_memory_mb": self.estimated_memory_mb,
            "context_limit": self.context_limit,
            "preferred_resolution": self.preferred_resolution,
            "artifacts": [
                {
                    "role": a.role,
                    "filename": a.filename,
                    "size_bytes": a.size_bytes,
                    "sha256": a.sha256,
                    "download_url": a.download_url
                }
                for a in self.artifacts
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelManifest":
        artifacts = [
            ArtifactInfo(
                role=a["role"],
                filename=a["filename"],
                size_bytes=a["size_bytes"],
                sha256=a["sha256"],
                download_url=a["download_url"]
            )
            for a in data.get("artifacts", [])
        ]
        return cls(
            schema_version=data.get("schema_version", 1),
            model_id=data["model_id"],
            adapter=data["adapter"],
            tier=data.get("tier", "M"),
            estimated_memory_mb=data.get("estimated_memory_mb", 1000),
            context_limit=data.get("context_limit", 2048),
            preferred_resolution=data.get("preferred_resolution", 384),
            artifacts=artifacts
        )

def verify_file_sha256(file_path: str, expected_sha256: str, block_size: int = 2 * 1024 * 1024) -> bool:
    """
    Computes streaming SHA-256 and compares in constant time.
    """
    if not os.path.exists(file_path):
        return False
    
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            hasher.update(chunk)
    
    computed = hasher.hexdigest().lower()
    return computed == expected_sha256.lower()
