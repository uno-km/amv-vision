import os
import shutil
import urllib.request
import urllib.error
import json
import re
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from .manifest import ModelManifest, ArtifactInfo, verify_file_sha256
from ..errors import (
    ModelNotFoundError,
    ModelCorruptedError,
    NoInstalledModelsError,
    ModelArtifactsMissingError,
    ModelDownloadError
)

MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

def validate_model_id(model_id: str) -> str:
    """Validates model_id against path traversal and special characters."""
    clean_id = str(model_id).strip()
    if not MODEL_ID_PATTERN.match(clean_id) or ".." in clean_id or "/" in clean_id or "\\" in clean_id:
        raise ValueError(f"Invalid model ID: '{model_id}'. Must match pattern ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$ and contain no path traversal characters.")
    return clean_id

def require_path_within_root(target_path: str, allowed_root: str, *, allow_root_itself: bool = False) -> Path:
    """Ensures target_path resolves strictly within allowed_root to prevent symlink and traversal attacks."""
    target = Path(target_path).expanduser().resolve()
    root = Path(allowed_root).expanduser().resolve()

    if target == root and not allow_root_itself:
        raise ValueError(f"Refusing operation directly on root itself: {root}")

    if target != root and root not in target.parents:
        raise ValueError(f"Path escape detected! Target '{target}' is not within allowed root '{root}'.")

    return target

class ModelState(Enum):
    ABSENT = "ABSENT"
    DOWNLOADING = "DOWNLOADING"
    VERIFYING = "VERIFYING"
    READY = "READY"
    IN_USE = "IN_USE"
    QUARANTINED = "QUARANTINED"

CATALOG: Dict[str, ModelManifest] = {
    "smolvlm-500m-q4": ModelManifest(
        schema_version=1,
        model_id="smolvlm-500m-q4",
        adapter="smolvlm",
        tier="M",
        estimated_memory_mb=750,
        context_limit=2048,
        preferred_resolution=384,
        artifacts=[
            ArtifactInfo(
                role="language_model",
                filename="SmolVLM-500M-Instruct-Q8_0.gguf",
                size_bytes=436_806_912,
                sha256="",
                download_url="https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf"
            ),
            ArtifactInfo(
                role="vision_projector",
                filename="mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
                size_bytes=108_783_360,
                sha256="",
                download_url="https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"
            )
        ]
    ),
    "qwen2-vl-2b-q4": ModelManifest(
        schema_version=1,
        model_id="qwen2-vl-2b-q4",
        adapter="qwen2vl",
        tier="L",
        estimated_memory_mb=2100,
        context_limit=2048,
        preferred_resolution=384,
        artifacts=[
            ArtifactInfo(
                role="language_model",
                filename="Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
                size_bytes=986046944,
                sha256="",
                download_url="https://huggingface.co/second-state/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
            ),
            ArtifactInfo(
                role="vision_projector",
                filename="Qwen2-VL-2B-Instruct-vision-encoder.gguf",
                size_bytes=2600000000,
                sha256="",
                download_url="https://huggingface.co/second-state/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-vision-encoder.gguf"
            )
        ]
    )
}

class ModelCacheManager:
    """
    On-device model cache with strict validation, custom/싸제 model discovery,
    and user-guided dynamic error contracts.
    """
    def __init__(self, cache_root: Optional[str] = None):
        self.cache_root = cache_root or os.path.expanduser("~/.cache/termux-vision")
        self.legacy_cache = os.path.expanduser("~/.cache/vlm_models")
        self.models_dir = os.path.join(self.cache_root, "models")
        self.downloads_dir = os.path.join(self.cache_root, "downloads")
        self.locks_dir = os.path.join(self.cache_root, "locks")
        self.quarantine_dir = os.path.join(self.cache_root, "quarantine")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.downloads_dir, exist_ok=True)
        os.makedirs(self.locks_dir, exist_ok=True)
        os.makedirs(self.quarantine_dir, exist_ok=True)

    def _sanitize_model_id(self, model_id: str) -> str:
        clean = str(model_id).strip()
        if not clean or ".." in clean or "/" in clean or "\\" in clean:
            raise ValueError(f"Invalid or unsafe model_id: '{model_id}'")
        return clean

    def get_model_dir(self, model_id: str) -> str:
        clean_id = self._sanitize_model_id(model_id)
        model_dir = os.path.abspath(os.path.join(self.models_dir, clean_id))
        if not model_dir.startswith(os.path.abspath(self.models_dir)):
            raise ValueError(f"Model directory escapes cache root: '{model_id}'")
        if os.path.exists(model_dir):
            return model_dir
        if clean_id == "qwen2-vl-2b-q4" and os.path.exists(self.legacy_cache):
            expected = [
                "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
                "Qwen2-VL-2B-Instruct-vision-encoder.gguf",
            ]
            if all(
                os.path.isfile(os.path.join(self.legacy_cache, name))
                and os.path.getsize(os.path.join(self.legacy_cache, name)) > 10_000_000
                for name in expected
            ):
                return self.legacy_cache
    def get_state(self, model_id: str) -> ModelState:
        try:
            mdir = self.get_model_dir(model_id)
        except ValueError as e:
            raise ModelCorruptedError(f"Invalid model_id or directory path structure for '{model_id}': {e}") from e

        if mdir == self.legacy_cache:
            return ModelState.READY

        if not os.path.exists(mdir):
            return ModelState.ABSENT

        quarantine_marker = os.path.join(mdir, "QUARANTINED")
        if os.path.exists(quarantine_marker):
            return ModelState.QUARANTINED

        # Check for formal manifest and ready marker
        ready_marker = os.path.join(mdir, "READY")
        manifest_file = os.path.join(mdir, "manifest.json")
        if os.path.exists(ready_marker) and os.path.exists(manifest_file):
            return ModelState.READY

        # Check for custom model directory (has *.gguf and mmproj*.gguf with valid sizes > 10MB)
        if os.path.isdir(mdir):
            files = os.listdir(mdir)
            ggufs = [
                f for f in files
                if f.endswith(".gguf") and os.path.getsize(os.path.join(mdir, f)) > 0
            ]
            vision_ggufs = [f for f in ggufs if "mmproj" in f.lower() or "encoder" in f.lower() or "projector" in f.lower()]
            text_ggufs = [f for f in ggufs if f not in vision_ggufs]
            if not vision_ggufs and len(ggufs) >= 2:
                text_ggufs = [ggufs[0]]
                vision_ggufs = [ggufs[1]]
            if text_ggufs and vision_ggufs:
                return ModelState.READY

        return ModelState.DOWNLOADING

    def is_model_installed(self, model_id: str) -> bool:
        return self.get_state(model_id) == ModelState.READY

    def get_installed_model_ids(self) -> List[str]:
        return [
            item["model_id"]
            for item in self.list_installed()
            if item.get("state") == ModelState.READY.value
        ]

    def require_any_installed_model(self) -> List[Dict[str, Any]]:
        installed = self.list_installed()
        ready = [
            item
            for item in installed
            if item.get("state") == ModelState.READY.value
        ]
        if not ready:
            raise NoInstalledModelsError(
                catalog_models=tuple(sorted(CATALOG.keys()))
            )
        return ready

    def require_installed_model(self, model_id: str) -> Tuple[ModelManifest, str]:
        # 1. Direct file path support (Custom 싸제 GGUF)
        expanded_path = os.path.abspath(os.path.expanduser(str(model_id)))
        if os.path.isfile(expanded_path):
            model_dir = os.path.dirname(expanded_path)
            model_filename = os.path.basename(expanded_path)
            mmproj_candidates = [
                f for f in os.listdir(model_dir)
                if f.endswith(".gguf") and ("mmproj" in f.lower() or "vision" in f.lower() or "encoder" in f.lower())
            ]
            vision_filename = mmproj_candidates[0] if mmproj_candidates else ""
            
            manifest = ModelManifest(
                schema_version=1,
                model_id=os.path.splitext(model_filename)[0],
                adapter="smolvlm" if "smol" in model_filename.lower() else "qwen2vl",
                tier="CUSTOM",
                estimated_memory_mb=1500,
                context_limit=2048,
                preferred_resolution=384,
                artifacts=[
                    ArtifactInfo(
                        role="language_model",
                        filename=model_filename,
                        size_bytes=os.path.getsize(expanded_path),
                        sha256="",
                        download_url=""
                    ),
                    ArtifactInfo(
                        role="vision_projector",
                        filename=vision_filename,
                        size_bytes=os.path.getsize(os.path.join(model_dir, vision_filename)) if vision_filename else 0,
                        sha256="",
                        download_url=""
                    )
                ]
            )
            return manifest, model_dir

        # 2. Installed models check
        installed = self.list_installed()
        installed_ids = [item["model_id"] for item in installed if item.get("state") == ModelState.READY.value]

        if not installed_ids:
            raise NoInstalledModelsError(
                catalog_models=tuple(sorted(CATALOG.keys()))
            )

        if model_id not in installed_ids:
            raise ModelNotFoundError(
                model_id=model_id,
                available_local_models=installed_ids,
                catalog_models=sorted(CATALOG.keys())
            )

        manifest = self.get_manifest(model_id)
        model_dir = self.get_model_dir(model_id)

        missing = tuple(
            artifact.filename
            for artifact in manifest.artifacts
            if not os.path.isfile(os.path.join(model_dir, artifact.filename))
        )

        if missing:
            raise ModelArtifactsMissingError(
                model_id=model_id,
                missing_artifacts=missing
            )

        return manifest, model_dir

    def get_manifest(self, model_id: str) -> ModelManifest:
        if self.is_model_installed(model_id):
            mdir = self.get_model_dir(model_id)
            mpath = os.path.join(mdir, "manifest.json")
            if os.path.exists(mpath):
                with open(mpath, "r", encoding="utf-8") as f:
                    return ModelManifest.from_dict(json.load(f))
            elif model_id in CATALOG:
                return CATALOG[model_id]
            else:
                # Custom directory discovered model
                files = os.listdir(mdir)
                ggufs = [f for f in files if f.endswith(".gguf")]
                vision_files = [f for f in ggufs if "mmproj" in f.lower() or "encoder" in f.lower() or "projector" in f.lower()]
                text_files = [f for f in ggufs if f not in vision_files]
                if not vision_files and len(ggufs) >= 2:
                    text_files = [ggufs[0]]
                    vision_files = [ggufs[1]]

                text_name = text_files[0] if text_files else "model.gguf"
                vision_name = vision_files[0] if vision_files else "mmproj.gguf"
                
                text_size = os.path.getsize(os.path.join(mdir, text_name)) if os.path.exists(os.path.join(mdir, text_name)) else 0
                vision_size = os.path.getsize(os.path.join(mdir, vision_name)) if os.path.exists(os.path.join(mdir, vision_name)) else 0

                return ModelManifest(
                    schema_version=1,
                    model_id=model_id,
                    adapter="smolvlm" if "smol" in model_id.lower() else "qwen2vl",
                    tier="CUSTOM",
                    estimated_memory_mb=max(500, int((text_size + vision_size) / (1024 * 1024) * 1.3)),
                    context_limit=2048,
                    preferred_resolution=384,
                    artifacts=[
                        ArtifactInfo(
                            role="language_model",
                            filename=text_name,
                            size_bytes=text_size,
                            sha256="",
                            download_url=""
                        ),
                        ArtifactInfo(
                            role="vision_projector",
                            filename=vision_name,
                            size_bytes=vision_size,
                            sha256="",
                            download_url=""
                        )
                    ]
                )
        elif model_id in CATALOG:
            return CATALOG[model_id]
        
        installed_ids = [m["model_id"] for m in self.list_installed()]
        raise ModelNotFoundError(
            model_id=model_id,
            available_local_models=installed_ids,
            catalog_models=sorted(CATALOG.keys())
        )

    def install(self, model_id: str, progress_callback=None, verify_sha256: bool = True) -> ModelManifest:
        clean_id = self._sanitize_model_id(model_id)
        if clean_id not in CATALOG:
            from .model_hub import download_vlm_model, is_known_remote_model
            if is_known_remote_model(clean_id):
                download_vlm_model(clean_id)
                return self.get_manifest(clean_id)

            installed_ids = [m["model_id"] for m in self.list_installed()]
            raise ModelDownloadError(
                model_source=clean_id,
                reason=f"Model '{clean_id}' is not in the catalog and could not be resolved from Hugging Face.",
                available_local=installed_ids
            )

        manifest = CATALOG[clean_id]
        target_dir = os.path.join(self.models_dir, clean_id)
        os.makedirs(target_dir, exist_ok=True)

        for art in manifest.artifacts:
            dest_file = os.path.join(target_dir, art.filename)
            if os.path.exists(dest_file):
                actual_sz = os.path.getsize(dest_file)
                size_valid = (art.size_bytes == 0) or (actual_sz == art.size_bytes) or (abs(actual_sz - art.size_bytes) < 4096)
                hash_valid = True
                if verify_sha256 and art.sha256 and len(art.sha256) == 64:
                    hash_valid = verify_file_sha256(dest_file, art.sha256)

                if hash_valid and size_valid and actual_sz > 0:
                    continue
                else:
                    corrupt_dest = os.path.join(self.quarantine_dir, f"{clean_id}-{art.filename}.corrupt")
                    if os.path.exists(corrupt_dest):
                        os.remove(corrupt_dest)
                    os.replace(dest_file, corrupt_dest)

            partial_file = os.path.join(self.downloads_dir, f"{art.filename}.partial")
            if os.path.exists(partial_file):
                os.remove(partial_file)

            try:
                req = urllib.request.Request(art.download_url, headers={"User-Agent": "Mozilla/5.0 (termux-vision)"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(partial_file, "wb") as f:
                    total = int(resp.info().get("Content-Length", 0))
                    downloaded = 0
                    while True:
                        chunk = resp.read(2 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(art.filename, downloaded, total)
            except Exception as e:
                if os.path.exists(partial_file):
                    os.remove(partial_file)
                installed_ids = [m["model_id"] for m in self.list_installed()]
                raise ModelDownloadError(
                    model_source=art.download_url,
                    reason=f"Download failed: {e}",
                    available_local=installed_ids
                )

            if verify_sha256 and art.sha256 and len(art.sha256) == 64:
                if not verify_file_sha256(partial_file, art.sha256):
                    os.remove(partial_file)
                    self.quarantine(clean_id, f"Downloaded file checksum mismatch for {art.filename}")
                    raise ModelCorruptedError(f"SHA-256 validation failed for downloaded artifact: {art.filename}")

            if os.path.exists(dest_file):
                os.remove(dest_file)
            shutil.move(partial_file, dest_file)

        quarantine_marker = os.path.join(target_dir, "QUARANTINED")
        if os.path.exists(quarantine_marker):
            os.remove(quarantine_marker)

        manifest_path = os.path.join(target_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        ready_path = os.path.join(target_dir, "READY")
        with open(ready_path, "w", encoding="utf-8") as f:
            f.write("OK\n")

        return manifest

    def verify_integrity(self, model_id: str) -> Dict[str, Any]:
        """Calculates actual size and SHA-256 of all model artifacts and returns truthful status."""
        mdir = self.get_model_dir(model_id)
        if not os.path.exists(mdir):
            return {"status": "ABSENT", "details": "Directory does not exist"}

        manifest = self.get_manifest(model_id)
        results = {}
        all_found = True

        for art in manifest.artifacts:
            fpath = os.path.join(mdir, art.filename)
            if not os.path.exists(fpath):
                results[art.filename] = {
                    "found": False,
                    "status": "MISSING",
                    "size_verified": False,
                    "sha256_verified": False,
                    "verified": False
                }
                all_found = False
                continue

            actual_size = os.path.getsize(fpath)
            size_ok = (art.size_bytes == 0) or (actual_size == art.size_bytes) or (abs(actual_size - art.size_bytes) < 4096)
            
            hash_ok = None
            if art.sha256 and len(art.sha256) == 64:
                hash_ok = verify_file_sha256(fpath, art.sha256)

            if not size_ok or (hash_ok is False):
                item_status = "CORRUPTED"
                is_item_verified = False
            elif hash_ok is None:
                item_status = "UNVERIFIED"
                is_item_verified = False
            else:
                item_status = "VERIFIED"
                is_item_verified = True

            results[art.filename] = {
                "found": True,
                "status": item_status,
                "size_bytes": actual_size,
                "expected_size": art.size_bytes,
                "size_verified": size_ok,
                "sha256_verified": hash_ok,
                "verified": is_item_verified
            }

        statuses = [res["status"] for res in results.values()]
        if "CORRUPTED" in statuses:
            status = "CORRUPTED"
        elif not all_found or "MISSING" in statuses:
            status = "INCOMPLETE"
        elif "UNVERIFIED" in statuses:
            status = "UNVERIFIED"
        else:
            status = "VERIFIED"

        return {
            "status": status,
            "artifacts": results
        }

    def quarantine(self, model_id: str, reason: str = ""):
        mdir = os.path.join(self.models_dir, model_id)
        if os.path.exists(mdir):
            q_path = os.path.join(mdir, "QUARANTINED")
            with open(q_path, "w", encoding="utf-8") as f:
                f.write(f"REASON: {reason}\n")

    def list_installed(self) -> List[Dict[str, Any]]:
        results = []
        if os.path.exists(self.models_dir):
            for name in os.listdir(self.models_dir):
                mdir = os.path.join(self.models_dir, name)
                if os.path.isdir(mdir) and self.is_model_installed(name):
                    try:
                        manifest = self.get_manifest(name)
                        total_size = sum(
                            os.path.getsize(os.path.join(mdir, a.filename))
                            for a in manifest.artifacts
                            if os.path.exists(os.path.join(mdir, a.filename))
                        )
                        results.append({
                            "model_id": name,
                            "adapter": manifest.adapter,
                            "tier": manifest.tier,
                            "state": self.get_state(name).value,
                            "size_mb": round(total_size / (1024 * 1024), 2),
                            "path": mdir
                        })
                    except Exception as _manifest_err:
                        logger.debug("Failed scanning cache entry '%s': %s", name, _manifest_err)

        if os.path.exists(self.legacy_cache):
            for k, manifest in CATALOG.items():
                if k not in [r["model_id"] for r in results]:
                    all_found = all(
                        os.path.exists(os.path.join(self.legacy_cache, a.filename)) and
                        os.path.getsize(os.path.join(self.legacy_cache, a.filename)) > 0
                        for a in manifest.artifacts
                    )
                    if all_found:
                        total_size = sum(
                            os.path.getsize(os.path.join(self.legacy_cache, a.filename))
                            for a in manifest.artifacts
                        )
                        results.append({
                            "model_id": k,
                            "adapter": manifest.adapter,
                            "tier": manifest.tier,
                            "state": ModelState.READY.value,
                            "size_mb": round(total_size / (1024 * 1024), 2),
                            "path": self.legacy_cache
                        })
        return results

    def remove(self, model_id: str) -> bool:
        clean_id = self._sanitize_model_id(model_id)
        mdir = os.path.join(self.models_dir, clean_id)
        if os.path.exists(mdir):
            shutil.rmtree(mdir)
            return True
        return False
