"""
AMEVA Unified Model Downloader for termux-vision.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
from .hardware import get_unified_model_search_dirs
from .vlm.model_hub import MODEL_REGISTRY, download_vlm_model, get_cache_dir
from .vlm.manifest import verify_file_sha256

AVAILABLE_MODELS = {
    k: {
        "name": v.get("text_file", f"{k}.gguf"),
        "size_mb": v.get("size_mb", 500),
        "desc": f"{k} Vision-Language Model"
    }
    for k, v in MODEL_REGISTRY.items()
}

def resolve_model_path(model_name: str = "smolvlm-500m") -> Path:
    search_dirs = get_unified_model_search_dirs("vision")
    for d in search_dirs:
        cand = d / model_name
        if cand.exists():
            return cand.resolve()
    c_dir = Path(get_cache_dir()) / model_name
    if c_dir.exists():
        return c_dir.resolve()
    return Path(get_cache_dir()) / model_name

def download_model(model_name: str = "smolvlm-500m", output_dir: Optional[Path] = None, force: bool = False) -> Path:
    dest_str = str(output_dir) if output_dir else None
    text_path, _ = download_vlm_model(model_name, dest_dir=dest_str, force=force)
    return Path(text_path)

def list_models() -> List[Dict[str, Any]]:
    return [{"id": k, **v} for k, v in AVAILABLE_MODELS.items()]
