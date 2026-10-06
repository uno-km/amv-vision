import os
import subprocess
import urllib.request
import urllib.error
from typing import Dict, Tuple, Optional, List, Any
from ..errors import ModelDownloadError

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen2-vl-2b": {
        "text_url": "https://huggingface.co/second-state/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
        "text_file": "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
        "vision_url": "https://huggingface.co/second-state/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-vision-encoder.gguf",
        "vision_file": "Qwen2-VL-2B-Instruct-vision-encoder.gguf",
        "size_mb": 3440,
        "default_res": 384
    },
    "smolvlm-500m": {
        "text_url": "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf",
        "text_file": "SmolVLM-500M-Instruct-Q8_0.gguf",
        "vision_url": "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
        "vision_file": "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
        "size_mb": 545,
        "default_res": 384
    }
}

def is_known_remote_model(model_name: str) -> bool:
    clean = str(model_name).lower().strip()
    return clean in MODEL_REGISTRY or clean.startswith("hf:") or "/" in clean or clean.startswith("http://") or clean.startswith("https://")

SHARED_MODELS_DIR = os.path.expanduser("~/.cache/termux-ai/models")

def get_cache_dir() -> str:
    cache_dir = os.path.expanduser("~/.cache/termux-vision/models")
    legacy_dir = os.path.expanduser("~/.cache/vlm_models")
    if os.path.exists(SHARED_MODELS_DIR):
        return SHARED_MODELS_DIR
    if os.path.exists(legacy_dir) and not os.path.exists(cache_dir):
        return legacy_dir
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def download_file_stream(url: str, dest: str, progress_callback=None) -> str:
    """Streams a file download with partial file isolation."""
    partial_dest = dest + ".partial"
    if os.path.exists(partial_dest):
        os.remove(partial_dest)

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    try:
        cmd = ["curl", "-L", "-f", "--retry", "3", "-o", str(partial_dest), str(url)]
        res = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0 or not os.path.exists(partial_dest) or os.path.getsize(partial_dest) == 0:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (termux-vision)"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(partial_dest, "wb") as f:
                total = int(resp.info().get("Content-Length", 0))
                downloaded = 0
                while True:
                    chunk = resp.read(2 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(os.path.basename(dest), downloaded, total)

        if os.path.exists(dest):
            os.remove(dest)
        os.replace(partial_dest, dest)
        return dest
    except Exception as e:
        if os.path.exists(partial_dest):
            os.remove(partial_dest)
        raise ModelDownloadError(
            model_source=url,
            reason=f"Network download failure: {e}"
        )

def download_custom_url_or_hf(source: str, dest_dir: Optional[str] = None, progress_callback=None) -> str:
    """
    Downloads an arbitrary model file from direct HTTP(S) URL or Hugging Face repo:
    Example:
      - https://huggingface.co/owner/repo/resolve/main/model.gguf
      - hf:owner/repo:model.gguf
    """
    target_dir = dest_dir or get_cache_dir()
    os.makedirs(target_dir, exist_ok=True)

    if source.startswith("http://") or source.startswith("https://"):
        filename = os.path.basename(source.split("?")[0])
        dest_file = os.path.join(target_dir, filename)
        return download_file_stream(source, dest_file, progress_callback=progress_callback)

    if source.startswith("hf:") or ":" in source:
        clean = source[3:] if source.startswith("hf:") else source
        parts = clean.split(":")
        if len(parts) == 2:
            repo_id, filename = parts[0].strip(), parts[1].strip()
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            dest_file = os.path.join(target_dir, filename)
            return download_file_stream(url, dest_file, progress_callback=progress_callback)

    raise ModelDownloadError(
        model_source=source,
        reason="Unsupported URL or Hugging Face format. Expected https://... or hf:owner/repo:file.gguf"
    )

def download_vlm_model(
    model_name: str = "smolvlm-500m",
    progress_callback=None,
    dest_dir: Optional[str] = None,
    force: bool = False,
) -> Tuple[str, str]:
    """
    Downloads or retrieves cached VLM model weights (text GGUF and vision projector GGUF).
    """
    name_clean = model_name.lower().strip()
    if name_clean not in MODEL_REGISTRY:
        raise ModelDownloadError(
            model_source=model_name,
            reason=f"Unknown VLM model identifier. Available presets: {list(MODEL_REGISTRY.keys())}"
        )

    info = MODEL_REGISTRY[name_clean]
    target_dir = dest_dir if dest_dir else os.path.join(get_cache_dir(), name_clean)
    os.makedirs(target_dir, exist_ok=True)

    text_dest = os.path.join(target_dir, info["text_file"])
    vision_dest = os.path.join(target_dir, info["vision_file"])

    # Fallback search in shared model cache
    shared_text = os.path.join(SHARED_MODELS_DIR, info["text_file"])
    shared_vision = os.path.join(SHARED_MODELS_DIR, info["vision_file"])

    if not force:
        if os.path.isfile(shared_text) and os.path.getsize(shared_text) > 0:
            text_dest = shared_text
        if os.path.isfile(shared_vision) and os.path.getsize(shared_vision) > 0:
            vision_dest = shared_vision

    if force or not os.path.exists(text_dest) or os.path.getsize(text_dest) == 0:
        download_file_stream(info["text_url"], text_dest, progress_callback=progress_callback)

    if force or not os.path.exists(vision_dest) or os.path.getsize(vision_dest) == 0:
        download_file_stream(info["vision_url"], vision_dest, progress_callback=progress_callback)

    return text_dest, vision_dest
