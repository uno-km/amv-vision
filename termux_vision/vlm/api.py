# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import os
import tempfile
from typing import Union, Dict, Any, Optional, Sequence
import numpy as np

from .cache import ModelCacheManager, CATALOG
from .manifest import ModelManifest, ArtifactInfo
from .memory import MemoryEstimate, check_memory_admission, acquire_engine_lock, release_engine_lock
from .runtime.subprocess import SubprocessVLMRuntime
from .runtime.resolver import RuntimeInfo, resolve_llama_cli
from .result import VLMResult
from ..io.loader import load_image, save_image
from ..transforms.functional import resize
from ..errors import (
    InsufficientMemoryError,
    ModelNotFoundError,
    NoInstalledModelsError,
    VulkanNotAvailableError
)

class VLMContext:
    """
    Context manager wrapper for loaded VLM inference engine with full parameter accessibility
    and strict parameter boundary validation (Requirement 3 & 3-1).
    """
    def __init__(
        self,
        runtime: SubprocessVLMRuntime,
        manifest: ModelManifest,
        warning_msg: Optional[str] = None
    ):
        self.runtime = runtime
        self.manifest = manifest
        self.warning_msg = warning_msg
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.runtime.close()
            finally:
                release_engine_lock()

    def describe(
        self,
        image: Union[str, np.ndarray],
        prompt: Optional[str] = None,
        max_tokens: int = 150,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = 1.2,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None,
        stop_tokens: Optional[Sequence[str]] = None,
        quality: str = "optimal",
        max_dim: Optional[int] = None
    ) -> VLMResult:
        if self._closed:
            raise RuntimeError("Cannot execute inference on closed VLMContext.")

        # Strict Null / Zero / Range Guard (Requirement 3-1: No Dangerous Silent Fallbacks)
        if image is None:
            raise ValueError("Parameter 'image' cannot be null/None. A valid file path or NumPy ndarray must be provided.")
        if isinstance(image, str) and not image.strip():
            raise ValueError("Parameter 'image' cannot be an empty string.")

        if max_tokens is None or max_tokens <= 0:
            raise ValueError(f"Parameter 'max_tokens' must be a positive integer > 0. Received: {max_tokens}")

        if temperature is None or temperature < 0.0:
            raise ValueError(f"Parameter 'temperature' must be a non-negative float >= 0.0. Received: {temperature}")

        if top_p is not None and (top_p <= 0.0 or top_p > 1.0):
            raise ValueError(f"Parameter 'top_p' must be within range (0.0, 1.0]. Received: {top_p}")

        if top_k is not None and top_k <= 0:
            raise ValueError(f"Parameter 'top_k' must be a positive integer >= 1. Received: {top_k}")

        if repeat_penalty is not None and repeat_penalty < 0.0:
            raise ValueError(f"Parameter 'repeat_penalty' cannot be negative. Received: {repeat_penalty}")

        default_prompt = "Describe the contents, objects, and visual scene of this image in detail."
        actual_prompt = prompt if prompt is not None else default_prompt
        if not actual_prompt.strip():
            raise ValueError("Parameter 'prompt' cannot be an empty string.")

        from ..transforms.scale import prepare_image_for_inference
        target_file, is_temporary = prepare_image_for_inference(image, quality=quality, max_dim=max_dim)
        temp_path = target_file if is_temporary else None

        try:
            result = self.runtime.execute(
                image_path=target_file,
                prompt=actual_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                seed=seed,
                system_prompt=system_prompt,
                stop_tokens=stop_tokens
            )
            if self.warning_msg:
                result = VLMResult(
                    text=result.text,
                    finish_reason=result.finish_reason,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    word_count=result.word_count,
                    metrics=result.metrics,
                    warnings=tuple(list(result.warnings) + [self.warning_msg])
                )
            return result
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as _rm_err:
                    import logging
                    logging.getLogger(__name__).debug("temp image cleanup failed (%s): %s", temp_path, _rm_err)

    def ask(self, image: Union[str, np.ndarray], question: str, **kwargs) -> str:
        if question is None or not question.strip():
            raise ValueError("Parameter 'question' cannot be null or empty.")
        res = self.describe(image, prompt=question, **kwargs)
        return res.text

def load(
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
    device: str = "auto",
    threads: Union[int, str] = "auto",
    memory_policy: str = "warn",
    memory_budget_mb: Optional[int] = None,
    fallback: Optional[bool] = None,
    allow_download: bool = False,
    cache_root: Optional[str] = None,
    runtime_path: Optional[str] = None,
    mmproj_path: Optional[str] = None,
    context_limit: Optional[int] = None,
    ngl: Optional[int] = None,
    rpc: Optional[str] = None,
    tensor_split: Optional[str] = None,
    cluster_rpc_servers: Optional[Union[str, Sequence[str]]] = None,
    cluster_split_mode: Optional[str] = None,
    cluster_tensor_split: Optional[str] = None,
    cluster_vram_budget: Optional[Dict[str, int]] = None,
) -> VLMContext:
    """
    Loads a VLM inference engine with clean separation between catalog model_id and custom model_path.
    """
    if model_id is not None and model_path is not None:
        raise ValueError(
            "Conflicting parameters: Cannot specify both 'model_id' and 'model_path'. "
            "Use 'model_id' for catalog presets (e.g. 'smolvlm-500m-q4') or 'model_path' for explicit .gguf file paths."
        )

    if model_id is None and model_path is None:
        model_id = "smolvlm-500m-q4"

    if model_id is not None and not str(model_id).strip():
        raise ValueError("Parameter 'model_id' cannot be empty.")

    if model_path is not None and not str(model_path).strip():
        raise ValueError("Parameter 'model_path' cannot be empty.")

    if context_limit is not None and context_limit < 64:
        raise ValueError(f"Parameter 'context_limit' must be >= 64. Received: {context_limit}")

    cache = ModelCacheManager(cache_root=cache_root)

    # 1. Resolve runtime
    runtime_info = resolve_llama_cli(explicit_path=runtime_path)

    # 2. Check if model is via explicit model_path or catalog model_id
    if model_path is not None:
        expanded_model = os.path.abspath(os.path.expanduser(str(model_path)))
        if not os.path.isfile(expanded_model):
            raise FileNotFoundError(f"Model file not found at 'model_path': {model_path}")
        manifest, model_dir = cache.require_installed_model(expanded_model)
        custom_text_path = expanded_model
        custom_vision_path = os.path.abspath(os.path.expanduser(mmproj_path)) if mmproj_path else None
    else:
        expanded_model = os.path.abspath(os.path.expanduser(str(model_id)))
        if os.path.isfile(expanded_model):
            # Graceful legacy fallback for direct file paths passed to model_id
            manifest, model_dir = cache.require_installed_model(expanded_model)
            custom_text_path = expanded_model
            custom_vision_path = os.path.abspath(os.path.expanduser(mmproj_path)) if mmproj_path else None
        else:
            if allow_download and not cache.is_model_installed(model_id):
                cache.install(model_id)

            manifest, model_dir = cache.require_installed_model(model_id)
            custom_text_path = None
            custom_vision_path = os.path.abspath(os.path.expanduser(mmproj_path)) if mmproj_path else None

    # 3. User Freedom: Memory Admission Check
    estimate = MemoryEstimate(
        model_weights_mb=int(manifest.estimated_memory_mb * 0.5),
        vision_encoder_mb=int(manifest.estimated_memory_mb * 0.25),
        kv_cache_mb=int(manifest.estimated_memory_mb * 0.15),
        compute_buffers_mb=int(manifest.estimated_memory_mb * 0.05),
        runtime_overhead_mb=int(manifest.estimated_memory_mb * 0.05),
        estimated_peak_mb=manifest.estimated_memory_mb,
        confidence="estimated"
    )
    _, warning_msg = check_memory_admission(
        estimate,
        user_budget_mb=memory_budget_mb,
        memory_policy=memory_policy
    )

    acquire_engine_lock()

    try:
        actual_threads = 4
        if isinstance(threads, int):
            if threads <= 0:
                raise ValueError(f"Parameter 'threads' must be a positive integer > 0. Received: {threads}")
            actual_threads = max(1, min(128, threads))
        elif isinstance(threads, str) and threads != "auto":
            try:
                val = int(threads)
                if val <= 0:
                    raise ValueError(f"Parameter 'threads' must be a positive integer > 0. Received: {threads}")
                actual_threads = max(1, min(128, val))
            except ValueError as _val_err:
                if "positive integer" in str(_val_err):
                    raise
                import logging
                logging.getLogger(__name__).warning("Invalid threads string %r, defaulting to 4: %s", threads, _val_err)

        # 4. Strict Device & Fallback Policy (Zero-Silent-Fallback Enforced)
        requested_device = device.lower().strip()
        if requested_device in ("vulkan", "vulkan-force", "gpu"):
            actual_backend = "vulkan"
            actual_fallback = False
        elif requested_device == "auto":
            actual_backend = "auto"
            actual_fallback = False
        else:
            actual_backend = "cpu"
            actual_fallback = False

        runtime = SubprocessVLMRuntime(
            manifest=manifest,
            model_dir=model_dir,
            executable=runtime_info.executable,
            threads=actual_threads,
            backend=actual_backend,
            fallback=actual_fallback,
            custom_text_model=custom_text_path,
            custom_vision_model=custom_vision_path,
            context_limit=context_limit,
            ngl=ngl,
            rpc=rpc,
            tensor_split=tensor_split,
            cluster_rpc_servers=cluster_rpc_servers,
            cluster_split_mode=cluster_split_mode,
            cluster_tensor_split=cluster_tensor_split,
            cluster_vram_budget=cluster_vram_budget,
        )

        return VLMContext(runtime=runtime, manifest=manifest, warning_msg=warning_msg)
    except Exception:
        release_engine_lock()
        raise
