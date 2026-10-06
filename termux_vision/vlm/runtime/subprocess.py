import os
import sys
import time
import subprocess
import tempfile
import signal
from typing import Dict, Any, Optional, Sequence, Union, List
from dataclasses import replace

from ..manifest import ModelManifest
from ..adapters import get_adapter
from ..result import VLMResult, InferenceMetrics
from ...errors import (
    SubprocessRuntimeError,
    RuntimeNotFoundError,
    VulkanNotAvailableError,
    GpuExecutionError
)
from ...cluster import (
    parse_cluster_rpc_spec,
    verify_rpc_cluster_nodes,
)

def _is_vulkan_failure(exc: Exception) -> bool:
    """Checks if an exception indicates a Vulkan driver/GPU failure."""
    text = str(exc).lower()
    markers = (
        "vulkan",
        "vk_error",
        "device lost",
        "failed to initialize gpu",
        "no vulkan device",
        "gpu backend",
        "vkcreateinstance",
        "vkcreatedevice",
        "ggml_vulkan",
        "vulkan loader"
    )
    return any(marker in text for marker in markers)

class SubprocessVLMRuntime:
    """
    Supervised native inference runner using real llama-cli executable.
    Provides process-group isolation on POSIX, strict hardware mode enforcement,
    and truthful metrics reporting.
    """
    def __init__(
        self,
        manifest: ModelManifest,
        model_dir: str,
        executable: str,
        threads: int = 4,
        backend: str = "cpu",
        timeout_sec: int = 300,
        fallback: bool = False,
        custom_text_model: Optional[str] = None,
        custom_vision_model: Optional[str] = None,
        context_limit: Optional[int] = None,
        ngl: Optional[int] = None,
        rpc: Optional[str] = None,
        tensor_split: Optional[str] = None,
        cluster_rpc_servers: Optional[Union[str, Sequence[str]]] = None,
        cluster_split_mode: Optional[str] = None,
        cluster_tensor_split: Optional[str] = None,
        cluster_vram_budget: Optional[Dict[str, int]] = None,
    ):
        self.manifest = manifest
        self.model_dir = model_dir
        self.executable = executable
        self.threads = threads
        self.backend = backend.lower().strip()
        self.timeout_sec = timeout_sec
        self.fallback = fallback
        self.adapter = get_adapter(manifest.adapter)
        self.context_limit = context_limit or manifest.context_limit or 2048
        self.custom_ngl = ngl

        # Cluster options
        eff_rpc = cluster_rpc_servers if cluster_rpc_servers is not None else rpc
        if eff_rpc:
            self.cluster_rpc_servers = parse_cluster_rpc_spec(eff_rpc)
            self.rpc = ",".join(self.cluster_rpc_servers)
        else:
            self.cluster_rpc_servers = []
            self.rpc = None

        self.cluster_split_mode = cluster_split_mode
        self.cluster_tensor_split = cluster_tensor_split or tensor_split
        self.tensor_split = self.cluster_tensor_split
        self.cluster_vram_budget = cluster_vram_budget

        # Artifact resolution
        if custom_text_model and os.path.isfile(custom_text_model):
            self.text_model_path = custom_text_model
        else:
            self.text_model_path = None

        if custom_vision_model and os.path.isfile(custom_vision_model):
            self.vision_model_path = custom_vision_model
        else:
            self.vision_model_path = None

        for a in manifest.artifacts:
            p = os.path.join(model_dir, a.filename)
            if os.path.exists(p):
                if a.role == "language_model" and not self.text_model_path:
                    self.text_model_path = p
                elif a.role == "vision_projector" and not self.vision_model_path:
                    self.vision_model_path = p

        # Fallback to search directory for any .gguf if custom
        if not self.text_model_path or not self.vision_model_path:
            if os.path.isdir(model_dir):
                files = os.listdir(model_dir)
                if not self.text_model_path:
                    texts = [os.path.join(model_dir, f) for f in files if f.endswith(".gguf") and not ("mmproj" in f.lower() or "vision" in f.lower())]
                    if texts:
                        self.text_model_path = texts[0]
                if not self.vision_model_path:
                    visions = [os.path.join(model_dir, f) for f in files if f.endswith(".gguf") and ("mmproj" in f.lower() or "vision" in f.lower())]
                    if visions:
                        self.vision_model_path = visions[0]

        if not self.text_model_path or not self.vision_model_path:
            raise FileNotFoundError(
                f"Missing required VLM artifacts in '{model_dir}' for '{manifest.model_id}'. "
                f"Found: text={self.text_model_path}, vision={self.vision_model_path}. "
                f"VLM models require both a text .gguf and a vision projector mmproj-*.gguf."
            )

    def _execute_once(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.2,
        target_backend: str = "cpu",
        **kwargs
    ) -> VLMResult:
        """
        Execute VLM inference via official termux-llamacpp SDK with strict 1:1 device routing.
        """
        # Resolve runtime cluster parameters
        override_rpc = kwargs.get("cluster_rpc_servers") or kwargs.get("rpc")
        if override_rpc:
            active_rpc_servers = parse_cluster_rpc_spec(override_rpc)
        else:
            active_rpc_servers = self.cluster_rpc_servers

        if active_rpc_servers:
            verify_rpc_cluster_nodes(active_rpc_servers)
            effective_rpc_str = ",".join(active_rpc_servers)
        else:
            effective_rpc_str = None

        effective_tensor_split = kwargs.get("cluster_tensor_split") or kwargs.get("tensor_split") or self.tensor_split

        try:
            from termux_llamacpp import LlamaRuntime
        except ImportError as imp_err:
            raise RuntimeNotFoundError(
                "Official termux-llamacpp SDK is required for VLM multimodal inference.\n"
                "Please install it via:\n"
                "  pip install termux-llamacpp\n"
                "Or run:\n"
                "  termux-vision install\n"
            ) from imp_err

        runtime = LlamaRuntime()
        try:
            response = runtime.generate_vlm(
                model=self.text_model_path,
                mmproj=self.vision_model_path,
                image=image_path,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                threads=self.threads,
                ctx_size=self.context_limit,
                device=target_backend,
                n_gpu_layers=self.custom_ngl,
                repeat_penalty=kwargs.get("repeat_penalty", 1.2),
                top_p=kwargs.get("top_p"),
                top_k=kwargs.get("top_k"),
                rpc=effective_rpc_str,
                tensor_split=effective_tensor_split,
            )
        except Exception as exec_err:
            err_str = str(exec_err)
            if "OOM" in err_str or "exit 137" in err_str:
                raise SubprocessRuntimeError(
                    f"VLM inference process was terminated by system (OOM / LowMemoryKiller).\n"
                    f"[Action Recommendation] Use a smaller model (e.g. smolvlm-500m-q4), reduce thread count (-t 2), lower max tokens, or close background apps."
                ) from exec_err
            raise SubprocessRuntimeError(f"VLM inference failed via termux-llamacpp runtime: {exec_err}") from exec_err

        text_output = response.text
        word_count = len(text_output.split())

        metrics = InferenceMetrics(
            backend=response.backend,
            model_id=self.manifest.model_id,
            load_ms=None,
            vision_ms=0.0,
            decode_ms=response.latency_ms or 0.0,
            tokens_per_second=response.generation_tps,
            peak_rss_mb=None
        )

        hw_diag = []
        if response.prompt_tps:
            hw_diag.append(f"[Prompt Speed] {response.prompt_tps} t/s")

        return VLMResult(
            text=text_output,
            finish_reason=response.finish_reason,
            input_tokens=None,
            output_tokens=None,
            word_count=word_count,
            metrics=metrics,
            warnings=tuple(hw_diag)
        )

    def execute(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.2,
        **kwargs
    ) -> VLMResult:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        try:
            try:
                return self._execute_once(
                    image_path,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    target_backend=self.backend,
                    **kwargs
                )
            except TypeError:
                return self._execute_once(
                    image_path,
                    prompt,
                    max_tokens,
                    temperature,
                    self.backend
                )
        except Exception as exc:
            if self.backend in ("vulkan", "gpu"):
                raise VulkanNotAvailableError(
                    reason=f"Vulkan execution failed: {exc}. Execution halted strictly without silent CPU fallback."
                ) from exc
            raise

    def close(self):
        pass
