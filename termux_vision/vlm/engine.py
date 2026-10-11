# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import os
import time
import subprocess
import tempfile
from typing import Dict, Any, Optional, Union, List
import numpy as np

from termux_vision.cluster import (
    parse_cluster_rpc_spec,
    verify_rpc_cluster_nodes,
)

class ZeroFlickerEngine:
    """
    On-Device VLM Execution Engine optimized for Android Termux.
    Enforces a strict <500MB memory footprint, 4-thread CPU governor,
    and single-turn non-interactive execution to guarantee zero SystemUI flickers.
    """
    def __init__(
        self,
        text_model_path: str,
        vision_model_path: str,
        threads: Optional[int] = None,
        eco_mode: bool = False,
        use_vulkan: bool = False,
        rpc: Optional[str] = None,
        tensor_split: Optional[str] = None,
        cluster_rpc_servers: Optional[Union[str, List[str]]] = None,
        cluster_split_mode: Optional[str] = None,
        cluster_tensor_split: Optional[str] = None,
        cluster_vram_budget: Optional[Dict[str, int]] = None,
    ):
        self.text_model_path = text_model_path
        self.vision_model_path = vision_model_path
        if threads is not None:
            self.threads = threads
        elif eco_mode:
            self.threads = 4
        else:
            self.threads = min(6, os.cpu_count() or 4)
        self.eco_mode = eco_mode
        self.use_vulkan = use_vulkan

        # Cluster configuration normalization
        resolved_rpc = cluster_rpc_servers if cluster_rpc_servers is not None else rpc
        if resolved_rpc:
            self.cluster_rpc_servers = parse_cluster_rpc_spec(resolved_rpc)
            self.rpc = ",".join(self.cluster_rpc_servers)
        else:
            self.cluster_rpc_servers = []
            self.rpc = None

        self.cluster_split_mode = cluster_split_mode
        self.cluster_tensor_split = cluster_tensor_split or tensor_split
        self.tensor_split = self.cluster_tensor_split
        self.cluster_vram_budget = cluster_vram_budget

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.6,
        repeat_penalty: float = 1.25,
        batch_size: int = 64,
        ubatch_size: Optional[int] = None,
        flash_attn: bool = False,
        no_mmproj_offload: bool = True,
        context_limit: int = 2048,
        rpc: Optional[str] = None,
        tensor_split: Optional[str] = None,
        cluster_rpc_servers: Optional[Union[str, List[str]]] = None,
        cluster_split_mode: Optional[str] = None,
        cluster_tensor_split: Optional[str] = None,
        cluster_vram_budget: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Executes single-turn multimodal generation with unified 2048 context limit.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # Resolve cluster parameters
        eff_cluster_rpc = cluster_rpc_servers if cluster_rpc_servers is not None else rpc
        if eff_cluster_rpc:
            parsed_rpc = parse_cluster_rpc_spec(eff_cluster_rpc)
        else:
            parsed_rpc = self.cluster_rpc_servers

        # Preflight TCP reachability verification (Zero-Silent-Fallback)
        if parsed_rpc:
            verify_rpc_cluster_nodes(parsed_rpc)
            effective_rpc = ",".join(parsed_rpc)
        else:
            effective_rpc = None

        effective_ts = cluster_tensor_split or tensor_split or self.tensor_split

        # Write prompt safely to temporary utf-8 file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as pf:
            prompt_path = pf.name
            pf.write(prompt)

        # 1. Resolve canonical execution engine (prioritizing dedicated termux-vlm-cli)
        bin_path = "termux-vlm-cli"
        llamacpp_runtime = None
        try:
            from .runtime.resolver import resolve_llama_cli
            bin_path = resolve_llama_cli().executable
        except Exception:
            try:
                from termux_llamacpp import LlamaRuntime
                llamacpp_runtime = LlamaRuntime()
                resolved = llamacpp_runtime.get_binary_path("llama-cli")
                if resolved:
                    bin_path = str(resolved)
            except Exception as _llamacpp_err:
                import logging
                logging.getLogger(__name__).debug("vision: termux-llamacpp SDK fallback unavailable: %s", _llamacpp_err)

        # 2. Prepare execution environment (prioritizing termux-llamacpp and ameva-runtime)
        if llamacpp_runtime and hasattr(llamacpp_runtime, "prepare_env"):
            proc_env = llamacpp_runtime.prepare_env(device="vulkan" if self.use_vulkan else "cpu")
        else:
            proc_env = os.environ.copy()

        home_dir = os.environ.get("HOME", os.path.expanduser("~"))
        rpc_lib_dir = os.path.join(home_dir, ".local", "share", "ameva", "rpc_llama", "bin")
        if os.path.isdir(rpc_lib_dir):
            cur_ld = proc_env.get("LD_LIBRARY_PATH", "")
            if rpc_lib_dir not in cur_ld:
                proc_env["LD_LIBRARY_PATH"] = f"{rpc_lib_dir}:{cur_ld}" if cur_ld else rpc_lib_dir

        if self.use_vulkan:
            try:
                from ameva_runtime.adapters import VisionAdapter
                proc_env = VisionAdapter.get_execution_environment(base_env=proc_env)
            except ImportError as _vulkan_err:
                raise RuntimeError(
                    "[FAIL-FAST] [ERROR: AMEVA-VISION-E001] GPU acceleration requires 'ameva-runtime'.\n"
                    "Cause: Hardware abstraction provider 'ameva-runtime' is not installed.\n"
                    "Action Required: Install ameva-runtime or run without GPU: use_vulkan=False\n"
                    "Documentation: https://uno-km.vercel.app/lib/vision/"
                ) from _vulkan_err
            except Exception as _vulkan_err:
                raise RuntimeError(
                    f"[FAIL-FAST] [ERROR: AMEVA-VISION-E001] Vision Vulkan acceleration initialization failed: {_vulkan_err}"
                ) from _vulkan_err

        try:
            try:
                from ameva_runtime.adapters import VisionAdapter
                cli_cmd = VisionAdapter.build_cli_args(
                    executable=bin_path,
                    text_model_path=self.text_model_path,
                    vision_model_path=self.vision_model_path,
                    image_path=image_path,
                    prompt_file=prompt_path,
                    target_backend="vulkan" if self.use_vulkan else "cpu",
                    threads=self.threads,
                    context_limit=context_limit,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    repeat_penalty=repeat_penalty,
                    pure_gpu=True,
                    fit_off=True,
                    batch_size=batch_size,
                    ubatch_size=ubatch_size,
                    flash_attn=flash_attn,
                    no_mmproj_offload=no_mmproj_offload,
                )
            except ImportError:
                cli_cmd = [
                    bin_path,
                    "-m", self.text_model_path,
                    "--mmproj", self.vision_model_path,
                    "--image", image_path,
                    "-f", prompt_path,
                    "-t", str(self.threads),
                    "-c", str(context_limit),
                    "-n", str(max_tokens),
                    "--temp", str(temperature),
                    "--repeat-penalty", str(repeat_penalty),
                    "-ngl", "99" if self.use_vulkan else "0",
                    "-b", str(batch_size),
                    "-ub", str(ubatch_size if ubatch_size is not None else batch_size),
                ]
                if not flash_attn:
                    cli_cmd.extend(["-fa", "off"])
                if no_mmproj_offload:
                    cli_cmd.append("--no-mmproj-offload")
                if self.use_vulkan:
                    cli_cmd.extend(["-ot", "token_embd.weight=Vulkan0", "-fit", "off"])

            if effective_rpc:
                cli_cmd.extend(["--rpc", str(effective_rpc)])
            if effective_ts:
                cli_cmd.extend(["--tensor-split", str(effective_ts)])

            t0 = time.perf_counter()
            process = subprocess.Popen(
                cli_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=proc_env,
                text=True
            )
            out, err = process.communicate()
            latency = time.perf_counter() - t0

            # Extract generated response text
            lines = out.splitlines()
            content_lines = []
            start_capture = False
            for line in lines:
                if line.strip().startswith(">"):
                    start_capture = True
                    continue
                if start_capture:
                    if "[" in line and "t/s" in line:
                        continue
                    if "Exiting" in line:
                        continue
                    cleaned = line.replace("|", "").replace("-", "").replace("/", "").replace("\\", "").strip()
                    if cleaned:
                        content_lines.append(cleaned)

            text_output = " ".join(content_lines) if content_lines else out.strip()

            return {
                "text": text_output,
                "latency_sec": round(latency, 2),
                "threads": self.threads,
                "eco_mode": self.eco_mode,
                "raw_stdout": out
            }
        finally:
            if os.path.exists(prompt_path):
                os.remove(prompt_path)
