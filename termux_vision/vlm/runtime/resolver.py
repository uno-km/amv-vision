from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from ...errors import RuntimeNotFoundError

DEFAULT_RUNTIME_NAME = "termux-vlm-cli"
RUNTIME_ENV_VAR = "TERMUX_VISION_LLAMA_CLI"

@dataclass(frozen=True)
class RuntimeInfo:
    executable: str
    version: Optional[str]
    source: str

def _is_executable_file(path: str) -> bool:
    candidate = Path(path).expanduser()
    if not candidate.is_file() or not os.access(str(candidate), os.X_OK):
        return False
    try:
        with open(candidate.resolve(), "rb") as f:
            return f.read(4) == b"\x7fELF"
    except Exception:
        return False

def _get_runtime_version(executable: str) -> Optional[str]:
    commands = (
        [executable, "--version"],
        [executable, "-h"],
    )
    for command in commands:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue

        output = "\n".join(
            part.strip()
            for part in (result.stdout, result.stderr)
            if part and part.strip()
        )
        if output:
            return output.splitlines()[0][:500]

    return None

def resolve_llama_cli(
    explicit_path: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> RuntimeInfo:
    environment = os.environ if env is None else env
    searched_paths: list[str] = []

    # 0. termux_llamacpp Python SDK integration (Primary SSOT)
    try:
        import termux_llamacpp as tl
        runtime = getattr(tl, "LlamaRuntime", None)
        if runtime is None and hasattr(tl, "engine"):
            runtime = getattr(tl.engine, "LlamaRuntime", None)
        if runtime:
            rt_inst = runtime()
            cli_bin = rt_inst.get_binary_path("llama-cli") if hasattr(rt_inst, "get_binary_path") else None
            if cli_bin and _is_executable_file(str(cli_bin)):
                resolved = os.path.abspath(str(cli_bin))
                return RuntimeInfo(
                    executable=resolved,
                    version=_get_runtime_version(resolved),
                    source="termux-llamacpp-sdk",
                )
    except Exception as e:
        import logging
        logging.getLogger("termux_vision.vlm.resolver").debug(
            "[termux-vision] termux_llamacpp SDK dynamic lookup skipped: %s", e
        )

    # 1. Explicit path
    if explicit_path:
        expanded = os.path.abspath(os.path.expanduser(explicit_path))
        searched_paths.append(expanded)
        if not _is_executable_file(expanded):
            raise RuntimeNotFoundError(
                executable=explicit_path,
                searched_paths=tuple(searched_paths),
            )
        return RuntimeInfo(
            executable=expanded,
            version=_get_runtime_version(expanded),
            source="explicit",
        )

    # 2. Environment variable
    env_path = environment.get(RUNTIME_ENV_VAR)
    if env_path:
        expanded = os.path.abspath(os.path.expanduser(env_path))
        searched_paths.append(expanded)
        if not _is_executable_file(expanded):
            raise RuntimeNotFoundError(
                executable=env_path,
                searched_paths=tuple(searched_paths),
            )
        return RuntimeInfo(
            executable=expanded,
            version=_get_runtime_version(expanded),
            source="environment",
        )

    # 3. PATH lookup (Multimodal VLM runtime prioritized)
    for name in ("termux-vlm-cli", "llama-mtmd-cli", "llama-cli"):
        discovered = shutil.which(name)
        if discovered:
            resolved = os.path.abspath(discovered)
            return RuntimeInfo(
                executable=resolved,
                version=_get_runtime_version(resolved),
                source="path",
            )

    # 4. Known Termux / Linux paths (SSOT Standard)
    prefix = environment.get("PREFIX", "/data/data/com.termux/files/usr")
    candidates = (
        os.path.join(prefix, "bin", "termux-vlm-cli"),
        os.path.expanduser("~/.local/bin/termux-vlm-cli"),
        os.path.expanduser("~/bin/termux-vlm-cli"),
        os.path.expanduser("~/.termux-llama/current/bin/llama-mtmd-cli"),
        os.path.expanduser("~/.termux-llama/current/bin/llama-cli"),
        os.path.expanduser("~/.termux-llamacpp/current/bin/llama-mtmd-cli"),
        os.path.expanduser("~/.termux-llamacpp/current/bin/llama-cli"),
        os.path.join(prefix, "bin", "llama-mtmd-cli"),
        os.path.join(prefix, "bin", "llama-cli"),
        os.path.join(prefix, "bin", "termux-llama-cli"),
        os.path.expanduser("~/.local/bin/llama-mtmd-cli"),
        os.path.expanduser("~/.local/bin/llama-cli"),
        os.path.expanduser("~/bin/llama-mtmd-cli"),
        os.path.expanduser("~/bin/llama-cli"),
    )

    for candidate in candidates:
        searched_paths.append(candidate)
        if _is_executable_file(candidate):
            resolved = os.path.abspath(candidate)
            return RuntimeInfo(
                executable=resolved,
                version=_get_runtime_version(resolved),
                source="known-path",
            )

    raise RuntimeNotFoundError(
        executable=DEFAULT_RUNTIME_NAME,
        searched_paths=tuple(searched_paths),
    )
