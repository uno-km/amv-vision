"""Unit tests for distributed RPC and multi-node tensor splitting in termux-vision VLM."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from termux_vision.vlm.engine import ZeroFlickerEngine
from termux_vision.vlm.runtime.subprocess import SubprocessVLMRuntime
from termux_vision.vlm.manifest import ModelManifest, ArtifactInfo
from termux_vision.cli.main import main


def test_cli_vlm_rpc_and_tensor_split_args(tmp_path):
    """Verify CLI parses --rpc and -ts flags for VLM subcommand and forwards to load()."""
    dummy_img = tmp_path / "test.jpg"
    dummy_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)

    with patch("termux_vision.vlm.api.load") as mock_load, \
         patch("termux_vision.cli.main.ModelCacheManager"):
        mock_ctx = MagicMock()
        mock_res = MagicMock()
        mock_res.text = "A test scene description"
        mock_res.warnings = []
        mock_res.metrics = None
        mock_ctx.__enter__.return_value.describe.return_value = mock_res
        mock_load.return_value = mock_ctx

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", [
                "termux-vision",
                "vlm",
                str(dummy_img),
                "-p", "Describe this image",
                "-m", "smolvlm-500m-q4",
                "--rpc", "192.168.0.220:50052,192.168.0.253:50052",
                "-ts", "50,50",
            ]):
                main()

        assert exc_info.value.code == 0
        assert mock_load.called
        kwargs = mock_load.call_args[1]
        assert kwargs["rpc"] == "192.168.0.220:50052,192.168.0.253:50052"
        assert kwargs["tensor_split"] == "50,50"


def test_subprocess_vlm_runtime_passes_rpc(tmp_path):
    """Verify SubprocessVLMRuntime passes rpc and tensor_split to LlamaRuntime.generate_vlm."""
    dummy_text = tmp_path / "text.gguf"
    dummy_text.write_bytes(b"GGUF" + b"\x00" * 32)
    dummy_vision = tmp_path / "vision.gguf"
    dummy_vision.write_bytes(b"GGUF" + b"\x00" * 32)

    manifest = ModelManifest(
        schema_version=1,
        model_id="test-vlm",
        adapter="smolvlm",
        tier="S",
        estimated_memory_mb=500,
        context_limit=2048,
        preferred_resolution=384,
        artifacts=[
            ArtifactInfo(role="language_model", filename="text.gguf", size_bytes=100, sha256="", download_url=""),
            ArtifactInfo(role="vision_projector", filename="vision.gguf", size_bytes=100, sha256="", download_url="")
        ]
    )

    runtime = SubprocessVLMRuntime(
        manifest=manifest,
        model_dir=str(tmp_path),
        executable="llama-cli",
        rpc="192.168.0.220:50052",
        tensor_split="60,40",
        custom_text_model=str(dummy_text),
        custom_vision_model=str(dummy_vision),
    )

    assert runtime.rpc == "192.168.0.220:50052"
    assert runtime.tensor_split == "60,40"

    with patch("termux_vision.vlm.runtime.subprocess.verify_rpc_cluster_nodes"), \
         patch("termux_llamacpp.LlamaRuntime") as mock_llama_cls:
        mock_llama = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Inference completed"
        mock_resp.backend = "rpc"
        mock_llama.generate_vlm.return_value = mock_resp
        mock_llama_cls.return_value = mock_llama

        res = runtime._execute_once(
            image_path=str(tmp_path / "img.jpg"),
            prompt="Hello world",
        )

        assert mock_llama.generate_vlm.called
        call_kwargs = mock_llama.generate_vlm.call_args[1]
        assert call_kwargs["rpc"] == "192.168.0.220:50052"
        assert call_kwargs["tensor_split"] == "60,40"


def test_zero_flicker_engine_rpc_args(tmp_path):
    """Verify ZeroFlickerEngine builds CLI args containing --rpc and --tensor-split."""
    dummy_img = tmp_path / "img.jpg"
    dummy_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)

    engine = ZeroFlickerEngine(
        text_model_path=str(tmp_path / "text.gguf"),
        vision_model_path=str(tmp_path / "vision.gguf"),
        rpc="192.168.0.253:50052",
        tensor_split="50,50",
    )

    assert engine.rpc == "192.168.0.253:50052"
    assert engine.tensor_split == "50,50"

    captured_cmd = []

    def fake_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("Generated output", "")
        mock_proc.poll.return_value = 0
        return mock_proc

    with patch("termux_vision.vlm.engine.verify_rpc_cluster_nodes"), \
         patch("subprocess.Popen", side_effect=fake_popen):
        res = engine.generate(
            image_path=str(dummy_img),
            prompt="Explain image",
        )

        assert "--rpc" in captured_cmd
        rpc_idx = captured_cmd.index("--rpc")
        assert captured_cmd[rpc_idx + 1] == "192.168.0.253:50052"

        assert "--tensor-split" in captured_cmd
        ts_idx = captured_cmd.index("--tensor-split")
        assert captured_cmd[ts_idx + 1] == "50,50"
