# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
AMEVA Unified Distributed RPC Cluster Orchestration Cold Tests for termux-vision.
Strict Protocol: Zero-Silent-Fallback & Fail-Fast Verification.
Component: [VISION-CLUSTER]
"""

import socket
import pytest
from unittest.mock import patch, MagicMock

import termux_vision as tv
from termux_vision.exceptions import (
    ClusterConnectionError,
    ClusterConfigurationError,
)
from termux_vision.cluster import (
    parse_cluster_rpc_spec,
    verify_rpc_cluster_nodes,
    verify_rpc_cluster_health,
)
from termux_vision.vlm.engine import ZeroFlickerEngine
from termux_vision.vlm.manifest import ModelManifest, ArtifactInfo
from termux_vision.vlm.runtime.subprocess import SubprocessVLMRuntime
from termux_vision.cli.main import main, EXIT_CLUSTER_ERROR


def test_parse_cluster_rpc_spec_valid():
    """Verify parsing and normalization of valid RPC specs."""
    servers_str = "192.0.2.11:50052, 192.0.2.12:50052"
    parsed = parse_cluster_rpc_spec(servers_str)
    assert parsed == ["192.0.2.11:50052", "192.0.2.12:50052"]

    servers_list = ["192.0.2.11:50052", "192.0.2.12:50052"]
    parsed2 = parse_cluster_rpc_spec(servers_list)
    assert parsed2 == ["192.0.2.11:50052", "192.0.2.12:50052"]

    assert parse_cluster_rpc_spec(None) == []
    assert parse_cluster_rpc_spec("") == []


def test_parse_cluster_rpc_spec_invalid():
    """Verify invalid RPC formats strictly raise ClusterConfigurationError."""
    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec("invalid_host_no_port")

    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec("host:not_a_number")

    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec("host:999999")  # Port out of range

    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec(12345)  # Invalid type


def test_verify_rpc_cluster_nodes_success():
    """Verify pre-flight check succeeds when all RPC nodes accept TCP connection."""
    servers = ["192.0.2.11:50052", "192.0.2.12:50052"]
    with patch("socket.create_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value = MagicMock()
        verify_rpc_cluster_nodes(servers, timeout=1.0)
        assert mock_conn.call_count == 2


def test_verify_rpc_cluster_nodes_fail_fast():
    """Verify Zero-Silent-Fallback: Unreachable node raises ClusterConnectionError immediately."""
    servers = ["192.0.2.11:50052", "192.0.2.12:50052"]

    def fake_connect(addr, timeout=None):
        host, port = addr
        if host == "192.0.2.12":
            raise ConnectionRefusedError("Connection refused by test worker")
        return MagicMock()

    with patch("socket.create_connection", side_effect=fake_connect):
        with pytest.raises(ClusterConnectionError) as exc_info:
            verify_rpc_cluster_nodes(servers, timeout=1.0)
        assert "Zero-Silent-Fallback Violation Prevented" in str(exc_info.value)
        assert "192.0.2.12:50052" in str(exc_info.value)


def test_verify_rpc_cluster_health_reporting():
    """Verify granular diagnostics from verify_rpc_cluster_health."""
    servers = ["192.0.2.11:50052", "192.0.2.12:50052"]

    def fake_connect(addr, timeout=None):
        host, port = addr
        if host == "192.0.2.12":
            raise socket.timeout("Timed out")
        return MagicMock()

    with patch("socket.create_connection", side_effect=fake_connect):
        health = verify_rpc_cluster_health(servers, timeout=1.0)
        assert health["all_healthy"] is False
        assert health["nodes"]["192.0.2.11:50052"]["reachable"] is True
        assert health["nodes"]["192.0.2.12:50052"]["reachable"] is False


def test_zero_flicker_engine_cluster_args(tmp_path):
    """Verify ZeroFlickerEngine normalizes and passes cluster parameters."""
    dummy_img = tmp_path / "img.jpg"
    dummy_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)

    engine = ZeroFlickerEngine(
        text_model_path=str(tmp_path / "text.gguf"),
        vision_model_path=str(tmp_path / "vision.gguf"),
        cluster_rpc_servers="192.0.2.11:50052, 192.0.2.12:50052",
        cluster_split_mode="tensor",
        cluster_tensor_split="40,60",
    )

    assert len(engine.cluster_rpc_servers) == 2
    assert engine.cluster_tensor_split == "40,60"
    assert engine.rpc == "192.0.2.11:50052,192.0.2.12:50052"

    with patch("termux_vision.vlm.engine.verify_rpc_cluster_nodes") as mock_verify, \
         patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("Test response", "")
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        res = engine.generate(str(dummy_img), prompt="test")
        assert mock_verify.called
        assert res["text"] == "Test response"


def test_subprocess_vlm_runtime_cluster_preflight_fail(tmp_path):
    """Verify SubprocessVLMRuntime fails fast if RPC cluster node is down."""
    dummy_text = tmp_path / "text.gguf"
    dummy_text.write_bytes(b"GGUF" + b"\x00" * 32)
    dummy_vision = tmp_path / "vision.gguf"
    dummy_vision.write_bytes(b"GGUF" + b"\x00" * 32)
    dummy_img = tmp_path / "img.jpg"
    dummy_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)

    manifest = ModelManifest(
        schema_version=1,
        model_id="test-vlm",
        adapter="smolvlm",
        tier="S",
        estimated_memory_mb=500,
        context_limit=2048,
        preferred_resolution=384,
        artifacts=[]
    )

    runtime = SubprocessVLMRuntime(
        manifest=manifest,
        model_dir=str(tmp_path),
        executable="llama-cli",
        cluster_rpc_servers="192.0.2.11:50052",
        custom_text_model=str(dummy_text),
        custom_vision_model=str(dummy_vision),
    )

    with patch("termux_vision.vlm.runtime.subprocess.verify_rpc_cluster_nodes", side_effect=ClusterConnectionError("Node down")):
        with pytest.raises(ClusterConnectionError):
            runtime.execute(str(dummy_img), prompt="test")


def test_cli_cluster_rpc_servers_flag(tmp_path):
    """Verify CLI --cluster-rpc-servers flag is parsed and forwarded to load()."""
    dummy_img = tmp_path / "test.jpg"
    dummy_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)

    with patch("termux_vision.vlm.api.load") as mock_load, \
         patch("termux_vision.cli.main.ModelCacheManager"):
        mock_ctx = MagicMock()
        mock_res = MagicMock()
        mock_res.text = "Multimodal cluster description"
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
                "--cluster-rpc-servers", "192.0.2.11:50052,192.0.2.12:50052",
                "--cluster-split-mode", "tensor",
                "--cluster-tensor-split", "40,60",
            ]):
                main()

        assert exc_info.value.code == 0
        assert mock_load.called
        kwargs = mock_load.call_args[1]
        assert kwargs["cluster_rpc_servers"] == "192.0.2.11:50052,192.0.2.12:50052"
        assert kwargs["cluster_split_mode"] == "tensor"
        assert kwargs["cluster_tensor_split"] == "40,60"
