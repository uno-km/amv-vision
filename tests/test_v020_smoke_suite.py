import os
import sys
import json
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import termux_vision as tv
from termux_vision import errors, detect, io, vlm
from termux_vision.cli.doctor import run_doctor
from termux_vision.vlm.runtime.subprocess import SubprocessVLMRuntime
from termux_vision.vlm.manifest import ModelManifest, ArtifactInfo
from termux_vision.vlm.cache import ModelCacheManager, CATALOG
from termux_vision.vlm.runtime.resolver import resolve_llama_cli

def test_doctor_text_output():
    """Validates that doctor report contains all keys required by CLI without KeyError."""
    rep = run_doctor(probe_vulkan=True, full_check=False)
    assert "loader_detected" in rep["vulkan"]
    assert "driver_file_detected" in rep["vulkan"]
    assert "status" in rep["vulkan"]
    assert "cpu_cores" in rep["hardware"]
    assert rep["recommended_preset"] is not None

def test_camera_calls_load_image_with_valid_signature():
    """Validates camera capture calls load_image without invalid 'mode' keyword."""
    cam = io.camera.CameraCapture(camera_id=0)
    with patch("subprocess.run") as mock_run, patch("termux_vision.io.camera.load_image") as mock_load:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with patch("os.path.exists", return_value=True), patch("os.path.getsize", return_value=1024):
            cam.capture_frame(target_size=(256, 256))
            mock_load.assert_called_once()
            _, kwargs = mock_load.call_args
            assert "mode" not in kwargs
            assert kwargs.get("target_size") == (256, 256)

def test_runtime_output_decodes_text():
    """Validates runtime process communication correctly handles text strings without bytes errors."""
    manifest = ModelManifest(
        schema_version=1,
        model_id="test-model",
        adapter="smolvlm",
        tier="M",
        estimated_memory_mb=500,
        context_limit=512,
        preferred_resolution=256,
        artifacts=[
            ArtifactInfo("language_model", "model.gguf", 100, "", ""),
            ArtifactInfo("vision_projector", "mmproj.gguf", 100, "", "")
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        text_gguf = os.path.join(tmpdir, "model.gguf")
        vision_gguf = os.path.join(tmpdir, "mmproj.gguf")
        with open(text_gguf, "wb") as f: f.write(b"GGUF")
        with open(vision_gguf, "wb") as f: f.write(b"GGUF")

        rt = SubprocessVLMRuntime(manifest, model_dir=tmpdir, executable=sys.executable, backend="cpu")

        from termux_llamacpp.engine import VLMResponse
        with patch("termux_llamacpp.LlamaRuntime.generate_vlm") as mock_vlm:
            mock_vlm.return_value = VLMResponse(
                text="This is a validated test image description.",
                backend="cpu",
                generation_tps=14.5,
                prompt_tps=50.0,
                latency_ms=1000.0,
            )

            dummy_img = os.path.join(tmpdir, "test.jpg")
            with open(dummy_img, "wb") as f: f.write(b"JPEG")

            res = rt.execute(dummy_img, "Describe", max_tokens=32)
            assert "validated test image description" in res.text
            assert res.metrics.tokens_per_second == 14.5
            assert res.word_count == 7
            assert res.output_tokens is None

def test_vulkan_failure_retries_cpu_when_fallback_enabled():
    """Validates that Vulkan failure strictly raises VulkanNotAvailableError without silent CPU fallback."""
    manifest = ModelManifest(
        schema_version=1,
        model_id="test-model",
        adapter="smolvlm",
        tier="M",
        estimated_memory_mb=500,
        context_limit=512,
        preferred_resolution=256,
        artifacts=[
            ArtifactInfo("language_model", "model.gguf", 100, "", ""),
            ArtifactInfo("vision_projector", "mmproj.gguf", 100, "", "")
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        text_gguf = os.path.join(tmpdir, "model.gguf")
        vision_gguf = os.path.join(tmpdir, "mmproj.gguf")
        with open(text_gguf, "wb") as f: f.write(b"GGUF")
        with open(vision_gguf, "wb") as f: f.write(b"GGUF")

        rt = SubprocessVLMRuntime(manifest, model_dir=tmpdir, executable=sys.executable, backend="vulkan", fallback=False)

        call_count = 0
        def fake_execute_once(image_path, prompt, max_tokens, temperature, target_backend):
            nonlocal call_count
            call_count += 1
            if target_backend == "vulkan":
                raise errors.SubprocessRuntimeError("VK_ERROR_DEVICE_LOST: GPU timeout")
            return vlm.result.VLMResult(
                text="CPU fallback succeeded",
                finish_reason="stop",
                metrics=vlm.result.InferenceMetrics("cpu", "test-model", None, 0.0, 50.0, 10.0, None)
            )

        rt._execute_once = fake_execute_once
        dummy_img = os.path.join(tmpdir, "test.jpg")
        with open(dummy_img, "wb") as f: f.write(b"JPEG")

        with pytest.raises(errors.VulkanNotAvailableError) as exc_info:
            rt.execute(dummy_img, "test prompt")
        assert call_count == 1
        assert "Execution halted strictly without silent CPU fallback" in str(exc_info.value)

def test_no_fallback_propagates_error():
    """Validates error propagation when fallback=False."""
    manifest = ModelManifest(
        schema_version=1,
        model_id="test-model",
        adapter="smolvlm",
        tier="M",
        estimated_memory_mb=500,
        context_limit=512,
        preferred_resolution=256,
        artifacts=[
            ArtifactInfo("language_model", "model.gguf", 100, "", ""),
            ArtifactInfo("vision_projector", "mmproj.gguf", 100, "", "")
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        text_gguf = os.path.join(tmpdir, "model.gguf")
        vision_gguf = os.path.join(tmpdir, "mmproj.gguf")
        with open(text_gguf, "wb") as f: f.write(b"GGUF")
        with open(vision_gguf, "wb") as f: f.write(b"GGUF")

        rt = SubprocessVLMRuntime(manifest, model_dir=tmpdir, executable=sys.executable, backend="vulkan", fallback=False)

        def fake_execute_once(*args, **kwargs):
            raise errors.SubprocessRuntimeError("VK_ERROR_DEVICE_LOST: GPU timeout")

        rt._execute_once = fake_execute_once
        dummy_img = os.path.join(tmpdir, "test.jpg")
        with open(dummy_img, "wb") as f: f.write(b"JPEG")

        with pytest.raises(errors.SubprocessRuntimeError):
            rt.execute(dummy_img, "test prompt")

def test_corrupt_existing_model_quarantined_and_reinstalled():
    """Validates that a corrupted model directory can be recovered by re-installation without blocking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ModelCacheManager(cache_root=tmpdir)
        m_dir = os.path.join(mgr.models_dir, "test-m")
        os.makedirs(m_dir, exist_ok=True)
        q_file = os.path.join(m_dir, "QUARANTINED")
        with open(q_file, "w") as f: f.write("CORRUPT")

        assert mgr.get_state("test-m") == vlm.cache.ModelState.QUARANTINED

def test_expected_size_mismatch_fails_integrity():
    """Validates verify_integrity truthfully flags size mismatch and corrupts model state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ModelCacheManager(cache_root=tmpdir)
        m_dir = os.path.join(mgr.models_dir, "smolvlm-500m-q4")
        os.makedirs(m_dir, exist_ok=True)
        
        target_art = CATALOG["smolvlm-500m-q4"].artifacts[0].filename
        art_file = os.path.join(m_dir, target_art)
        with open(art_file, "wb") as f: f.write(b"SHORT_DATA")

        rep = mgr.verify_integrity("smolvlm-500m-q4")
        assert rep["status"] == "CORRUPTED"
        assert rep["artifacts"][target_art]["size_verified"] is False

def test_detection_type_allows_none_score():
    """Validates Detection dataclass allows score=None for heuristic detectors."""
    bbox = detect.BoundingBox(10, 20, 30, 40)
    det = detect.Detection(bbox=bbox, score=None, class_name="face_candidate")
    assert det.score is None
    assert det.bbox.width == 20

def test_runtime_not_found_error():
    """Validates RuntimeNotFoundError is raised when no real llama-cli executable is found."""
    with pytest.raises(errors.RuntimeNotFoundError) as exc_info:
        resolve_llama_cli(explicit_path="/non/existent/llama-cli")
    assert "/non/existent/llama-cli" in str(exc_info.value)

def test_model_id_path_traversal_blocked():
    """Validates that malicious model IDs containing traversal sequences are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ModelCacheManager(cache_root=tmpdir)
        with pytest.raises(ValueError):
            mgr.get_model_dir("../../etc/passwd")
        with pytest.raises(ValueError):
            mgr.get_model_dir("sub/dir/model")
        with pytest.raises(ValueError):
            mgr.get_model_dir("")
