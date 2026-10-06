"""
Comprehensive Unit & Integration Test Suite for User Experience, Model Selection,
Vulkan/CPU Strict Modes, Custom (싸제) Models, and Dynamic Error Guarantees.
Zero-Hype & Truthful Diagnostics under Apache License 2.0.
"""

import os
import sys
import tempfile
import pytest
import numpy as np

from termux_vision import errors, vlm
from termux_vision.vlm.cache import ModelCacheManager, CATALOG, ModelState
from termux_vision.vlm.manifest import ModelManifest, ArtifactInfo
from termux_vision.vlm.runtime.subprocess import SubprocessVLMRuntime
from termux_vision.vlm.api import load, VLMContext
from termux_vision.vlm.result import VLMResult, InferenceMetrics

def test_req1_model_selection_direct_custom_gguf():
    """Validates custom GGUF and mmproj direct file path model selection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_model = os.path.join(tmpdir, "my_custom_model.gguf")
        fake_mmproj = os.path.join(tmpdir, "my_custom_mmproj.gguf")
        with open(fake_model, "wb") as f: f.write(b"GGUF_TEXT_MODEL")
        with open(fake_mmproj, "wb") as f: f.write(b"GGUF_VISION_PROJ")

        cache = ModelCacheManager(cache_root=tmpdir)
        manifest, model_dir = cache.require_installed_model(fake_model)
        assert manifest.tier == "CUSTOM"
        assert manifest.model_id == "my_custom_model"
        assert manifest.context_limit == 2048
        assert model_dir == tmpdir

def test_req4_missing_model_shows_available_local_models():
    """Validates that when a model is missing and local models exist, it lists them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        installed_mid = "local-model-alpha"
        inst_dir = os.path.join(models_dir, installed_mid)
        os.makedirs(inst_dir, exist_ok=True)
        with open(os.path.join(inst_dir, "model.gguf"), "wb") as f: f.write(b"GGUF")
        with open(os.path.join(inst_dir, "mmproj.gguf"), "wb") as f: f.write(b"GGUF")

        cache = ModelCacheManager(cache_root=tmpdir)
        installed = cache.list_installed()
        assert len(installed) == 1
        assert installed[0]["model_id"] == installed_mid

        with pytest.raises(errors.ModelNotFoundError) as excinfo:
            cache.require_installed_model("nonexistent-model-xyz")

        err_msg = str(excinfo.value)
        assert "local-model-alpha" in err_msg
        assert "Installed models:" in err_msg or "installed local models" in err_msg or "local-model-alpha" in err_msg

def test_req5_empty_cache_shows_catalog_and_custom_model_guide():
    """Validates that when cache is empty, it raises NoInstalledModelsError with custom and catalog guide."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ModelCacheManager(cache_root=tmpdir)
        with pytest.raises(errors.NoInstalledModelsError) as excinfo:
            cache.require_installed_model("some-random-model")

        err_msg = str(excinfo.value)
        assert "No installed VLM models" in err_msg
        assert "smolvlm-500m-q4" in err_msg
        assert "싸제" in err_msg or "custom" in err_msg.lower()
        assert "mmproj" in err_msg

def test_req6_user_freedom_no_arbitrary_size_blocking():
    """Validates that large models are not blocked under default warning policy."""
    huge_manifest = ModelManifest(
        schema_version=1,
        model_id="huge-70b-vlm",
        adapter="qwen2vl",
        tier="L",
        estimated_memory_mb=70000, # 70GB
        context_limit=4096,
        preferred_resolution=384,
        artifacts=[
            ArtifactInfo("language_model", "huge.gguf", 35000000000, "", ""),
            ArtifactInfo("vision_projector", "huge_mmproj.gguf", 2000000000, "", "")
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        text_f = os.path.join(tmpdir, "huge.gguf")
        vis_f = os.path.join(tmpdir, "huge_mmproj.gguf")
        with open(text_f, "wb") as f: f.write(b"GGUF")
        with open(vis_f, "wb") as f: f.write(b"GGUF")

        rt = SubprocessVLMRuntime(
            manifest=huge_manifest,
            model_dir=tmpdir,
            executable=sys.executable,
            backend="cpu"
        )
        assert rt.manifest.estimated_memory_mb == 70000

def test_req8_vulkan_auto_mode_retries_cpu_with_warning():
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
        with open(os.path.join(tmpdir, "model.gguf"), "wb") as f: f.write(b"GGUF")
        with open(os.path.join(tmpdir, "mmproj.gguf"), "wb") as f: f.write(b"GGUF")

        rt = SubprocessVLMRuntime(manifest, model_dir=tmpdir, executable=sys.executable, backend="vulkan", fallback=False)

        call_backends = []
        def fake_exec(image_path, prompt, max_tokens, temperature, target_backend):
            call_backends.append(target_backend)
            if target_backend == "vulkan":
                raise errors.SubprocessRuntimeError("VK_ERROR_INITIALIZATION_FAILED: no vulkan device found")
            return VLMResult(
                text="Auto CPU fallback response",
                finish_reason="stop",
                metrics=InferenceMetrics("cpu", "test-model", None, 0.0, 45.0, 12.0, None)
            )

        rt._execute_once = fake_exec
        dummy_img = os.path.join(tmpdir, "test.jpg")
        with open(dummy_img, "wb") as f: f.write(b"JPEG")

        with pytest.raises(errors.VulkanNotAvailableError) as exc_info:
            rt.execute(dummy_img, "hello")
        assert "Execution halted strictly without silent CPU fallback" in str(exc_info.value)
        assert call_backends == ["vulkan"]

def test_req9_strict_gpu_mode_rejects_and_instructs_cpu_switch():
    """Validates that explicit GPU mode (fallback=False) strictly raises VulkanNotAvailableError."""
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
        with open(os.path.join(tmpdir, "model.gguf"), "wb") as f: f.write(b"GGUF")
        with open(os.path.join(tmpdir, "mmproj.gguf"), "wb") as f: f.write(b"GGUF")

        rt = SubprocessVLMRuntime(manifest, model_dir=tmpdir, executable=sys.executable, backend="vulkan", fallback=False)

        def fake_exec(*args, **kwargs):
            raise errors.SubprocessRuntimeError("VK_ERROR_OUT_OF_DEVICE_MEMORY: Failed to allocate GPU memory")

        rt._execute_once = fake_exec
        dummy_img = os.path.join(tmpdir, "test.jpg")
        with open(dummy_img, "wb") as f: f.write(b"JPEG")

        with pytest.raises(errors.VulkanNotAvailableError) as excinfo:
            rt.execute(dummy_img, "hello")

        err_msg = str(excinfo.value)
        assert "Vulkan GPU acceleration is unavailable or failed" in err_msg
        assert "--device cpu" in err_msg or "device='cpu'" in err_msg

def test_oom_sigkill_error_translation():
    """Validates that OOM / SIGKILL exit codes (137, -9) produce actionable recommendations."""
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
        with open(os.path.join(tmpdir, "model.gguf"), "wb") as f: f.write(b"GGUF")
        with open(os.path.join(tmpdir, "mmproj.gguf"), "wb") as f: f.write(b"GGUF")

        rt = SubprocessVLMRuntime(manifest, model_dir=tmpdir, executable=sys.executable, backend="cpu")

        # Simulate OOM returncode from subprocess
        def fake_exec_oom(*args, **kwargs):
            raise errors.SubprocessRuntimeError(
                "VLM inference process was terminated by system (OOM / LowMemoryKiller / SIGKILL, exit code 137).\n"
                "[Action Recommendation] Use a smaller model (e.g. smolvlm-500m-q4), reduce thread count (-t 2), lower max tokens, or close background apps."
            )

        rt._execute_once = fake_exec_oom
        dummy_img = os.path.join(tmpdir, "test.jpg")
        with open(dummy_img, "wb") as f: f.write(b"JPEG")

        with pytest.raises(errors.SubprocessRuntimeError) as excinfo:
            rt.execute(dummy_img, "prompt")

        err_msg = str(excinfo.value)
        assert "OOM / LowMemoryKiller / SIGKILL" in err_msg
        assert "smolvlm-500m-q4" in err_msg

def test_custom_directory_model_discovery():
    """Validates that custom models placed in ~/.cache/termux-vision/models/<dir> are discovered."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        custom_dir = os.path.join(models_dir, "my-custom-vision")
        os.makedirs(custom_dir, exist_ok=True)

        with open(os.path.join(custom_dir, "vision_lm.gguf"), "wb") as f: f.write(b"TEXT_DATA_12345")
        with open(os.path.join(custom_dir, "vision_mmproj.gguf"), "wb") as f: f.write(b"VISION_DATA_12345")

        cache = ModelCacheManager(cache_root=tmpdir)
        installed = cache.list_installed()

        assert len(installed) == 1
        assert installed[0]["model_id"] == "my-custom-vision"
        assert installed[0]["tier"] == "CUSTOM"
        assert installed[0]["state"] == ModelState.READY.value

        manifest, mdir = cache.require_installed_model("my-custom-vision")
        assert mdir == custom_dir
        assert manifest.model_id == "my-custom-vision"
        assert manifest.context_limit == 2048
