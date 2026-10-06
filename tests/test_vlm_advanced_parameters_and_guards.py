"""
Unit Tests for Advanced Parameter Extensions and Strict Null/Zero Guards.
Zero-Hype & Truthful Diagnostics under Apache License 2.0.
"""

import os
import sys
import tempfile
import pytest
import numpy as np

from termux_vision.vlm.api import load, VLMContext
from termux_vision.vlm.manifest import ModelManifest, ArtifactInfo
from termux_vision.vlm.runtime.subprocess import SubprocessVLMRuntime
from termux_vision.vlm.model_hub import is_known_remote_model

@pytest.fixture
def mock_runtime():
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
        yield rt, tmpdir

def test_null_image_raises_value_error(mock_runtime):
    rt, _ = mock_runtime
    ctx = VLMContext(rt, rt.manifest)
    with pytest.raises(ValueError, match="Parameter 'image' cannot be null/None"):
        ctx.describe(None, prompt="test")

def test_empty_string_image_raises_value_error(mock_runtime):
    rt, _ = mock_runtime
    ctx = VLMContext(rt, rt.manifest)
    with pytest.raises(ValueError, match="Parameter 'image' cannot be an empty string"):
        ctx.describe("   ", prompt="test")

def test_empty_prompt_raises_value_error(mock_runtime):
    rt, _ = mock_runtime
    ctx = VLMContext(rt, rt.manifest)
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Parameter 'prompt' cannot be an empty string"):
        ctx.describe(dummy_img, prompt="   ")

def test_invalid_max_tokens_guard(mock_runtime):
    rt, _ = mock_runtime
    ctx = VLMContext(rt, rt.manifest)
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Parameter 'max_tokens' must be a positive integer > 0"):
        ctx.describe(dummy_img, prompt="hello", max_tokens=0)

    with pytest.raises(ValueError, match="Parameter 'max_tokens' must be a positive integer > 0"):
        ctx.describe(dummy_img, prompt="hello", max_tokens=-10)

def test_invalid_temperature_guard(mock_runtime):
    rt, _ = mock_runtime
    ctx = VLMContext(rt, rt.manifest)
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Parameter 'temperature' must be a non-negative float"):
        ctx.describe(dummy_img, prompt="hello", temperature=-0.5)

def test_invalid_top_p_and_top_k_guard(mock_runtime):
    rt, _ = mock_runtime
    ctx = VLMContext(rt, rt.manifest)
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Parameter 'top_p' must be within range"):
        ctx.describe(dummy_img, prompt="hello", top_p=1.5)

    with pytest.raises(ValueError, match="Parameter 'top_p' must be within range"):
        ctx.describe(dummy_img, prompt="hello", top_p=-0.1)

    with pytest.raises(ValueError, match="Parameter 'top_k' must be a positive integer"):
        ctx.describe(dummy_img, prompt="hello", top_k=0)

def test_advanced_parameters_passed_to_subprocess(mock_runtime):
    rt, _ = mock_runtime
    recorded_kwargs = {}

    def fake_exec(image_path, prompt, max_tokens, temperature, target_backend, **kwargs):
        recorded_kwargs.update(kwargs)
        recorded_kwargs["max_tokens"] = max_tokens
        recorded_kwargs["temperature"] = temperature
        recorded_kwargs["target_backend"] = target_backend
        from termux_vision.vlm.result import VLMResult, InferenceMetrics
        return VLMResult("ok", "stop", 1, InferenceMetrics("cpu", "test", None, 0.0, 10.0, 10.0, None), ())

    rt._execute_once = fake_exec
    ctx = VLMContext(rt, rt.manifest)
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)

    res = ctx.describe(
        dummy_img,
        prompt="Describe photo",
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        seed=42,
        system_prompt="You are an expert AI vision assistant.",
        stop_tokens=["<|end|>", "###"]
    )

    assert res.text == "ok"
    assert recorded_kwargs["max_tokens"] == 200
    assert recorded_kwargs["temperature"] == 0.7
    assert recorded_kwargs["top_p"] == 0.9
    assert recorded_kwargs["top_k"] == 40
    assert recorded_kwargs["repeat_penalty"] == 1.1
    assert recorded_kwargs["presence_penalty"] == 0.5
    assert recorded_kwargs["frequency_penalty"] == 0.5
    assert recorded_kwargs["seed"] == 42
    assert recorded_kwargs["system_prompt"] == "You are an expert AI vision assistant."
    assert recorded_kwargs["stop_tokens"] == ["<|end|>", "###"]

def test_free_model_hub_remote_detection():
    assert is_known_remote_model("smolvlm-500m") is True
    assert is_known_remote_model("hf:myorg/myrepo:model.gguf") is True
    assert is_known_remote_model("https://huggingface.co/myorg/myrepo/resolve/main/model.gguf") is True
    assert is_known_remote_model("http://example.com/model.gguf") is True

def test_load_model_id_and_model_path_exclusivity():
    from termux_vision.vlm.api import load
    with pytest.raises(ValueError) as excinfo:
        load(model_id="smolvlm-500m-q4", model_path="/some/path.gguf")
    assert "Cannot specify both 'model_id' and 'model_path'" in str(excinfo.value)

