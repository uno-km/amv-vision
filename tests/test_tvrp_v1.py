import pytest
import struct
from termux_vision.vlm.protocol import TVRPRequest, TVRPResponse, ProtocolError, MAX_CONTROL_MESSAGE_BYTES
from termux_vision.vlm.memory import MemoryEstimate, check_memory_admission
from termux_vision import errors

def test_tvrp_request_roundtrip():
    req = TVRPRequest(
        protocol_version=1,
        request_id="req-test-123",
        op="describe",
        model_id="smolvlm-500m-q4",
        image_path="/tmp/test.jpg",
        prompt="Describe image",
        options={"max_tokens": 64, "threads": 4}
    )
    raw = req.serialize()
    assert len(raw) > 4
    length = struct.unpack("<I", raw[:4])[0]
    assert length == len(raw) - 4

    deserialized = TVRPRequest.deserialize(raw)
    assert deserialized.protocol_version == 1
    assert deserialized.request_id == "req-test-123"
    assert deserialized.op == "describe"
    assert deserialized.prompt == "Describe image"
    assert deserialized.options["max_tokens"] == 64

def test_tvrp_response_roundtrip():
    res = TVRPResponse(
        protocol_version=1,
        request_id="req-test-123",
        status="ok",
        result={"text": "A person on a beach", "output_tokens": 10},
        metrics={"backend": "cpu", "tokens_per_second": 12.5},
        error=None,
        warnings=("Test warning",)
    )
    raw = res.serialize()
    deserialized = TVRPResponse.deserialize(raw)
    assert deserialized.status == "ok"
    assert deserialized.result["text"] == "A person on a beach"
    assert deserialized.warnings == ("Test warning",)

def test_tvrp_max_message_size_guard():
    huge_prompt = "A" * (MAX_CONTROL_MESSAGE_BYTES + 10)
    req = TVRPRequest(
        protocol_version=1,
        request_id="req-huge",
        op="describe",
        model_id="smolvlm-500m-q4",
        image_path="/tmp/test.jpg",
        prompt=huge_prompt,
        options={}
    )
    with pytest.raises(ProtocolError):
        req.serialize()

def test_memory_admission_margins():
    est = MemoryEstimate(
        model_weights_mb=400,
        vision_encoder_mb=200,
        kv_cache_mb=100,
        compute_buffers_mb=30,
        runtime_overhead_mb=20,
        estimated_peak_mb=750,
        confidence="estimated"
    )
    # Check admission with warn policy (passes with warning if budget low)
    admitted, warn = check_memory_admission(est, user_budget_mb=1500, memory_policy="warn")
    assert admitted is True

    # Strict policy must raise InsufficientMemoryError when budget is lower than peak
    with pytest.raises(errors.InsufficientMemoryError):
        check_memory_admission(est, user_budget_mb=500, memory_policy="strict")
