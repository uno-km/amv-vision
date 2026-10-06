"""
termux_vision.control.status — Vision 상태 파일 Writer + Heartbeat
"""
from __future__ import annotations
from typing import Any
from ameva_component.heartbeat import HeartbeatWriter


class VisionStatusWriter(HeartbeatWriter):
    """VisionControl 상태를 10초마다 상태 파일에 원자적으로 기록합니다."""

    def __init__(self, control: Any) -> None:
        super().__init__(control, name="vision")
