from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

@dataclass(frozen=True)
class InferenceMetrics:
    backend: str
    model_id: str
    load_ms: Optional[float]
    vision_ms: float
    decode_ms: float
    tokens_per_second: Optional[float]
    peak_rss_mb: Optional[float]

@dataclass(frozen=True)
class VLMResult:
    text: str
    finish_reason: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    word_count: Optional[int] = None
    metrics: Optional[InferenceMetrics] = None
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "word_count": self.word_count,
            "metrics": asdict(self.metrics) if self.metrics else None,
            "warnings": list(self.warnings)
        }
