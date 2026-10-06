from typing import Protocol, Dict, Any, Optional
import numpy as np

class VLMModelAdapter(Protocol):
    """
    Protocol defining model-specific prompt templates, image preprocessors,
    and special token encapsulation.
    """
    adapter_name: str

    def format_prompt(self, user_prompt: str, task: str = "describe") -> str:
        """Formats user query into model-specific ChatML / special token format."""
        ...

    def get_preferred_resolution(self) -> int:
        """Returns optimal downsampling square resolution (e.g. 384 or 512)."""
        ...
