from typing import Dict, Any, Type
from .base import VLMModelAdapter
from .smolvlm import SmolVLMAdapter
from .qwen2vl import Qwen2VLAdapter

SHIPPED_ADAPTER_REGISTRY: Dict[str, Type] = {
    "smolvlm": SmolVLMAdapter,
    "qwen2vl": Qwen2VLAdapter
}

def get_adapter(adapter_name: str) -> VLMModelAdapter:
    name = adapter_name.lower().strip()
    if name in SHIPPED_ADAPTER_REGISTRY:
        return SHIPPED_ADAPTER_REGISTRY[name]()
    elif "qwen" in name:
        return Qwen2VLAdapter()
    return SmolVLMAdapter()
