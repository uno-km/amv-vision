from dataclasses import dataclass
import numpy as np
from ..errors import IncompatibleEmbeddingError

@dataclass(frozen=True)
class Embedding:
    """
    Type-safe, normalized feature embedding vector.
    """
    values: np.ndarray
    model_id: str
    dimension: int
    normalized: bool = True

    def __post_init__(self):
        arr = np.ascontiguousarray(self.values.flatten(), dtype=np.float32)
        if len(arr) != self.dimension:
            raise ValueError(f"Embedding length ({len(arr)}) does not match dimension ({self.dimension}).")
        
        if self.normalized:
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            else:
                arr = np.zeros_like(arr)
        
        object.__setattr__(self, "values", arr)

def compute_similarity(a: Embedding, b: Embedding) -> float:
    """
    Computes cosine similarity between two verified embeddings from the SAME model space.
    """
    if a.model_id != b.model_id:
        raise IncompatibleEmbeddingError(
            f"Cannot compute similarity between different embedding models: '{a.model_id}' vs '{b.model_id}'"
        )
    if a.dimension != b.dimension:
        raise IncompatibleEmbeddingError(
            f"Embedding dimension mismatch: {a.dimension} vs {b.dimension}"
        )

    # Dot product of L2-normalized vectors is exact Cosine Similarity
    sim = float(np.dot(a.values, b.values))
    return max(-1.0, min(1.0, sim))
