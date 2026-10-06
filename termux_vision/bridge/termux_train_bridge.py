import numpy as np

def to_termux_tensor(data, requires_grad: bool = False):
    """
    Convert a NumPy image array or patch matrix to termux_train.Tensor.
    Gracefully imports termux_train or creates equivalent autograd structure.
    """
    try:
        import termux_train as tt
        if isinstance(data, np.ndarray):
            # Direct flat C-buffer pass without .tolist() to eliminate 8x memory expansion and LMK kills
            return tt.Tensor(data, dtype="float32", requires_grad=requires_grad)
        elif isinstance(data, (list, tuple)):
            return tt.Tensor(data, dtype="float32", requires_grad=requires_grad)
        elif hasattr(data, "to_numpy"):
            return tt.Tensor(data.to_numpy(), dtype="float32", requires_grad=requires_grad)
        else:
            return tt.Tensor(data, dtype="float32", requires_grad=requires_grad)
    except ImportError:
        # Fallback if termux-train is not installed in current environment
        class StandaloneVisionTensor:
            def __init__(self, arr, requires_grad=False):
                self.data = np.array(arr, dtype=np.float32)
                self.shape = self.data.shape
                self.requires_grad = requires_grad
                self.grad = None
            def to_numpy(self):
                return self.data
            def __repr__(self):
                return f"StandaloneVisionTensor(shape={self.shape}, dtype={self.data.dtype})"
        return StandaloneVisionTensor(data, requires_grad=requires_grad)

def from_termux_tensor(tensor) -> np.ndarray:
    """
    Convert a termux_train.Tensor or StandaloneVisionTensor back to NumPy array.
    """
    if hasattr(tensor, "to_numpy"):
        return np.asarray(tensor.to_numpy(), dtype=np.float32)
    elif hasattr(tensor, "numpy"):
        return np.asarray(tensor.numpy(), dtype=np.float32)
    elif hasattr(tensor, "tolist"):
        return np.asarray(tensor.tolist(), dtype=np.float32)
    elif hasattr(tensor, "_data"):
        if hasattr(tensor._data, "to_list"):
            return np.asarray(tensor._data.to_list(), dtype=np.float32)
        return np.asarray(tensor._data, dtype=np.float32)
    elif hasattr(tensor, "data"):
        if isinstance(tensor.data, np.ndarray):
            return np.asarray(tensor.data, dtype=np.float32)
        return np.asarray(tensor.data, dtype=np.float32)
    elif isinstance(tensor, np.ndarray):
        return np.asarray(tensor, dtype=np.float32)
    else:
        return np.asarray(tensor, dtype=np.float32)

class VisionPatchDataset:
    """
    On-Device Dataset for batching image patches for ViT or classifier training.
    """
    def __init__(self, patches: np.ndarray, labels: np.ndarray = None, batch_size: int = 32, shuffle: bool = True):
        self.patches = np.asarray(patches, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32) if labels is not None else None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.patches)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            batch_x = self.patches[batch_idx]
            batch_tensor_x = to_termux_tensor(batch_x)
            
            if self.labels is not None:
                batch_y = self.labels[batch_idx]
                batch_tensor_y = to_termux_tensor(batch_y)
                yield batch_tensor_x, batch_tensor_y
            else:
                yield batch_tensor_x
