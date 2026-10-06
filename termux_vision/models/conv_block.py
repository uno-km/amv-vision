import warnings
import numpy as np

class DepthwiseSeparableConv2D:
    """
    Lightweight MobileNet-style Depthwise Separable 2D Convolution Layer.
    Splits standard conv into:
    1. Depthwise Conv (Spatial filtering per channel)
    2. Pointwise Conv (1x1 Linear combination across channels)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, weights: dict = None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if weights is not None:
            self.dw_weights = np.asarray(weights["dw"], dtype=np.float32)
            self.pw_weights = np.asarray(weights["pw"], dtype=np.float32)
            self.bias = np.asarray(weights.get("bias", np.zeros(out_channels, dtype=np.float32)), dtype=np.float32)
            self._weights_loaded = True
        else:
            # Deterministic Kaiming Normal initialization with fixed standard normal scaling
            scale_dw = np.sqrt(2.0 / (kernel_size * kernel_size * in_channels))
            scale_pw = np.sqrt(2.0 / in_channels)
            rng = np.random.RandomState(42 + in_channels * 100 + out_channels)
            self.dw_weights = (rng.randn(in_channels, kernel_size, kernel_size) * scale_dw).astype(np.float32)
            self.pw_weights = (rng.randn(out_channels, in_channels) * scale_pw).astype(np.float32)
            self.bias = np.zeros(out_channels, dtype=np.float32)
            self._weights_loaded = False
            warnings.warn(
                f"DepthwiseSeparableConv2D ({in_channels}->{out_channels}) initialized with untrained random weights (Kaiming Normal, Seed 42). "
                "Inference output is non-deterministic / uncalibrated. Load trained checkpoint weights for production inference.",
                UserWarning,
                stacklevel=2
            )

    @property
    def is_trained(self) -> bool:
        """Returns True if weights were explicitly loaded from a trained checkpoint."""
        return self._weights_loaded

    def set_weights(self, dw_weights: np.ndarray, pw_weights: np.ndarray, bias: np.ndarray = None):
        self.dw_weights = np.asarray(dw_weights, dtype=np.float32)
        self.pw_weights = np.asarray(pw_weights, dtype=np.float32)
        if bias is not None:
            self.bias = np.asarray(bias, dtype=np.float32)
        self._weights_loaded = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Input shape: (C_in, H, W) or (B, C_in, H, W)
        Output shape: (C_out, H_out, W_out) or (B, C_out, H_out, W_out)
        """
        has_batch = (x.ndim == 4)
        if not has_batch:
            x_batched = x[np.newaxis, ...]
        else:
            x_batched = x

        b, c_in, h, w = x_batched.shape
        k = self.kernel_size
        pad = k // 2

        # 1. Depthwise Convolution (Same padding)
        padded = np.pad(x_batched, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, (k, k), axis=(2, 3))
        # windows shape: (B, C_in, H, W, k, k)
        
        # Strided slicing if stride > 1
        if self.stride > 1:
            windows = windows[:, :, ::self.stride, ::self.stride, :, :]

        # Depthwise output: (B, C_in, H_out, W_out)
        dw_out = np.einsum('bchwkl,ckl->bchw', windows, self.dw_weights)
        
        # ReLU6 activation
        dw_out = np.clip(dw_out, 0.0, 6.0)

        # 2. Pointwise Convolution (1x1): (B, C_out, H_out, W_out)
        pw_out = np.einsum('bchw,oc->bohw', dw_out, self.pw_weights) + self.bias[None, :, None, None]

        if not has_batch:
            return pw_out[0]
        return pw_out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
