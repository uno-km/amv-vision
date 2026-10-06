"""
AMEVA Unified Exception Hierarchy for Termux AI Engines.
Component: [VISION]
"""
from typing import Optional, Sequence, Any


class AmevaTermuxError(Exception):
    """Root exception for all Termux On-Device AI Engines."""
    COMPONENT_TAG = "[VISION]"
    DEFAULT_CODE = "E000_UNKNOWN"

    def __init__(self, message: str = "", code: Optional[Any] = None, details: Optional[Any] = None):
        self.code = code or self.DEFAULT_CODE
        self.details = details
        self.raw_message = message
        formatted = f"{self.COMPONENT_TAG} [{self.code}] {message}" if self.COMPONENT_TAG not in message else message
        super().__init__(formatted)


TermuxVisionError = AmevaTermuxError


class PlatformNotSupportedError(AmevaTermuxError):
    DEFAULT_CODE = "E002_PLATFORM_NOT_SUPPORTED"


class HardwareCompatibilityError(AmevaTermuxError):
    DEFAULT_CODE = "E003_HARDWARE_INCOMPATIBLE"


class ProvisioningError(AmevaTermuxError):
    DEFAULT_CODE = "E005_PROVISIONING_FAILED"


class InferenceTimeoutError(AmevaTermuxError):
    DEFAULT_CODE = "E006_INFERENCE_TIMEOUT"


class InferenceExecutionError(AmevaTermuxError):
    DEFAULT_CODE = "E007_INFERENCE_FAILED"


class ImageDecodeError(TermuxVisionError):
    pass


class ImageEncodeError(TermuxVisionError):
    pass


class DecompressionBombError(TermuxVisionError):
    pass


class IncompatibleEmbeddingError(TermuxVisionError):
    pass


class InsufficientMemoryError(TermuxVisionError):
    def __init__(self, required_mb: int, available_mb: int, message: str = ""):
        self.required_mb = required_mb
        self.available_mb = available_mb
        msg = message or f"Insufficient memory: required {required_mb}MB, available {available_mb}MB"
        super().__init__(msg)


class ModelNotFoundError(TermuxVisionError):
    DEFAULT_CODE = "E001_MODEL_NOT_FOUND"

    def __init__(
        self,
        message: str = "",
        model_id: Optional[str] = None,
        available_local_models: Sequence[str] = (),
        catalog_models: Sequence[str] = ()
    ):
        self.model_id = model_id
        self.available_local_models = tuple(available_local_models)
        self.catalog_models = tuple(catalog_models)

        if not message:
            parts = []
            if model_id:
                parts.append(f"Model '{model_id}' was not found locally or on remote registry.")
            else:
                parts.append("Requested model was not found.")

            if self.available_local_models:
                parts.append(
                    f"\nCurrently installed local models ({len(self.available_local_models)}):\n"
                    + "\n".join(f"  - {m}" for m in self.available_local_models)
                )
                parts.append(
                    f"\nTo run with an installed model:\n"
                    f"  termux-vision vlm <IMAGE> --model {self.available_local_models[0]}"
                )
            elif self.catalog_models:
                parts.append(
                    f"\nNo models installed locally. Available catalog presets:\n"
                    + "\n".join(f"  - {m}" for m in self.catalog_models)
                )
                parts.append(
                    f"\nInstall an official model:\n"
                    f"  termux-vision model install {self.catalog_models[0]}"
                )
            message = "\n".join(parts)

        super().__init__(message)


class NoInstalledModelsError(ModelNotFoundError):
    def __init__(self, catalog_models: Sequence[str] = ()):
        self.catalog_models = tuple(catalog_models)
        cat_text = "\n".join(f"  - {m}" for m in sorted(self.catalog_models)) if self.catalog_models else "  (none)"
        msg = (
            "No installed VLM models were found in local cache (~/.cache/termux-vision/models).\n\n"
            "How to use VLM models:\n"
            "1. Install an official catalog model:\n"
            f"{cat_text}\n"
            f"   Example: termux-vision model install smolvlm-500m-q4\n\n"
            "2. Use custom/external ('싸제') models:\n"
            "   - Place your text model GGUF and vision projector mmproj GGUF in:\n"
            "     ~/.cache/termux-vision/models/<custom_model_name>/\n"
            "   - Or pass direct file paths:\n"
            "     termux-vision vlm <IMAGE> --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf\n"
            "   (Note: VLM inference requires both a language model .gguf and a vision projector mmproj-*.gguf)"
        )
        super().__init__(message=msg, catalog_models=self.catalog_models)


class ModelSelectionRequiredError(ModelNotFoundError):
    def __init__(self, installed_models: Sequence[str] = ()):
        self.installed_models = tuple(installed_models)
        models_text = "\n".join(f"  - {m}" for m in self.installed_models)
        example_model = self.installed_models[0] if self.installed_models else "MODEL_ID"
        msg = (
            f"Multiple models are installed. Please specify one with --model:\n"
            f"{models_text}\n\n"
            f"Example:\n"
            f"  termux-vision vlm <IMAGE> --model {example_model} -p \"Describe this image\""
        )
        super().__init__(message=msg, available_local_models=self.installed_models)


class ModelArtifactsMissingError(ModelNotFoundError):
    def __init__(self, model_id: str, missing_artifacts: Sequence[str] = ()):
        self.model_id = model_id
        self.missing_artifacts = tuple(missing_artifacts)
        missing_text = ", ".join(missing_artifacts)
        msg = (
            f"Model '{model_id}' is incomplete. Missing artifacts: {missing_text}\n"
            f"VLM models require both a language model (.gguf) and a vision encoder (mmproj-*.gguf).\n\n"
            f"To repair/re-download:\n"
            f"  termux-vision model remove {model_id}\n"
            f"  termux-vision model install {model_id}"
        )
        super().__init__(message=msg, model_id=model_id)


class ModelCorruptedError(TermuxVisionError):
    pass


class ModelDownloadError(TermuxVisionError):
    def __init__(self, model_source: str, reason: str = "", available_local: Sequence[str] = ()):
        self.model_source = model_source
        self.reason = reason
        self.available_local = tuple(available_local)
        
        parts = [f"Failed to download model from '{model_source}': {reason}"]
        if self.available_local:
            parts.append(
                f"\nInstalled local models available:\n"
                + "\n".join(f"  - {m}" for m in self.available_local)
                + f"\nRun with local model: termux-vision vlm <IMAGE> --model {self.available_local[0]}"
            )
        else:
            parts.append(
                "\nNo models found in local cache. You can place GGUF files in ~/.cache/termux-vision/models/ "
                "or specify explicit file paths with --model <model.gguf> --mmproj <mmproj.gguf>."
            )
        super().__init__("\n".join(parts))


class SubprocessRuntimeError(TermuxVisionError):
    pass


class RuntimeNotFoundError(TermuxVisionError):
    def __init__(self, executable: str = "llama-cli", searched_paths: Sequence[str] = ()):
        self.executable = executable
        self.searched_paths = tuple(searched_paths)
        searched = ""
        if searched_paths:
            searched = "\n\nSearched paths:\n" + "\n".join(f"  - {path}" for path in searched_paths)
        message = (
            f"Required runtime engine '{executable}' was not found.{searched}\n\n"
            f"Please install the official termux-llamacpp native runtime:\n"
            f"  - Python:  pip install termux-llamacpp\n"
            f"  - Node.js: npm install -g termux-llamacpp\n"
            f"  - Setup:   termux-llama install (compiles native binary if needed)"
        )
        super().__init__(message)


class VulkanNotAvailableError(SubprocessRuntimeError):
    def __init__(self, reason: str = ""):
        self.reason = reason
        detail = f"\nFailure detail: {reason}" if reason else ""
        msg = (
            f"Vulkan GPU acceleration is unavailable or failed on this device.{detail}\n\n"
            f"[Action Required] Explicit GPU mode cannot proceed. "
            f"Please switch to CPU mode:\n"
            f"  CLI: --device cpu\n"
            f"  Python API: device='cpu'\n"
            f"Or use automatic detection: --device auto (device='auto')"
        )
        super().__init__(msg)


class GpuExecutionError(VulkanNotAvailableError):
    pass


class CameraPermissionError(TermuxVisionError):
    pass


class TermuxAPIUnavailableError(TermuxVisionError):
    pass


class OpenCLNotImplementedError(TermuxVisionError):
    DEFAULT_CODE = "E014_OPENCL_NOT_IMPLEMENTED"

    def __init__(self, reason: str = ""):
        self.reason = reason
        detail = f"\nDetail: {reason}" if reason else ""
        msg = (
            f"OpenCL compute backend is not currently implemented in this release.{detail}\n\n"
            f"[Hardware Context] OpenCL vision compute pipeline is reserved for ARM Mali/Exynos devices "
            f"(e.g., Galaxy S20, A53). Tracked under Jira [SCRUM-VISION-OPENCL].\n"
            f"  - Use Vulkan GPU: --gpu or --device vulkan\n"
            f"  - Use CPU NEON:  --cpu or --device cpu"
        )
        super().__init__(msg)


class IncompleteRuntimeBinaryError(TermuxVisionError):
    DEFAULT_CODE = "E015_INCOMPLETE_RUNTIME_BINARY"

    def __init__(self, binary_path: str, reason: str = ""):
        self.binary_path = binary_path
        self.reason = reason
        detail = f"\nDetail: {reason}" if reason else ""
        msg = (
            f"Native binary at '{binary_path}' lacks required multimodal vision capabilities (--mmproj missing).{detail}\n\n"
            f"[Root Cause] Detected text-only build artifact (e.g. llama-completion ~4.9MB) instead of full multimodal runtime (>=5.8MB).\n"
            f"[Action Required] Run official installer to provision verified SOTA prebuilt assets:\n"
            f"  termux-vision install -y\n"
            f"Or compile from source: termux-vision install --from-source"
        )
        super().__init__(msg)


class NativeBuildError(TermuxVisionError):
    DEFAULT_CODE = "E016_NATIVE_BUILD_FAILED"

    def __init__(self, target: str, error_log: str = ""):
        self.target = target
        self.error_log = error_log
        log_snippet = f"\nCompiler Output:\n{error_log}" if error_log else ""
        msg = (
            f"Failed to compile native C++ vision module '{target}'.{log_snippet}\n\n"
            f"[Action Required] Ensure clang and build prerequisites are installed:\n"
            f"  pkg install -y clang build-essential\n"
            f"Or use prebuilt release assets: termux-vision install"
        )
        super().__init__(msg)


class InstallationSmokeTestError(TermuxVisionError):
    DEFAULT_CODE = "E017_INSTALL_SMOKE_TEST_FAILED"

    def __init__(self, test_name: str, reason: str = ""):
        self.test_name = test_name
        self.reason = reason
        msg = (
            f"Installation smoke test failed for component '{test_name}': {reason}\n"
            f"The environment cannot be certified as production-ready. Please review diagnostic logs."
        )
        super().__init__(msg)


class ClusterConnectionError(TermuxVisionError):
    """Raised when one or more distributed RPC worker nodes cannot be reached or fail handshake."""
    DEFAULT_CODE = "E010_CLUSTER_CONNECTION"


class ClusterConfigurationError(TermuxVisionError):
    """Raised when cluster parameters (tensor split, backend, vram budget) are invalid."""
    DEFAULT_CODE = "E011_CLUSTER_CONFIG"

