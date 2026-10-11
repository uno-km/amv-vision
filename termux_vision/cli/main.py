# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import argparse
import sys
import os
import json
import time
from typing import Optional

from .. import __version__
from ..io.loader import load_image, save_image, get_image_info
from ..transforms.functional import resize, to_grayscale
from ..cv.filters import canny
from ..cv.crop import crop
from ..detect.haar import HaarCascadeDetector, detect_faces
from .doctor import run_doctor
from ..vlm.cache import ModelCacheManager, CATALOG
from ..vlm.model_hub import download_custom_url_or_hf
from ..errors import (
    RuntimeNotFoundError,
    NoInstalledModelsError,
    ModelSelectionRequiredError,
    ModelArtifactsMissingError,
    ModelNotFoundError,
    ModelDownloadError,
    VulkanNotAvailableError,
    InsufficientMemoryError,
    ClusterConnectionError,
    ClusterConfigurationError,
)
from termux_vision.exceptions import RuntimeRequiredError

EXIT_SUCCESS = 0
EXIT_ARG_ERROR = 2
EXIT_MODEL_NOT_FOUND = 10
EXIT_MODEL_VERIFY_FAIL = 11
EXIT_INSUFFICIENT_MEMORY = 12
EXIT_IMAGE_DECODE_FAIL = 14
EXIT_INFERENCE_FAIL = 15
EXIT_RUNTIME_NOT_FOUND = 20
EXIT_NO_INSTALLED_MODELS = 21
EXIT_MODEL_INCOMPLETE = 22
EXIT_MODEL_SELECTION_REQUIRED = 23
EXIT_GPU_ERROR = 24
EXIT_CLUSTER_ERROR = 25

def _print_available_models(cache: ModelCacheManager) -> None:
    installed = cache.list_installed()
    print("\nInstalled models in local cache (~/.cache/termux-vision/models):", file=sys.stderr)
    if not installed:
        print("  (none - local cache is empty)", file=sys.stderr)
    else:
        for index, item in enumerate(installed, start=1):
            print(f"  {index}. {item['model_id']}", file=sys.stderr)
            print(f"     State: {item.get('state', 'UNKNOWN')}", file=sys.stderr)
            print(f"     Size: {item.get('size_mb', 0)} MB", file=sys.stderr)
            print(f"     Path: {item.get('path', 'unknown')}", file=sys.stderr)

    print("\nOfficial catalog presets:", file=sys.stderr)
    for model_id, manifest in sorted(CATALOG.items()):
        marker = "installed" if any(item["model_id"] == model_id for item in installed) else "not installed"
        print(f"  - {model_id} [{manifest.tier}, {marker}]", file=sys.stderr)

def _print_runtime_install_help() -> None:
    print("\n[Runtime Installation Guide]", file=sys.stderr)
    print("  To install llama-cli on Termux:", file=sys.stderr)
    print("    pkg install termux-llamacpp", file=sys.stderr)
    print("  Or run the one-touch installer:", file=sys.stderr)
    print("    bash install.sh", file=sys.stderr)

def resolve_model_id(cache: ModelCacheManager, requested_model_id: Optional[str], interactive: bool = True) -> str:
    if requested_model_id:
        expanded = os.path.abspath(os.path.expanduser(requested_model_id))
        if os.path.isfile(expanded):
            return expanded

        installed = cache.list_installed()
        installed_ids = tuple(sorted(item["model_id"] for item in installed))
        if requested_model_id not in installed_ids and requested_model_id not in CATALOG:
            raise ModelNotFoundError(
                model_id=requested_model_id,
                available_local_models=installed_ids,
                catalog_models=sorted(CATALOG.keys())
            )
        return requested_model_id

    installed = cache.list_installed()
    installed_ids = tuple(sorted(item["model_id"] for item in installed))

    if len(installed_ids) == 1:
        selected = installed_ids[0]
        print(f"[INFO] Selected installed model: {selected}", file=sys.stderr)
        return selected

    if len(installed_ids) > 1:
        raise ModelSelectionRequiredError(installed_models=installed_ids)

    # Empty cache: Check interactive download (Requirement 4)
    if interactive and sys.stdin.isatty():
        default_model = "smolvlm-500m-q4"
        print("\n---------------------------------------------------------", file=sys.stderr)
        print("  [Notice] No local VLM model is currently installed.", file=sys.stderr)
        print(f"  Default Model : {default_model} (SmolVLM 500M Instruct)", file=sys.stderr)
        print(f"  Download Size : ~550 MB", file=sys.stderr)
        print(f"  Target Path   : ~/.cache/termux-vision/models/{default_model}/", file=sys.stderr)
        print("---------------------------------------------------------", file=sys.stderr)
        try:
            ans = input("Do you want to download and install this model now? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans in ("y", "yes"):
            print(f"[*] Downloading {default_model} (~550MB)...", file=sys.stderr)
            cache.install(default_model)
            print(f"[+] Successfully installed '{default_model}'.", file=sys.stderr)
            return default_model

    raise NoInstalledModelsError(catalog_models=sorted(CATALOG.keys()))

def main():
    """CLI entry point.

    A GPU request without a usable ameva-runtime ends here with the shared explanation on stderr
    and exit status 69, whichever sub-command raised it.
    """
    from termux_vision.exceptions import RuntimeRequiredError
    try:
        return _main_impl()
    except RuntimeRequiredError as err:
        sys.stderr.write(str(err))
        sys.stderr.flush()
        return 69


def _main_impl():
    parser = argparse.ArgumentParser(
        prog="termux-vision",
        description=f"termux-vision CLI v{__version__}: High-performance on-device Computer Vision & VLM Framework for Android Termux."
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: doctor
    p_doc = subparsers.add_parser("doctor", help="Inspect device hardware, Vulkan GPU, and runtime environment")
    p_doc.add_argument("--json", action="store_true", help="Output diagnostic report in JSON format")
    p_doc.add_argument("--probe-vulkan", action="store_true", help="Inspect Vulkan driver presence")
    p_doc.add_argument("--full", action="store_true", help="Run full model integrity check")

    # Command: model
    p_model = subparsers.add_parser("model", help="Manage VLM model artifacts and cache")
    p_model_sub = p_model.add_subparsers(dest="model_action", help="Model subcommands")
    
    p_m_list = p_model_sub.add_parser("list", help="List installed on-device models")
    p_m_list.add_argument("--json", action="store_true", help="Output list in JSON format")

    p_m_install = p_model_sub.add_parser("install", help="Download and verify an official catalog VLM model")
    p_m_install.add_argument("model_id", help="Model identifier (e.g. smolvlm-500m-q4, qwen2-vl-2b-q4)")

    p_m_download = p_model_sub.add_parser("download", help="Freely download any model from Hugging Face or direct URL (Requirement 5)")
    p_m_download.add_argument("source", help="Direct HTTP(S) URL or Hugging Face repo (e.g. hf:owner/repo:file.gguf or https://...)")
    p_m_download.add_argument("-o", "--output", default=None, help="Target destination directory")

    p_m_remove = p_model_sub.add_parser("remove", help="Delete a model from local cache")
    p_m_remove.add_argument("model_id", help="Model identifier to delete")

    # Command: info
    p_info = subparsers.add_parser("info", help="Inspect image metadata")
    p_info.add_argument("image_path", help="Path to input image")

    # Command: canny
    p_canny = subparsers.add_parser("canny", help="Apply 5-stage Canny Edge Detection")
    p_canny.add_argument("image_path", help="Path to input image")
    p_canny.add_argument("-o", "--output", default="edges.png", help="Output path for edge image")
    p_canny.add_argument("--low", type=float, default=40.0, help="Low hysteresis threshold")
    p_canny.add_argument("--high", type=float, default=120.0, help="High hysteresis threshold")
    p_canny.add_argument("--resize", type=str, default=None, help="Resize image (e.g. 512x512)")
    p_canny.add_argument("-d", "--device", "-b", "--backend", dest="device", default="auto", choices=["auto", "gpu", "vulkan", "opencl", "cpu"], help="Compute backend device")
    p_canny.add_argument("--gpu", action="store_const", const="gpu", dest="device", help="Force 100%% Vulkan GPU Compute")
    p_canny.add_argument("--cpu", action="store_const", const="cpu", dest="device", help="Force 100%% CPU ARM64 NEON")
    p_canny.add_argument("--opencl", action="store_const", const="opencl", dest="device", help="Force OpenCL Compute")

    # Command: detect-face
    p_face = subparsers.add_parser("detect-face", help="Detect face regions via deep learning or cascade")
    p_face.add_argument("image_path", help="Path to input image")
    p_face.add_argument("-o", "--output", default="face_crop.jpg", help="Output path for primary face crop")
    p_face.add_argument("--json", action="store_true", help="Output detection boxes in JSON format")
    p_face.add_argument("--backend", default="auto", choices=["auto", "neural", "haar"], help="Face detection engine (default: auto -> neural)")
    p_face.add_argument("--score-threshold", type=float, default=0.70, help="Confidence score threshold for neural detector (default: 0.70)")

    # Command: vlm
    p_vlm = subparsers.add_parser("vlm", help="Run VLM Multimodal Image Description / QA with full parameter control")
    p_vlm.add_argument("image_path", help="Path to input image")
    p_vlm.add_argument("-p", "--prompt", default=None, help="Text prompt query")
    p_vlm.add_argument("-m", "--model", default=None, help="Catalog model preset identifier (e.g. 'smolvlm-500m-q4')")
    p_vlm.add_argument("--model-path", default=None, help="Explicit file path to custom .gguf language model")
    p_vlm.add_argument("--mmproj", default=None, help="Path to vision projector model (mmproj-*.gguf)")
    p_vlm.add_argument("-d", "--device", "-b", "--backend", dest="device", default="auto", choices=["auto", "gpu", "vulkan", "opencl", "cpu"], help="Device backend")
    p_vlm.add_argument("--gpu", action="store_const", const="gpu", dest="device", help="Force Vulkan GPU acceleration")
    p_vlm.add_argument("--cpu", action="store_const", const="cpu", dest="device", help="Force ARM64 CPU execution")
    p_vlm.add_argument("--opencl", action="store_const", const="opencl", dest="device", help="Force OpenCL backend")
    p_vlm.add_argument("--runtime", default=None, help="Explicit path to llama-cli executable")
    p_vlm.add_argument("--memory-policy", default="warn", choices=["warn", "strict", "unrestricted"], help="Memory admission policy")
    p_vlm.add_argument("--allow-download", action="store_true", help="Automatically download model if missing from cache")
    p_vlm.add_argument("-t", "--threads", default="auto", help="Inference threads or 'auto'")
    p_vlm.add_argument("-n", "--max-tokens", type=int, default=150, help="Maximum generated tokens")
    p_vlm.add_argument("--temp", "--temperature", dest="temperature", type=float, default=0.2, help="Sampling temperature")
    p_vlm.add_argument("--top-p", type=float, default=None, help="Top-p nucleus sampling (0.0 < p <= 1.0)")
    p_vlm.add_argument("--top-k", type=int, default=None, help="Top-k sampling threshold")
    p_vlm.add_argument("--repeat-penalty", type=float, default=1.2, help="Repetition penalty")
    p_vlm.add_argument("--presence-penalty", type=float, default=None, help="Presence penalty")
    p_vlm.add_argument("--frequency-penalty", type=float, default=None, help="Frequency penalty")
    p_vlm.add_argument("--seed", type=int, default=None, help="Random RNG seed")
    p_vlm.add_argument("--system-prompt", default=None, help="System prompt context")
    p_vlm.add_argument("-q", "--quality", choices=["original", "high", "optimal", "fast"], default="optimal", help="Image quality scaling preset (original: raw 1:1, high: 1280px, optimal: 768px, fast: 384px)")
    p_vlm.add_argument("--max-dim", type=int, default=None, help="Explicit maximum image dimension in pixels (aspect ratio preserved)")
    p_vlm.add_argument("-c", "--ctx-size", "--ctx", dest="ctx_size", type=int, default=None, help="Context size")
    p_vlm.add_argument("--ngl", type=int, default=None, help="Number of GPU offload layers")
    p_vlm.add_argument("--rpc", type=str, default=None, help="Distributed RPC server addresses (e.g. '192.168.0.220:50052,192.168.0.253:50052')")
    p_vlm.add_argument("-ts", "--tensor-split", type=str, default=None, help="Fraction of the model to offload across devices (e.g. '50,50' or '60,40')")
    p_vlm.add_argument("--cluster-rpc-servers", type=str, default=None, help="AMEVA cluster RPC server endpoints (comma-separated host:port)")
    p_vlm.add_argument("--cluster-split-mode", type=str, default=None, help="AMEVA cluster split mode (e.g. 'layer', 'tensor')")
    p_vlm.add_argument("--cluster-tensor-split", type=str, default=None, help="AMEVA cluster tensor split ratios (e.g. '50,50')")
    p_vlm.add_argument("--json", action="store_true", help="Output full result and metrics in JSON format")

    # Command: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run on-device vision benchmark")
    p_bench.add_argument("image_path", nargs="?", default=None, help="Optional image path for VLM bench")
    p_bench.add_argument("-m", "--model", default="smolvlm-500m-q4", help="Model ID for benchmark")
    p_bench.add_argument("-d", "--device", "-b", "--backend", dest="device", default="auto", choices=["auto", "gpu", "vulkan", "opencl", "cpu"], help="Compute backend device")
    p_bench.add_argument("--gpu", action="store_const", const="gpu", dest="device", help="Force 100%% Vulkan GPU Compute")
    p_bench.add_argument("--cpu", action="store_const", const="cpu", dest="device", help="Force 100%% CPU ARM64 NEON")
    p_bench.add_argument("--opencl", action="store_const", const="opencl", dest="device", help="Force OpenCL Compute")
    p_bench.add_argument("--memory-policy", default="unrestricted", choices=["warn", "strict", "unrestricted"])
    p_bench.add_argument("--runs", type=int, default=3, help="Benchmark run count")
    p_bench.add_argument("--json", action="store_true", help="Output benchmark results in JSON format")

    # Command: install
    p_inst = subparsers.add_parser("install", help="Verify and provision native vision engines and multimodal VLM runtime")
    p_inst.add_argument("-y", "-Y", "--yes", "-yes", "--all", "-a", dest="auto_yes", action="store_true", help="Automatically approve installation and updates without interactive prompts")
    p_inst.add_argument("--from-source", action="store_true", help="Force local native C++ compilation via Clang/NEON instead of prebuilt assets")
    p_inst.add_argument("--force", action="store_true", help="Force reinstallation even if assets are up-to-date")

    # ── AMEVA Component Protocol v1 ─────────────────────────────────────────
    try:
        from ameva_component.cli_support import build_protocol_subcommands
        build_protocol_subcommands(subparsers)
    except (ImportError, Exception) as _proto_err:
        import logging
        logging.getLogger("termux_vision.cli").debug(
            "Optional AMEVA component CLI protocol subcommands omitted: %s", _proto_err
        )
    # ────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    if args.command == "install":
        from ..installer import install_all
        try:
            ok = install_all(
                from_source=getattr(args, "from_source", False),
                force=getattr(args, "force", False),
                auto_yes=getattr(args, "auto_yes", False),
                verbose=True,
            )
            sys.exit(0 if ok else 1)
        except RuntimeRequiredError:
            raise
        except Exception as inst_err:
            print(f"\n[ERROR: Installation Failed] {inst_err}", file=sys.stderr)
            sys.exit(1)

    if args.command == "doctor":
        rep = run_doctor(probe_vulkan=getattr(args, "probe_vulkan", False), full_check=getattr(args, "full", False))
        if args.json:
            print(json.dumps(rep, indent=2))
        else:
            print("=== termux-vision Diagnostic Doctor ===")
            print(f"  Platform : {rep['platform']['system']} ({rep['platform']['machine']}) | Android: {rep['platform']['is_android']}")
            print(f"  RAM      : Total {rep['hardware']['total_ram_mb']}MB | Available {rep['hardware']['available_ram_mb']}MB")
            print(f"  CPU Cores: {rep['hardware']['cpu_cores']}")
            print(f"  Vulkan   : Loader={rep['vulkan']['loader_detected']} | Driver={rep['vulkan']['driver_file_detected']} | Status={rep['vulkan']['status']}")
            print(f"  Models   : {rep['vlm_runtime']['installed_models_count']} installed in {rep['vlm_runtime']['cache_dir']}")
            print(f"  Preset   : {rep['recommended_preset']}")
        sys.exit(EXIT_SUCCESS)

    elif args.command == "model":
        cache = ModelCacheManager()
        if args.model_action == "list":
            inst = cache.list_installed()
            if args.json:
                print(json.dumps(inst, indent=2))
            else:
                print(f"=== Installed VLM Models ({len(inst)}) ===")
                if not inst:
                    print("  (none in ~/.cache/termux-vision/models)")
                else:
                    for m in inst:
                        print(f"  - {m['model_id']:<20} | Tier: {m['tier']} | State: {m.get('state', 'READY')} | Size: {m['size_mb']}MB")
                print("\nAvailable Official Presets:")
                for k, v in CATALOG.items():
                    print(f"  * {k:<20} | Tier: {v.tier} | Est. RAM: {v.estimated_memory_mb}MB")
            sys.exit(EXIT_SUCCESS)

        elif args.model_action == "install":
            print(f"[*] Installing model: {args.model_id}...")
            try:
                def on_prog(fname, d, t):
                    pct = (d / t * 100.0) if t > 0 else 0.0
                    print(f"\r  Downloading {fname}: {d/(1024*1024):.1f}/{t/(1024*1024):.1f}MB ({pct:.1f}%)", end="", flush=True)
                
                cache.install(args.model_id, progress_callback=on_prog)
                print(f"\n[+] Successfully installed '{args.model_id}'.")
                sys.exit(EXIT_SUCCESS)
            except RuntimeRequiredError:
                raise
            except Exception as e:
                print(f"\n[-] Model installation failed: {e}", file=sys.stderr)
                sys.exit(EXIT_MODEL_VERIFY_FAIL)

        elif args.model_action == "download":
            print(f"[*] Downloading custom model: {args.source}...")
            try:
                def on_prog(fname, d, t):
                    pct = (d / t * 100.0) if t > 0 else 0.0
                    print(f"\r  Downloading {fname}: {d/(1024*1024):.1f}/{t/(1024*1024):.1f}MB ({pct:.1f}%)", end="", flush=True)

                saved_path = download_custom_url_or_hf(args.source, dest_dir=args.output, progress_callback=on_prog)
                print(f"\n[+] Model successfully saved to: {saved_path}")
                sys.exit(EXIT_SUCCESS)
            except RuntimeRequiredError:
                raise
            except Exception as e:
                print(f"\n[-] Model download failed: {e}", file=sys.stderr)
                sys.exit(EXIT_MODEL_VERIFY_FAIL)

        elif args.model_action == "remove":
            if cache.remove(args.model_id):
                print(f"[+] Model '{args.model_id}' removed from cache.")
                sys.exit(EXIT_SUCCESS)
            else:
                print(f"[-] Model '{args.model_id}' was not found in cache.", file=sys.stderr)
                sys.exit(EXIT_MODEL_NOT_FOUND)

    elif args.command == "info":
        info = get_image_info(args.image_path)
        print("=== Image Metadata ===")
        for k, v in info.items():
            print(f"  {k}: {v}")
        sys.exit(EXIT_SUCCESS)

    elif args.command == "canny":
        img = load_image(args.image_path)
        if args.resize:
            w, h = map(int, args.resize.split("x"))
            img = resize(img, (w, h))

        gray = to_grayscale(img)
        t0 = time.perf_counter()
        edges = canny(gray, low_threshold=args.low, high_threshold=args.high, device=args.device)
        lat = (time.perf_counter() - t0) * 1000.0
        save_image(edges, args.output, metadata="strip")
        print(f"[+] Canny edges computed on [{args.device.upper()}] in {lat:.2f}ms. Saved to {args.output}")
        sys.exit(EXIT_SUCCESS)

    elif args.command == "detect-face":
        img = load_image(args.image_path)
        detections = detect_faces(
            img,
            score_threshold=args.score_threshold,
            backend=args.backend
        )
        if args.json:
            res_dict = [
                {"bbox": d.bbox.to_xywh(), "score": d.score} for d in detections
            ]
            print(json.dumps({"count": len(detections), "detections": res_dict}, indent=2))
        else:
            print(f"[+] Detected {len(detections)} face regions.")
            if detections:
                # Sort by score if available, otherwise by bounding box area
                best_det = max(detections, key=lambda d: (d.score if d.score is not None else 0.0, d.bbox.area))
                score_str = f" (Confidence: {best_det.score * 100:.1f}%)" if best_det.score is not None else ""
                print(f"    - Primary Face Box: {best_det.bbox.to_xywh()}{score_str}")
                face_crop = crop(img, best_det.bbox, copy=False)
                save_image(face_crop, args.output, metadata="strip")
                print(f"[+] Saved primary face crop to {args.output}")
        sys.exit(EXIT_SUCCESS)

    elif args.command == "vlm":
        from ..vlm.api import load
        cache = ModelCacheManager()

        if args.model and args.model_path:
            parser.error("Conflicting arguments: Cannot specify both '-m/--model' and '--model-path'. Use '--model' for catalog presets or '--model-path' for custom GGUF files.")

        try:
            load_kwargs = {}
            if args.model_path:
                expanded = os.path.abspath(os.path.expanduser(args.model_path))
                if not os.path.isfile(expanded):
                    parser.error(f"Specified --model-path does not exist: {args.model_path}")
                load_kwargs["model_path"] = expanded
            else:
                target_model = resolve_model_id(cache, args.model, interactive=not args.allow_download)
                load_kwargs["model_id"] = target_model

            with load(
                **load_kwargs,
                device=args.device,
                memory_policy=args.memory_policy,
                threads=args.threads,
                runtime_path=args.runtime,
                mmproj_path=args.mmproj,
                allow_download=args.allow_download,
                context_limit=args.ctx_size,
                ngl=args.ngl,
                rpc=args.rpc,
                tensor_split=args.tensor_split,
                cluster_rpc_servers=getattr(args, "cluster_rpc_servers", None),
                cluster_split_mode=getattr(args, "cluster_split_mode", None),
                cluster_tensor_split=getattr(args, "cluster_tensor_split", None),
            ) as engine:
                res = engine.describe(
                    args.image_path,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repeat_penalty=args.repeat_penalty,
                    presence_penalty=args.presence_penalty,
                    frequency_penalty=args.frequency_penalty,
                    seed=args.seed,
                    system_prompt=args.system_prompt,
                    quality=args.quality,
                    max_dim=args.max_dim
                )
                if args.json:
                    print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
                else:
                    if res.warnings:
                        for w in res.warnings:
                            print(f"[WARNING] {w}", file=sys.stderr)
                    backend = res.metrics.backend if res.metrics else "unknown"
                    tps_str = f" | {res.metrics.tokens_per_second:.1f} t/s" if res.metrics and res.metrics.tokens_per_second else ""
                    print(f"\n[VLM Result | backend={backend}{tps_str}]")
                    print(res.text)
            sys.exit(EXIT_SUCCESS)

        except (ClusterConnectionError, ClusterConfigurationError) as exc:
            print(f"[ERROR] [AMEVA-VISION-CLUSTER] {exc}", file=sys.stderr)
            sys.exit(EXIT_CLUSTER_ERROR)

        except VulkanNotAvailableError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(EXIT_GPU_ERROR)

        except RuntimeNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            _print_runtime_install_help()
            _print_available_models(cache)
            sys.exit(EXIT_RUNTIME_NOT_FOUND)

        except NoInstalledModelsError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(EXIT_NO_INSTALLED_MODELS)

        except ModelSelectionRequiredError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(EXIT_MODEL_SELECTION_REQUIRED)

        except ModelArtifactsMissingError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(EXIT_MODEL_INCOMPLETE)

        except ModelNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(EXIT_MODEL_NOT_FOUND)

        except ModelDownloadError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(EXIT_MODEL_VERIFY_FAIL)

        except ValueError as exc:
            print(f"[ERROR] Invalid parameter: {exc}", file=sys.stderr)
            sys.exit(EXIT_ARG_ERROR)

        except RuntimeRequiredError:
            raise
        except Exception as exc:
            print(f"[ERROR] VLM execution failed: {exc}", file=sys.stderr)
            sys.exit(EXIT_INFERENCE_FAIL)

    elif args.command == "benchmark":
        print("=== termux-vision On-Device Benchmark ===")
        import numpy as np
        dummy = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        t0 = time.perf_counter()
        res = resize(dummy, (256, 256))
        t_res = (time.perf_counter() - t0) * 1000.0
        print(f"  - Resize (512x512 -> 256x256): {t_res:.2f} ms")

        gray = to_grayscale(res)
        # Warmup
        canny(gray, 40, 120, device=args.device)
        canny_times = []
        runs = max(1, getattr(args, "runs", 3))
        for _ in range(runs):
            t1 = time.perf_counter()
            edges = canny(gray, 40, 120, device=args.device)
            canny_times.append((time.perf_counter() - t1) * 1000.0)
        t_can_mean = float(np.mean(canny_times))
        t_can_min = float(np.min(canny_times))
        print(f"  - Grayscale & Canny Edge (256x256, {args.device.upper()}, {runs} runs): Mean {t_can_mean:.2f} ms (Min: {t_can_min:.2f} ms)")

        detector = HaarCascadeDetector()
        t2 = time.perf_counter()
        boxes = detector.detect_multiscale(res, scale_factor=1.2, min_size=(32, 32))
        t_haar = (time.perf_counter() - t2) * 1000.0
        print(f"  - Heuristic Cascade Scan: {t_haar:.2f} ms")

        if args.image_path:
            print(f"\n[*] Running VLM benchmark on: {args.image_path}")
            from ..vlm.api import load
            with load(model_id=args.model, device=args.device, memory_policy=args.memory_policy) as engine:
                t_vlm_0 = time.perf_counter()
                vlm_res = engine.describe(args.image_path, max_tokens=64)
                lat_vlm = (time.perf_counter() - t_vlm_0) * 1000.0
                tps_str = f"{vlm_res.metrics.tokens_per_second:.1f} t/s" if vlm_res.metrics and vlm_res.metrics.tokens_per_second else "N/A"
                print(f"  - VLM Generation Latency: {lat_vlm:.1f} ms ({tps_str})")
        print("[+] Benchmark Complete.")
        sys.exit(EXIT_SUCCESS)

    elif args.command in ("component", "model", "instance"):
        try:
            from ameva_component.cli_support import dispatch_protocol
            from termux_vision.control import VisionControl
            dispatch_protocol(args, VisionControl())
        except ImportError:
            print("[ERROR] ameva-component-sdk not installed.", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(EXIT_ARG_ERROR)

if __name__ == "__main__":
    main()
