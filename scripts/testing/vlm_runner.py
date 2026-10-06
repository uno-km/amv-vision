#!/usr/bin/env python3
"""
AMEVA Edge VLM Runner
=====================
Universal on-device Multimodal Vision-Language Model execution engine.
Follows the official uno-km family CLI standard:
  - Computing device acceleration: -d, --device, -b, --backend (auto, gpu, cpu, vulkan)
  - Full parameterization: resolution scaling, token limits, quantization selection, Mali tuning.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="AMEVA Edge VLM Engine - Universal Multimodal Inference Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Family Standard: Compute Device / Backend
    parser.add_argument(
        "-d", "--device", "-b", "--backend",
        dest="device",
        default="auto",
        choices=["auto", "gpu", "vulkan", "opencl", "cpu"],
        help="Compute acceleration device backend"
    )

    # Input & Prompt Parameters
    parser.add_argument("-i", "--image", required=True, help="Path to input image")
    parser.add_argument("-p", "--prompt", default="Describe what you see in this image in detail.", help="Text prompt query")

    # Model & Projector Quantization
    parser.add_argument("-m", "--model", default="models/SmolVLM-Instruct-Q4_K_M.gguf", help="Path to LLM GGUF model")
    parser.add_argument("--mmproj", default="models/mmproj-SmolVLM-Instruct-Q8_0.gguf", help="Path to vision projector GGUF model")
    parser.add_argument("--chat-template", default="auto", help="Chat template (auto, smolvlm, chatml, etc.)")

    # Generation & Context Controls
    parser.add_argument("-n", "--max-tokens", "--n-predict", dest="max_tokens", type=int, default=60, help="Maximum generated tokens")
    parser.add_argument("-c", "--ctx-size", "--ctx", dest="ctx_size", type=int, default=2048, help="Context size")
    parser.add_argument("-t", "--threads", type=int, default=6, help="CPU execution threads")
    parser.add_argument("--ngl", type=int, default=None, help="Explicit number of GPU layers to offload (auto-assigned if omitted)")

    # Vision Scaling & Dynamic Token Bounds (Unrestricted)
    parser.add_argument("-W", "--width", type=int, default=None, help="Resize image width (pixels)")
    parser.add_argument("-H", "--height", type=int, default=None, help="Resize image height (pixels)")
    parser.add_argument("--image-size", default=None, help="Resize image to WxH (e.g. 224x224, 384x384)")
    parser.add_argument("--image-min-tokens", type=int, default=None, help="Minimum image tokens for dynamic resolution ViT")
    parser.add_argument("--image-max-tokens", type=int, default=None, help="Maximum image tokens for dynamic resolution ViT")

    # Mali & Vulkan Tuning
    parser.add_argument("--tune-mali", action="store_true", help="Enable Mali GPU-specific MMVQ tuning (GGML_VK_FORCE_MMVQ=1)")

    # Output & Telemetry
    parser.add_argument("--json", action="store_true", help="Output full benchmark metrics in JSON format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print raw verbose logs")

    return parser.parse_args()

def prepare_image(image_path, width=None, height=None, image_size=None):
    """Resize image locally if dimensions are specified."""
    actual_path = Path(image_path).resolve()
    if not actual_path.exists():
        raise FileNotFoundError(f"[ERROR] Image file not found: {image_path}")

    target_w, target_h = None, None
    if image_size:
        parts = image_size.lower().split("x")
        if len(parts) == 2:
            target_w, target_h = int(parts[0]), int(parts[1])
    elif width or height:
        target_w = width
        target_h = height

    if target_w and target_h:
        try:
            from PIL import Image
            img = Image.open(actual_path).convert("RGB")
            img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            target_resized = actual_path.parent / f"{actual_path.stem}_{target_w}x{target_h}{actual_path.suffix}"
            img_resized.save(target_resized)
            return str(target_resized)
        except Exception as e:
            sys.stderr.write(f"[Warning] Image resize failed ({e}), using original: {actual_path}\n")

    return str(actual_path)

def execute_vlm(args):
    """Execute on-device VLM inference via native llama-mtmd-cli."""
    dev = args.device.lower().strip()
    env = os.environ.copy()
    if args.tune_mali:
        env["GGML_VK_FORCE_MMVQ"] = "1"

    prepared_image = prepare_image(args.image, args.width, args.height, args.image_size)

    chat_template = args.chat_template
    if chat_template == "auto":
        m_lower = args.model.lower()
        if "qwen2" in m_lower:
            chat_template = "chatml"
        elif "smolvlm" in m_lower:
            chat_template = "smolvlm"
        else:
            chat_template = "smolvlm"

    base_cmd = [
        "llama-mtmd-cli",
        "-m", args.model,
        "--mmproj", args.mmproj,
        "--image", prepared_image,
        "-p", f"'{args.prompt}'",
        "--chat-template", chat_template,
        "-c", str(args.ctx_size),
        "-n", str(args.max_tokens),
        "-t", str(args.threads),
        "--no-warmup",
        "-v",
    ]

    # Device backend routing
    if dev in ("gpu", "vulkan"):
        ngl = args.ngl if args.ngl is not None else 99
        base_cmd.extend([
            "-ngl", str(ngl),
            "-ot", "token_embd.weight=Vulkan0",
            "-fit", "off"
        ])
    elif dev == "cpu":
        base_cmd.extend([
            "-ngl", "0",
            "--no-mmproj-offload"
        ])
    elif dev == "auto":
        ngl = args.ngl if args.ngl is not None else 99
        base_cmd.extend([
            "-ngl", str(ngl),
            "-fit", "on"
        ])

    if args.image_max_tokens:
        base_cmd.extend(["--image-max-tokens", str(args.image_max_tokens)])
    if args.image_min_tokens:
        base_cmd.extend(["--image-min-tokens", str(args.image_min_tokens)])

    full_cli = " ".join(base_cmd)

    start_time = time.time()
    proc = subprocess.run(full_cli, shell=True, capture_output=True, text=True, env=env)
    elapsed_wall = time.time() - start_time
    output = proc.stdout + proc.stderr

    metrics = {
        "backend": dev,
        "image": str(prepared_image),
        "model": args.model,
        "mmproj": args.mmproj,
        "exit_code": proc.returncode,
        "wall_time_s": round(elapsed_wall, 2),
        "cpu_mapped_mib": 0.00 if dev in ("gpu", "vulkan") else None,
        "vulkan_model_mib": None,
        "vulkan_compute_mib": None,
        "kv_buffer_mib": None,
        "vision_slice_ms": None,
        "vision_decode_ms": None,
        "prompt_eval_ms": None,
        "prompt_eval_tok_s": None,
        "eval_ms": None,
        "eval_tok_s": None,
        "response_text": "",
    }

    cpu_match = re.search(r"CPU_Mapped model buffer size\s*=\s*([\d\.]+)\s*MiB", output)
    if cpu_match:
        metrics["cpu_mapped_mib"] = float(cpu_match.group(1))

    vk_model_match = re.search(r"Vulkan0 model buffer size\s*=\s*([\d\.]+)\s*MiB", output)
    if vk_model_match:
        metrics["vulkan_model_mib"] = float(vk_model_match.group(1))

    vk_comp_match = re.search(r"Vulkan0 compute buffer size\s*=\s*([\d\.]+)\s*MiB", output)
    if vk_comp_match:
        metrics["vulkan_compute_mib"] = float(vk_comp_match.group(1))

    kv_match = re.search(r"Vulkan0 KV buffer size\s*=\s*([\d\.]+)\s*MiB", output)
    if kv_match:
        metrics["kv_buffer_mib"] = float(kv_match.group(1))

    slice_match = re.search(r"image slice encoded in\s*(\d+)\s*ms", output)
    if slice_match:
        metrics["vision_slice_ms"] = int(slice_match.group(1))

    decode_match = re.search(r"image decoded.*?in\s*(\d+)\s*ms", output)
    if decode_match:
        metrics["vision_decode_ms"] = int(decode_match.group(1))

    p_eval_match = re.search(r"prompt eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)", output)
    if p_eval_match:
        metrics["prompt_eval_ms"] = float(p_eval_match.group(1))
        metrics["prompt_eval_tok_s"] = float(p_eval_match.group(3))

    eval_match = re.search(r"eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)", output)
    if eval_match:
        metrics["eval_ms"] = float(eval_match.group(1))
        metrics["eval_tok_s"] = float(eval_match.group(3))

    lines = output.splitlines()
    clean_lines = []
    for line in lines:
        if re.match(r"^[\d\.]+\s+[IDWE]\s+", line):
            continue
        if re.match(r"^\.{5,}", line):
            continue
        if "llama_" in line or "ggml_" in line or "main:" in line or "WARN:" in line:
            continue
        if "mtmd_cli_context:" in line or "<|im_start|>" in line or "User:" in line or "Assistant:" in line:
            continue
        if "--- vision hparams ---" in line or "clip_ctx:" in line:
            continue
        stripped = line.strip()
        if stripped:
            clean_lines.append(stripped)

    metrics["response_text"] = " ".join(clean_lines)
    return metrics, output

def main():
    args = parse_args()
    metrics, raw_output = execute_vlm(args)

    if args.json:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    print("\n" + "=" * 65)
    print("           AMEVA EDGE VLM EXECUTION SCORECARD")
    print("=" * 65)
    print(f" Compute Backend       : {metrics['backend']}")
    print(f" Image Target          : {metrics['image']}")
    print(f" Model GGUF            : {metrics['model']}")
    print(f" Vision Projector      : {metrics['mmproj']}")
    print("-" * 65)
    cpu_val = f"{metrics['cpu_mapped_mib']:.2f} MiB" if metrics['cpu_mapped_mib'] is not None else "N/A"
    vk_val = f"{metrics['vulkan_model_mib']} MiB" if metrics['vulkan_model_mib'] is not None else "N/A"
    comp_val = f"{metrics['vulkan_compute_mib']} MiB" if metrics['vulkan_compute_mib'] is not None else "N/A"
    kv_val = f"{metrics['kv_buffer_mib']} MiB" if metrics['kv_buffer_mib'] is not None else "N/A"
    slice_val = f"{metrics['vision_slice_ms']} ms ({round(metrics['vision_slice_ms']/1000.0, 2)} s)" if metrics['vision_slice_ms'] else "N/A"
    decode_val = f"{metrics['vision_decode_ms']} ms" if metrics['vision_decode_ms'] else "N/A"
    peval_val = f"{metrics['prompt_eval_ms']} ms ({metrics['prompt_eval_tok_s']} tok/s)" if metrics['prompt_eval_ms'] else "N/A"
    eval_val = f"{metrics['eval_tok_s']} tok/s ({metrics['eval_ms']} ms)" if metrics['eval_tok_s'] else "N/A"

    print(f" CPU Mapped VRAM       : {cpu_val}")
    print(f" Vulkan Model VRAM     : {vk_val}")
    print(f" Vulkan Compute Buffer : {comp_val}")
    print(f" Vulkan KV Buffer      : {kv_val}")
    print("-" * 65)
    print(f" Vision Slice Encode   : {slice_val}")
    print(f" Vision Batch Decode   : {decode_val}")
    print(f" Prompt Eval Time      : {peval_val}")
    print(f" Text Eval Rate        : {eval_val}")
    print(f" Total Wall Clock Time : {metrics['wall_time_s']} s")
    print("-" * 65)
    print(f" Model Response:\n\"{metrics['response_text']}\"")
    print("=" * 65 + "\n")

    if args.verbose:
        print("\n--- RAW LOG OUTPUT ---")
        print(raw_output)

if __name__ == "__main__":
    main()
