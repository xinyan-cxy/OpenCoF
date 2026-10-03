# Copyright (c) 2025 Ziyu Guo (MME-CoF authors)
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# This file integrates / is adapted from the MME-CoF benchmark (MIT License) (https://video-cof.github.io/).
# Modifications by ByteDance Ltd. and/or its affiliates.
import os
import json
import time
import math
import cv2
import torch
import argparse
import shutil
import uuid
import sys
import contextlib
import numpy as np
import torch.multiprocessing as mp
from PIL import Image, ImageOps
from pathlib import Path
from colorama import Fore, Style, init
from tqdm import tqdm

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def parse_args():
    parser = argparse.ArgumentParser(description="MME-CoF Full Benchmark Inference Pipeline")

    parser.add_argument("--data_file", type=str,
                        default="/path/to/data/MME-CoF/data.json",
                        help="Path to MME-CoF data.json (full 120 items)")
    parser.add_argument("--data_root", type=str,
                        default="/path/to/data/MME-CoF",
                        help="Root directory for image paths in data file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated videos")
    parser.add_argument("--model_path", type=str,
                        default="/path/to/models/diffsynth-studio/Wan2.2-I2V-A14B",
                        help="Path to the base Wan model")
    parser.add_argument("--lora_path", type=str, default="",
                        help="Path to the LoRA .safetensors file")
    parser.add_argument("--state_dict_path", type=str, default="",
                        help="Path to full model state dict (optional)")
    parser.add_argument("--high_noise_lora_path", type=str, default="",
                        help="Path to high noise LoRA .safetensors file")

    parser.add_argument("--no_skip", action="store_true",
                        help="Force regenerate existing videos")
    parser.add_argument("--tiled", action="store_true", default=True,
                        help="Enable VAE tiled decoding")
    parser.add_argument("--no_tiled", action="store_false", dest="tiled")

    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--denoising_strength", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=5)
    parser.add_argument("--switch_dit_boundary", type=float, default=0.875,
                        help="Normalized timestep at which the pipeline switches from "
                        "the high-noise expert (dit) to the low-noise expert (dit2). "
                        "Pipeline default is 0.875; override (e.g. 0.9) to align with "
                        "the training MoE split when training used a different boundary.")
    parser.add_argument("--num_cond_reasoning_tokens", type=int, default=0)
    parser.add_argument(
        "--num_reasoning_tokens",
        type=int,
        default=-1,
        help="Visual-side reasoning tokens for DiT. -1 means auto-infer from LoRA checkpoint.",
    )

    parser.add_argument("--shard_id", type=int, default=0,
                        help="Current node shard ID (default 0 for single-node)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards/nodes (default 1 for single-node)")

    return parser.parse_args()


def _inspect_lora_reasoning_tokens(lora_path):
    if not lora_path or not os.path.exists(lora_path):
        return None
    try:
        from safetensors import safe_open
        with safe_open(lora_path, framework="pt", device="cpu") as handle:
            key = next((k for k in handle.keys() if k.endswith("reasoning_tokens")), None)
            if key is None:
                return None
            tensor = handle.get_tensor(key)
            if tensor.ndim < 2:
                return None
            return int(tensor.shape[1])
    except Exception:
        return None


def _resolve_num_reasoning_tokens(args, lora_paths):
    if args.num_reasoning_tokens >= 0:
        return int(args.num_reasoning_tokens), "cli"
    found = []
    for path in lora_paths:
        value = _inspect_lora_reasoning_tokens(path)
        if value is not None:
            found.append((path, value))
    if not found:
        return None, "default"
    unique = sorted({value for _, value in found})
    if len(unique) > 1:
        return int(found[0][1]), "mixed_lora"
    return int(unique[0]), "lora"


def _apply_num_reasoning_tokens(pipe, num_reasoning_tokens):
    if num_reasoning_tokens is None:
        return
    for name in ("dit", "dit2"):
        module = getattr(pipe, name, None)
        if module is None or not hasattr(module, "reasoning_tokens"):
            continue
        if module.reasoning_tokens.shape[1] == num_reasoning_tokens:
            module.num_reasoning_tokens = num_reasoning_tokens
            continue
        dim = getattr(module, "dim", module.reasoning_tokens.shape[-1])
        with torch.no_grad():
            rt = torch.randn(
                1, num_reasoning_tokens, dim,
                device=pipe.device, dtype=pipe.torch_dtype,
            ) / (dim ** 0.5)
        module.num_reasoning_tokens = num_reasoning_tokens
        module.reasoning_tokens = torch.nn.Parameter(rt, requires_grad=True)


def _validate_dual_noise_lora_paths(args):
    if "Wan2.2-I2V-A14B" not in args.model_path:
        return
    if not args.high_noise_lora_path:
        return

    high_noise_path = args.high_noise_lora_path
    low_noise_path = high_noise_path.replace("high_noise_lora", "low_noise_lora")

    if not os.path.exists(high_noise_path):
        raise FileNotFoundError(f"high_noise LoRA not found: {high_noise_path}")
    if not os.path.exists(low_noise_path):
        raise FileNotFoundError(f"low_noise LoRA not found: {low_noise_path}")


def add_padding_832_480(image_path, color=(255, 255, 255)):
    target_w, target_h = 832, 480
    try:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        final_img = Image.new("RGB", (target_w, target_h), color)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        final_img.paste(img_resized, (paste_x, paste_y))
        return final_img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def load_wan_model(device_id, args):
    device = f"cuda:{device_id}"
    tqdm.write(f"{Fore.CYAN}[GPU {device_id}] Loading Wan Model from {args.model_path}...{Style.RESET_ALL}")

    if "Wan2.2-I2V-A14B" in args.model_path:
        low_noise_path = args.high_noise_lora_path.replace("high_noise_lora", "low_noise_lora") if args.high_noise_lora_path else ""
        model_configs = [
            ModelConfig(path=[
                f"{args.model_path}/high_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
                for i in range(1, 7)
            ]),
            ModelConfig(path=[
                f"{args.model_path}/low_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
                for i in range(1, 7)
            ]),
            ModelConfig(path=f"{args.model_path}/Wan2.1_VAE.pth"),
            ModelConfig(path=f"{args.model_path}/models_t5_umt5-xxl-enc-bf16.pth"),
        ]
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=ModelConfig(path=f"{args.model_path}/google/umt5-xxl"),
            num_cond_reasoning_tokens=args.num_cond_reasoning_tokens,
        )
        num_reasoning_tokens, reason_source = _resolve_num_reasoning_tokens(
            args, [args.high_noise_lora_path, low_noise_path, args.lora_path]
        )
        _apply_num_reasoning_tokens(pipe, num_reasoning_tokens)
        if num_reasoning_tokens is not None:
            tqdm.write(
                f"{Fore.CYAN}[GPU {device_id}] Set num_reasoning_tokens={num_reasoning_tokens} "
                f"(source={reason_source}).{Style.RESET_ALL}"
            )
        if args.high_noise_lora_path and os.path.exists(args.high_noise_lora_path):
            tqdm.write(f"{Fore.CYAN}[GPU {device_id}] Loading high_noise LoRA...{Style.RESET_ALL}")
            pipe.load_lora(pipe.dit, args.high_noise_lora_path, alpha=args.lora_alpha)
            if os.path.exists(low_noise_path):
                tqdm.write(f"{Fore.CYAN}[GPU {device_id}] Loading low_noise LoRA...{Style.RESET_ALL}")
                pipe.load_lora(pipe.dit2, low_noise_path, alpha=args.lora_alpha)
            else:
                tqdm.write(
                    f"{Fore.YELLOW}[GPU {device_id}] low_noise LoRA not found: {low_noise_path}{Style.RESET_ALL}"
                )

    elif "Wan2.2-TI2V-5B" in args.model_path:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(path=[
                    f"{args.model_path}/diffusion_pytorch_model-0000{i}-of-00003.safetensors"
                    for i in range(1, 4)
                ]),
                ModelConfig(path=f"{args.model_path}/Wan2.2_VAE.pth"),
                ModelConfig(path=f"{args.model_path}/models_t5_umt5-xxl-enc-bf16.pth"),
            ],
            tokenizer_config=ModelConfig(path=f"{args.model_path}/google/umt5-xxl"),
            num_cond_reasoning_tokens=args.num_cond_reasoning_tokens,
        )
        num_reasoning_tokens, reason_source = _resolve_num_reasoning_tokens(args, [args.lora_path])
        _apply_num_reasoning_tokens(pipe, num_reasoning_tokens)
        if num_reasoning_tokens is not None:
            tqdm.write(
                f"{Fore.CYAN}[GPU {device_id}] Set num_reasoning_tokens={num_reasoning_tokens} "
                f"(source={reason_source}).{Style.RESET_ALL}"
            )
        if args.state_dict_path and os.path.exists(args.state_dict_path):
            from diffsynth.core import load_state_dict
            state_dict = load_state_dict(args.state_dict_path)
            pipe.dit.load_state_dict(state_dict)
        elif args.lora_path and os.path.exists(args.lora_path):
            tqdm.write(f"{Fore.CYAN}[GPU {device_id}] Loading LoRA: {os.path.basename(args.lora_path)}{Style.RESET_ALL}")
            pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)

    return pipe


def worker_process(rank, gpu_ids, task_chunk, args):
    gpu_id = gpu_ids[rank]
    skip_existing = not args.no_skip

    if not task_chunk:
        return

    try:
        if skip_existing:
            all_done = all(os.path.exists(t[4]) for t in task_chunk)
            if all_done:
                tqdm.write(f"{Fore.YELLOW}[GPU {gpu_id}] All tasks exist. Skipping.{Style.RESET_ALL}")
                return

        pipe = load_wan_model(gpu_id, args)
        pbar = tqdm(task_chunk, total=len(task_chunk), position=rank,
                    desc=f"GPU {gpu_id}", leave=True, ascii=True)

        for task in pbar:
            idx, image_path, prompt, category, output_path = task
            pbar.set_postfix_str(f"idx={idx} cat={category[:12]}")

            if skip_existing and os.path.exists(output_path):
                continue

            try:
                if not os.path.exists(image_path):
                    tqdm.write(f"{Fore.RED}[GPU {gpu_id}] Missing: {image_path}{Style.RESET_ALL}")
                    continue

                input_image = add_padding_832_480(image_path)
                if input_image is None:
                    continue

                current_seed = args.seed + idx

                with suppress_stderr():
                    video = pipe(
                        prompt=prompt,
                        input_image=input_image,
                        num_frames=args.num_frames,
                        seed=current_seed,
                        tiled=args.tiled,
                        denoising_strength=args.denoising_strength,
                        cfg_scale=args.cfg_scale,
                        switch_DiT_boundary=args.switch_dit_boundary,
                    )

                os.makedirs("./tmp", exist_ok=True)
                temp_filename = f"./tmp/wan_temp_{gpu_id}_{uuid.uuid4().hex}.mp4"

                try:
                    with suppress_stderr():
                        save_video(video, temp_filename, fps=args.fps, quality=args.quality)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.move(temp_filename, output_path)
                except Exception as save_err:
                    tqdm.write(f"{Fore.RED}[GPU {gpu_id}] Save Error: {save_err}{Style.RESET_ALL}")
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)

            except Exception as e:
                tqdm.write(f"{Fore.RED}[GPU {gpu_id}] Error idx={idx}: {e}{Style.RESET_ALL}")

    except Exception as e:
        tqdm.write(f"{Fore.RED}[GPU {gpu_id}] Critical Error: {e}{Style.RESET_ALL}")


def main():
    init()
    args = parse_args()
    _validate_dual_noise_lora_paths(args)

    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}MME-CoF Full Benchmark Inference{Style.RESET_ALL}")
    print(f"  Data: {args.data_file}")
    print(f"  Model: {args.model_path}")
    print(f"  LoRA: {args.lora_path or args.high_noise_lora_path or 'None'}")
    print(f"  Output: {args.output_dir}")
    print(f"  Skip existing: {not args.no_skip}")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")

    if not os.path.exists(args.data_file):
        print(f"{Fore.RED}Error: Data file not found: {args.data_file}{Style.RESET_ALL}")
        return

    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from MME-CoF dataset")

    all_tasks = []
    for item in data:
        if "image" not in item:
            continue
        idx = item["idx"]
        image_path = os.path.join(args.data_root, item["image"])
        prompt = item.get("reasoning_prompt", "")
        category = item.get("category", "unknown")
        base_name = os.path.splitext(item["image"])[0]
        output_path = os.path.join(args.output_dir, f"{base_name}.mp4")
        all_tasks.append((idx, image_path, prompt, category, output_path))

    print(f"Total tasks (before sharding): {len(all_tasks)}")

    if args.num_shards > 1:
        all_tasks = [t for i, t in enumerate(all_tasks) if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(all_tasks)} tasks")

    if not all_tasks:
        print("No tasks for this shard.")
        sys.exit(42)

    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
        print(f"Launching on {num_gpus} GPUs")

        chunk_size = math.ceil(len(all_tasks) / num_gpus)
        chunks = [all_tasks[i:i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]
        while len(chunks) < num_gpus:
            chunks.append([])

        processes = []
        for rank in range(num_gpus):
            p = mp.Process(target=worker_process, args=(rank, gpu_ids, chunks[rank], args))
            p.start()
            processes.append(p)
            time.sleep(1)

        for p in processes:
            p.join()
        print(f"\n{Fore.GREEN}=== MME-CoF Inference Complete ==={Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}No CUDA GPUs found.{Style.RESET_ALL}")

    generated = sum(1 for t in all_tasks if os.path.exists(t[4]))
    print(f"Generated: {generated}/{len(all_tasks)} videos")


if __name__ == "__main__":
    main()
