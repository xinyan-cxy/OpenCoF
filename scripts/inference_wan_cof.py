# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Image-to-video inference with Wan-CoF (or any LoRA pair trained by scripts/train/).

    python scripts/inference_wan_cof.py \
        --model_dir /path/to/Wan2.2-I2V-A14B \
        --image input.png --prompt "..." --output output.mp4

By default the adapters are fetched from the Hugging Face Hub (xy06/Wan-CoF).
Pass --high_noise_lora / --low_noise_lora to use local checkpoints instead.
The number of visual / textual reasoning tokens is read from the checkpoint.
"""
import argparse

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video

WIDTH, HEIGHT = 832, 480


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Local Wan2.2-I2V-A14B directory.")
    parser.add_argument("--lora_repo", default="xy06/Wan-CoF")
    parser.add_argument("--high_noise_lora", default=None)
    parser.add_argument("--low_noise_lora", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--switch_dit_boundary", type=float, default=0.875)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def num_tokens(lora_path, key):
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        return f.get_slice(key).get_shape()[1] if key in f.keys() else 0


def resize_and_pad(path):
    img = Image.open(path).convert("RGB")
    scale = min(WIDTH / img.width, HEIGHT / img.height)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    canvas.paste(img, ((WIDTH - img.width) // 2, (HEIGHT - img.height) // 2))
    return canvas


def set_num_reasoning_tokens(pipe, n):
    # load_lora copies `reasoning_tokens` in place, so the shape must match the checkpoint first.
    for module in (pipe.dit, pipe.dit2):
        if hasattr(module, "reasoning_tokens") and module.reasoning_tokens.shape[1] != n:
            dim = module.reasoning_tokens.shape[-1]
            module.reasoning_tokens = torch.nn.Parameter(
                torch.zeros(1, n, dim, device=pipe.device, dtype=pipe.torch_dtype))
        module.num_reasoning_tokens = n


def main():
    args = parse_args()
    high = args.high_noise_lora or hf_hub_download(args.lora_repo, "high_noise_lora/model.safetensors")
    low = args.low_noise_lora or hf_hub_download(args.lora_repo, "low_noise_lora/model.safetensors")

    d = args.model_dir
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=[f"{d}/high_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)]),
            ModelConfig(path=[f"{d}/low_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)]),
            ModelConfig(path=f"{d}/Wan2.1_VAE.pth"),
            ModelConfig(path=f"{d}/models_t5_umt5-xxl-enc-bf16.pth"),
        ],
        tokenizer_config=ModelConfig(path=f"{d}/google/umt5-xxl"),
        num_cond_reasoning_tokens=num_tokens(high, "cond_reasoning_emb"),
    )
    set_num_reasoning_tokens(pipe, num_tokens(high, "reasoning_tokens"))
    pipe.load_lora(pipe.dit, high, alpha=1.0)
    pipe.load_lora(pipe.dit2, low, alpha=1.0)

    video = pipe(
        prompt=args.prompt,
        input_image=resize_and_pad(args.image),
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=True,
        cfg_scale=args.cfg_scale,
        switch_DiT_boundary=args.switch_dit_boundary,
    )
    save_video(video, args.output, fps=args.fps, quality=5)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
