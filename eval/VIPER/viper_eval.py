# Copyright (c) 2026 Yifan Li (VIPER authors)
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# This file integrates / is adapted from the VIPER benchmark (MIT License) (https://arxiv.org/abs/2512.24952).
# Modifications by ByteDance Ltd. and/or its affiliates.
"""
VIPER Benchmark VLM Judge.
Adapted from VIPER/eval/scripts/gpt-4o.py but using OpenAI-compatible API
(same pattern as mme_cof_eval.py) instead of OpenRouter.

Evaluates process_consistency, outcome_consistency, and decision.

For o1/o3/o4/gpt-5 reasoning judges, temperature is dropped and max_tokens is
sent as max_completion_tokens, since these models reject the former. By
default (max_tokens=0) no token limit is sent, as in the original gpt-4o.py.
"""
import os
import io
import re
import json
import time
import base64
import logging
import argparse
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

VIPER_DATA_ROOT = "/path/to/data/VIPER"


def parse_args():
    parser = argparse.ArgumentParser(description="VIPER Evaluation (OpenAI-compatible API)")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to viper_inference.json from viper_infer.py")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output directory for eval results")
    parser.add_argument("--system_prompt_path", type=str,
                        default=os.path.join(VIPER_DATA_ROOT, "eval", "prompt", "system_prompt.txt"))
    parser.add_argument("--domain_prompt_root", type=str,
                        default=os.path.join(VIPER_DATA_ROOT, "eval", "prompt", "domain_prompt"))

    parser.add_argument("--api_base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--eval_model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="0 = omit (as in the original gpt-4o.py); >0 is sent to the API "
                             "(as max_completion_tokens for reasoning models)")

    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=None,
                        help="Evaluate top-k videos per item (default: all)")
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_skip", action="store_true")

    return parser.parse_args()


def is_reasoning_model(model_name: str) -> bool:
    """o1 / o3 / o4-mini / gpt-5 reject temperature / presence_penalty and use
    max_completion_tokens instead of max_tokens."""
    name = (model_name or "").lower().strip()
    return name.startswith(("o1", "o3", "o4", "gpt-5"))


# ---------------------------------------------------------------------------
# Media helpers (OpenCV-based, no decord dependency)
# ---------------------------------------------------------------------------

def extract_frames_from_video(video_path, fps=1.0):
    """Sample frames at given fps. Always include first and last."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    duration = total_frames / video_fps
    target_n = max(2, int(round(duration * fps)))

    if target_n >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, target_n).astype(int).tolist()
        indices[-1] = total_frames - 1
        indices = sorted(set(indices))

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


def image_to_data_url(image, fmt="JPEG"):
    if image.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode in ("RGBA", "LA"):
            bg.paste(image, mask=image.split()[-1])
        else:
            bg.paste(image)
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def load_images(paths):
    images = []
    for p in (paths or []):
        if not p or not os.path.exists(p):
            continue
        try:
            images.append(Image.open(p))
        except Exception:
            continue
    return images


# ---------------------------------------------------------------------------
# Response parsing (same as original gpt-4o.py)
# ---------------------------------------------------------------------------

def extract_answer(response_text):
    m = re.search(r"\s*<answer>(.*?)</answer>", response_text, re.DOTALL | re.IGNORECASE)
    if not m:
        return {"valid": False, "decision": "", "process_consistency": "", "outcome_consistency": ""}

    answer = m.group(1).strip()
    answer = re.sub(r"^```json\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"^```\s*", "", answer).strip()
    answer = re.sub(r"\s*```$", "", answer).strip()

    try:
        obj = json.loads(answer)
        decision = str(obj.get("decision", "")).lower()
        pc = str(obj.get("process_consistency", "")).lower()
        oc = str(obj.get("outcome_consistency", "")).lower()

        valid_vals = {"correct", "incorrect"}
        if decision in valid_vals and pc in valid_vals and oc in valid_vals:
            return {"valid": True, "decision": decision,
                    "process_consistency": pc, "outcome_consistency": oc}
    except Exception:
        pass

    low = answer.lower()
    if "incorrect" in low:
        return {"valid": True, "decision": "incorrect", "process_consistency": "", "outcome_consistency": ""}
    if "correct" in low:
        return {"valid": True, "decision": "correct", "process_consistency": "", "outcome_consistency": ""}

    return {"valid": False, "decision": "", "process_consistency": "", "outcome_consistency": ""}


# ---------------------------------------------------------------------------
# Prompt builder (mirrors original gpt-4o.py build_evaluation_content)
# ---------------------------------------------------------------------------

def build_evaluation_content(
    system_prompt, domain_prompt, task_prompt, protocol,
    init_image, video_frames, reference_images=None, reference_text=None,
):
    content = []

    base_text = (
        f"{system_prompt}\n\n{domain_prompt}\n\n"
        f"The task prompt:\n{task_prompt}\n\n"
        f"The initial image:\n"
    )
    content.append({"type": "text", "text": base_text})

    if init_image is not None:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(init_image)}})

    protocol_text = "\n".join(protocol) if isinstance(protocol, list) else str(protocol or "")
    content.append({"type": "text", "text": f"Process constraints:\n{protocol_text}\n"})

    content.append({"type": "text", "text": "The generated video:\n"})
    for fr in video_frames:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(fr)}})

    if reference_images:
        intro = (
            "To assist your decision, here are reference frames sampled from a correct video:\n"
            if len(reference_images) > 1
            else "To assist your decision, here is a correct target frame:\n"
        )
        content.append({"type": "text", "text": intro})
        for img in reference_images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})

    if reference_text:
        content.append({"type": "text", "text": "To assist your decision, here is an example correct text reasoning process:\n"})
        for t in reference_text:
            content.append({"type": "text", "text": t})

    content.append({
        "type": "text",
        "text": (
            "Return your reasoning inside and the final JSON inside "
            "<answer>...</answer>. The JSON must include: decision, process_consistency, outcome_consistency. "
            "Each value must be either 'correct' or 'incorrect'."
        ),
    })

    return content


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def evaluate_single_video(
    client, item, video_url, video_idx,
    system_prompt, domain_prompt, fps, args,
):
    item_id = item.get("id", "unknown")

    try:
        frames = extract_frames_from_video(video_url, fps=fps)
    except Exception as e:
        return {
            "item_id": item_id, "domain": item.get("domain", ""),
            "server_url": video_url, "video_idx": video_idx,
            "decision": "", "process_consistency": "", "outcome_consistency": "",
            "success": False, "error": str(e),
        }

    init_image = None
    img_path = item.get("image")
    if img_path and os.path.exists(img_path):
        try:
            init_image = Image.open(img_path)
        except Exception:
            pass

    reference_images = load_images(item.get("reference_frames", []))
    reference_text = item.get("reference_text", [])
    if isinstance(reference_text, str):
        reference_text = [reference_text] if reference_text else []

    content = build_evaluation_content(
        system_prompt=system_prompt,
        domain_prompt=domain_prompt,
        task_prompt=item.get("prompt", ""),
        protocol=item.get("protocol", ""),
        init_image=init_image,
        video_frames=frames,
        reference_images=reference_images,
        reference_text=reference_text,
    )

    messages = [{"role": "user", "content": content}]
    kwargs = {
        "model": args.eval_model,
        "messages": messages,
    }
    if is_reasoning_model(args.eval_model):
        # o1/o3/o4 / gpt-5 reasoning models reject temperature & presence_penalty;
        # max_tokens is replaced by max_completion_tokens. The original VIPER
        # gpt-4o.py also doesn't pass max_tokens -- so default (=0) just omits it.
        if args.max_tokens and args.max_tokens > 0:
            kwargs["max_completion_tokens"] = args.max_tokens
    else:
        kwargs["temperature"] = 0
        if args.max_tokens and args.max_tokens > 0:
            kwargs["max_tokens"] = args.max_tokens

    for attempt in range(5):
        try:
            completion = client.chat.completions.create(**kwargs)
            raw_text = completion.choices[0].message.content or ""
            parsed = extract_answer(raw_text)
            if parsed["valid"]:
                return {
                    "item_id": item_id, "domain": item.get("domain", ""),
                    "server_url": video_url, "video_idx": video_idx,
                    "raw_answer": raw_text,
                    "decision": parsed["decision"],
                    "process_consistency": parsed["process_consistency"],
                    "outcome_consistency": parsed["outcome_consistency"],
                    "success": True, "error": "",
                }
            logger.warning(f"Invalid response format for {item_id} (attempt {attempt+1})")
            if attempt < 4:
                time.sleep(2)
        except Exception as e:
            logger.error(f"API error for {item_id}: {e} (attempt {attempt+1})")
            if attempt < 4:
                time.sleep(2 ** attempt)

    return {
        "item_id": item_id, "domain": item.get("domain", ""),
        "server_url": video_url, "video_idx": video_idx,
        "raw_answer": raw_text if 'raw_text' in dir() else "",
        "decision": "", "process_consistency": "", "outcome_consistency": "",
        "success": False, "error": "Max retries exceeded",
    }


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("API_KEY", "")
    api_base_url = args.api_base_url or os.environ.get("API_BASE_URL", "")
    if not api_key or not api_base_url:
        logger.error("Need --api_base_url and --api_key (or env API_BASE_URL, API_KEY).")
        return

    system_prompt = read_text(args.system_prompt_path)

    if args.output_path is None:
        args.output_path = os.path.join(os.path.dirname(args.data_file), "evaluation")
    os.makedirs(args.output_path, exist_ok=True)

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    with open(args.data_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if isinstance(dataset, dict):
        dataset = [dataset]

    input_basename = os.path.splitext(os.path.basename(args.data_file))[0]
    model_suffix = args.eval_model.replace("/", "_").replace("\\", "_")
    fps_info = f"fps@{args.fps}"
    pass_info = f"pass@{args.k}" if args.k else "pass@all"
    output_filename = f"{input_basename}_{fps_info}_{pass_info}_{model_suffix}.json"
    output_path = os.path.join(args.output_path, output_filename)

    existing_success = {}
    if args.resume and os.path.exists(output_path):
        try:
            with open(output_path) as f:
                prev = json.load(f)
            for r in prev:
                if r.get("success"):
                    existing_success[(r.get("item_id"), r.get("video_idx"))] = r
            logger.info(f"Resume: loaded {len(existing_success)} successful results")
        except Exception:
            pass

    total_videos = sum(len(item.get("server_url", [])) for item in dataset)
    logger.info(f"Dataset: {len(dataset)} items, {total_videos} videos")
    logger.info(f"Model: {args.eval_model}  "
                f"(reasoning={is_reasoning_model(args.eval_model)})")
    logger.info(f"max_tokens: {args.max_tokens or '(omit, like original gpt-4o.py)'}")
    logger.info(f"Output: {output_path}")

    # Build evaluation tasks
    eval_tasks = []
    for item in dataset:
        item_id = item.get("id", "unknown")
        domain = item.get("domain", "unknown")

        domain_file = os.path.join(args.domain_prompt_root, f"{domain}.txt")
        domain_prompt = read_text(domain_file)

        urls = item.get("server_url", [])
        if args.k and args.k > 0:
            urls = urls[:args.k]

        if not urls:
            key = (item_id, 0)
            if not (args.resume and key in existing_success):
                eval_tasks.append({
                    "item": item, "video_url": "", "video_idx": 0,
                    "system_prompt": system_prompt, "domain_prompt": domain_prompt,
                })
            continue

        for video_idx, url in enumerate(urls):
            key = (item_id, video_idx)
            if args.resume and key in existing_success:
                continue
            eval_tasks.append({
                "item": item, "video_url": url, "video_idx": video_idx,
                "system_prompt": system_prompt, "domain_prompt": domain_prompt,
            })

    logger.info(f"Prepared {len(eval_tasks)} evaluation tasks")

    if args.resume and not eval_tasks:
        logger.info("All evaluations already completed.")
        return

    def do_eval(task):
        if not task["video_url"]:
            return {
                "item_id": task["item"].get("id", "unknown"),
                "domain": task["item"].get("domain", ""),
                "server_url": "", "video_idx": task["video_idx"],
                "decision": "", "process_consistency": "", "outcome_consistency": "",
                "success": False, "error": "No video available",
            }
        return evaluate_single_video(
            client, task["item"], task["video_url"], task["video_idx"],
            task["system_prompt"], task["domain_prompt"],
            args.fps, args,
        )

    new_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(do_eval, t): t for t in eval_tasks}
        with tqdm(total=len(eval_tasks), desc="Evaluating") as pbar:
            for fut in as_completed(futures):
                try:
                    new_results.append(fut.result())
                except Exception as e:
                    t = futures[fut]
                    logger.error(f"Task crashed: {e}")
                    new_results.append({
                        "item_id": t["item"].get("id", "unknown"),
                        "domain": t["item"].get("domain", ""),
                        "server_url": t["video_url"], "video_idx": t["video_idx"],
                        "decision": "", "process_consistency": "", "outcome_consistency": "",
                        "success": False, "error": str(e),
                    })
                pbar.update(1)

    all_results = list(existing_success.values()) + new_results if existing_success else new_results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in all_results if r.get("success"))
    logger.info(f"Completed: {ok}/{len(all_results)} successful")
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
