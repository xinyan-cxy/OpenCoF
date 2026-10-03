# Copyright (c) 2025 Xuming He (RULER-Bench authors)
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# This file integrates / is adapted from the RULER-Bench benchmark (MIT License).
# Modifications by ByteDance Ltd. and/or its affiliates.
"""
RULER-Bench evaluation. The user content, extract_frames and
chat.completions.create(presence_penalty=0) follow the official eval/eval.py
(qwen_vl_processor). Only the I2V subset is evaluated: a sample needs an
input_image_path that exists on disk (matching the Wan2.2-I2V inference subset).

For o1/o3/o4 reasoning judges, temperature and presence_penalty are dropped and
max_tokens is sent as max_completion_tokens, since these models reject them.
"""
import os
import json
import re
import base64
import argparse
import cv2
import concurrent.futures
import time
from tqdm import tqdm
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="RULER-Bench eval (official-format prompts; I2V-only)")
    parser.add_argument("--data_file", type=str,
                        default="/path/to/cof/RULER-Bench/data.jsonl")
    parser.add_argument("--data_root", type=str,
                        default="/path/to/cof/RULER-Bench")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--system_prompt_file", type=str,
                        default="/path/to/cof/RULER-Bench/eval/system_prompt.txt")

    parser.add_argument("--api_base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--eval_model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="0 = omit (as in the official eval.py); >0 is sent to the API "
                             "(as max_completion_tokens for reasoning models)")

    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--no_skip", action="store_true")
    return parser.parse_args()


def is_reasoning_model(model_name: str) -> bool:
    """o1 / o3 / o4-mini reject temperature / presence_penalty and use
    max_completion_tokens instead of max_tokens."""
    name = (model_name or "").lower().strip()
    return name.startswith("o1") or name.startswith("o3") or name.startswith("o4")


def text_format(text):
    return {"type": "text", "text": text}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def img_format(image_path):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}


def resize_keep_ratio(image, max_side=1024):
    """Same as the official eval.py."""
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image
    scale = max_side / long_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_image_from_array(image_array):
    success, buffer = cv2.imencode(".jpg", image_array)
    if not success:
        raise ValueError("Frame encoding failed.")
    return base64.b64encode(buffer).decode("utf-8")


def img_format_from_array(image_array):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encode_image_from_array(image_array)}"},
    }


def extract_frames(video_path, fps=2, max_side=512):
    """Same as extract_frames in the official eval.py (fps=2, max_side=512)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    frame_list = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame = resize_keep_ratio(frame, max_side=max_side)
            frame_list.append(img_format_from_array(frame))
        frame_idx += 1
    cap.release()
    return frame_list


def format_check(response, correct_len):
    try:
        response = response.strip()
        response = re.sub(r"^```[a-zA-Z]*\n?", "", response)
        response = re.sub(r"```$", "", response)
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not match:
            return False
        answer_text = match.group(1).strip()
        sample = json.loads(answer_text)
        if isinstance(sample, list) and len(sample) == correct_len:
            for item in sample:
                if item not in ["Good", "Medium", "Poor"]:
                    return False
            return True
        return False
    except Exception:
        return False


def build_user_content_official(sample, generated_video_path, data_root):
    """
    Same content order and wording as qwen_vl_processor in RULER-Bench/eval/eval.py.
    data_root plays the role of the official `prefix` (f\"{prefix}{input_image_path}\").
    """
    content = []
    checklist = sample["checklist"]
    prompt = sample["prompt"]
    input_image_path = sample.get("input_image_path", "")
    input_video_path = sample.get("input_video_path", "")
    gt_image_path = sample.get("gt_image_path", "")
    implicit_explanation = sample.get("implicit_explanation", "")

    if input_image_path:
        input_image_full_path = os.path.join(data_root, input_image_path)
        content.append(text_format("\n## Input Image:\n"))
        content.append(img_format(input_image_full_path))

    prompt_text = f"\n## Input Prompt:\n{prompt}\n"
    content.append(text_format(prompt_text))

    if gt_image_path:
        gt_image_full_path = os.path.join(data_root, gt_image_path)
        content.append(text_format("\n## Ground truth Image:\n"))
        content.append(img_format(gt_image_full_path))

    if implicit_explanation:
        implicit_explanation_text = f"\n## Implicit Explanation:\n{implicit_explanation}\n"
        content.append(text_format(implicit_explanation_text))

    if input_video_path:
        input_video_full_path = os.path.join(data_root, input_video_path)
        content.append(text_format("\n## Input Video:\n"))
        content.extend(extract_frames(input_video_full_path, fps=2))

    content.append(text_format("\n## Video to evaluate:\n"))
    content.extend(extract_frames(generated_video_path, fps=2))

    content.append(text_format("\n## CheckList Questions:\n"))
    for idx, check_item in enumerate(checklist):
        key = list(check_item.keys())[0]
        question = check_item[key]
        question_text = f"\n# Question {idx+1}:\n{question}\n"
        content.append(text_format(question_text))

    return content


def evaluate_sample(sample, args, client, system_prompt):
    task_name = sample["task_name"]
    index = sample["index"]
    save_path = os.path.join(args.save_dir, args.model_name, task_name, f"{index}.json")

    if not args.no_skip and os.path.exists(save_path):
        return save_path

    iip = (sample.get("input_image_path") or "").strip()
    if not iip or not os.path.exists(os.path.join(args.data_root, iip)):
        return None

    video_path = os.path.join(args.video_dir, task_name, f"{index}.mp4")
    if not os.path.exists(video_path):
        tqdm.write(f"Video not found: {video_path}")
        return None

    try:
        content = build_user_content_official(sample, video_path, args.data_root)
    except Exception as e:
        tqdm.write(f"  [{task_name}/{index}] build content failed: {e}")
        return None

    create_kwargs = {
        "model": args.eval_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    }
    if is_reasoning_model(args.eval_model):
        # o1/o3/o4 reasoning models reject temperature & presence_penalty;
        # max_tokens is replaced by max_completion_tokens. The official
        # RULER-Bench eval.py also calls o3 without max_tokens -- we omit it
        # by default (max_tokens=0). Override --max_tokens >0 to cap.
        if args.max_tokens and args.max_tokens > 0:
            create_kwargs["max_completion_tokens"] = args.max_tokens
    else:
        create_kwargs["presence_penalty"] = 0
        create_kwargs["temperature"] = 0
        if args.max_tokens and args.max_tokens > 0:
            create_kwargs["max_tokens"] = args.max_tokens

    checklist = sample["checklist"]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**create_kwargs)
            response_text = response.choices[0].message.content

            if not format_check(response_text, len(checklist)):
                if attempt < max_retries - 1:
                    tqdm.write(f"  [{task_name}/{index}] Format check failed, retry {attempt+1}")
                    continue
                return None

            response_text = response_text.strip()
            response_text = re.sub(r"^```[a-zA-Z]*\n?", "", response_text)
            response_text = re.sub(r"```$", "", response_text)
            think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
            answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
            think_text = think_match.group(1).strip() if think_match else ""
            answer_text = answer_match.group(1).strip() if answer_match else "[]"
            eval_results = json.loads(answer_text)

            save_data = sample.copy()
            save_data["reasoning"] = think_text
            save_data["eval_results"] = eval_results
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=4)
            return save_path
        except Exception as e:
            if attempt < max_retries - 1:
                tqdm.write(f"  [{task_name}/{index}] Error: {e}, retry {attempt+1}")
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"  [{task_name}/{index}] Failed: {e}")
                return None
    return None


def main():
    args = parse_args()
    if not args.save_dir:
        args.save_dir = os.path.join(args.video_dir, "eval_results")

    api_key = args.api_key or os.environ.get("API_KEY", "")
    api_base_url = args.api_base_url or os.environ.get("API_BASE_URL", "")
    if not api_base_url or not api_key:
        print("Error: --api_base_url and --api_key (or env API_BASE_URL, API_KEY).")
        return

    print("=" * 60)
    print("RULER-Bench Eval | I2V-only | user content = official eval.py")
    print(f"  data_root:     {args.data_root}")
    print(f"  video_dir:     {args.video_dir}")
    print(f"  model_name:    {args.model_name}")
    print(f"  eval_model:    {args.eval_model}  "
          f"(reasoning={is_reasoning_model(args.eval_model)})")
    print(f"  max_tokens:    {args.max_tokens or '(omit, like official)'}")
    print(f"  save_dir:      {args.save_dir}")
    print("=" * 60)

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    with open(args.system_prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    samples = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample_data = json.loads(line)
            iip = (sample_data.get("input_image_path") or "").strip()
            if not iip:
                continue
            if not os.path.exists(os.path.join(args.data_root, iip)):
                continue

            task_name = sample_data["task_name"]
            index = sample_data["index"]
            save_path = os.path.join(args.save_dir, args.model_name, task_name, f"{index}.json")
            if not args.no_skip and os.path.exists(save_path):
                continue
            video_path = os.path.join(args.video_dir, task_name, f"{index}.mp4")
            if not os.path.exists(video_path):
                continue
            samples.append(sample_data)

    print(f"I2V samples to evaluate (has input image + generated mp4): {len(samples)}")
    if not samples:
        print("No samples.")
        return

    def eval_wrapper(s):
        return evaluate_sample(s, args, client, system_prompt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(tqdm(executor.map(eval_wrapper, samples), total=len(samples), desc="Evaluating"))

    success = sum(1 for r in results if r is not None)
    print(f"\nDone: {success}/{len(samples)} ok -> {args.save_dir}/{args.model_name}/")


if __name__ == "__main__":
    main()
