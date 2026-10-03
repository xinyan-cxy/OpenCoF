# Copyright (c) 2025 Ziyu Guo (MME-CoF authors)
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# This file integrates / is adapted from the MME-CoF benchmark (MIT License) (https://video-cof.github.io/).
# Modifications by ByteDance Ltd. and/or its affiliates.
"""
MME-CoF Benchmark Evaluation.
Uses an OpenAI-compatible API (e.g. Gemini 2.5 Pro via base_url + api_key).
"""
import os
import json
import time
import argparse
import cv2
import re
import base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from colorama import Fore, Style, init
from tqdm import tqdm
from openai import OpenAI

# Same as evaluate_video in the official MME-CoF genai_client.py.
TEXT_PROMPT_1 = "Evaluate whether the following video faithfully and coherently visualizes the process described in the text."
TEXT_PROMPT_2 = """
Judge the video only based on visible evidence.
Rate each of the following five aspects on a 0–4 scale (0 = poor, 4 = excellent):

Directly formalize your output answer as a JSON format:

{
  "instruction_alignment": 0,   // How well the video follows the described structure and sequence
  "temporal_consistency": 0,    // Smoothness and continuity between frames
  "visual_stability": 0,        // Stability of appearance, motion, and viewpoint
  "content_fidelity": 0,        // Preservation of key elements without hallucination or loss
  "focus_relevance": 0          // Whether visual attention stays on the intended objects or regions
}

"""

EVAL_DIMENSIONS = [
    "instruction_alignment",
    "temporal_consistency",
    "visual_stability",
    "content_fidelity",
    "focus_relevance",
]

CATEGORIES = [
    "visual detail reasoning", "visual trace reasoning",
    "real world spatial reasoning", "3D geometry reasoning",
    "2D geometry reasoning", "physics based reasoning",
    "rotation reasoning", "table and chart reasoning",
    "object counting reasoning", "gui reasoning",
    "embodied reasoning", "medical reasoning",
]


def parse_args():
    parser = argparse.ArgumentParser(description="MME-CoF Benchmark Evaluation (OpenAI-compatible API, e.g. Gemini 2.5 Pro)")
    parser.add_argument("--data_file", type=str,
                        default="/path/to/cof/MME-CoF/data.json")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing generated videos (output_dir from inference)")
    parser.add_argument("--output_json", type=str, default="",
                        help="Path to save evaluation results JSON (default: video_dir/eval_results.json)")
    parser.add_argument("--api_base_url", type=str, default="",
                        help="OpenAI-compatible API base URL (e.g. Gemini 2.5 Pro endpoint)")
    parser.add_argument("--api_key", type=str, default="",
                        help="API key")
    parser.add_argument("--eval_model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--sample_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel threads for API calls")
    parser.add_argument("--no_skip", action="store_true",
                        help="Force re-evaluate all items")
    parser.add_argument("--run_id", type=int, default=None,
                        help="Run identifier for multi-run eval (e.g. 1, 2, 3). "
                             "Controls per-item cache dir and output filename suffix.")
    return parser.parse_args()


def extract_frames_base64(mp4_path, num_frames=16):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    results = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if success:
            _, buffer = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            results.append(b64)
    cap.release()
    return results


def parse_eval_response(response_text):
    """Parse JSON scores from model response."""
    try:
        json_match = re.search(r'\{[^}]*"instruction_alignment"[^}]*\}', response_text, re.DOTALL)
        if json_match:
            clean = re.sub(r'//.*', '', json_match.group(0))
            return json.loads(clean)
    except Exception:
        pass
    scores = {}
    for dim in EVAL_DIMENSIONS:
        match = re.search(rf'"{dim}"\s*:\s*(\d)', response_text)
        if match:
            scores[dim] = int(match.group(1))
    return scores if len(scores) == 5 else None


def chat_with_frames(client, prompt, frame_b64_list, model_name, max_tokens):
    """OpenAI-compatible chat with text + images."""
    content = []
    for b64 in frame_b64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    kwargs = {"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content


def has_complete_scores(result):
    """Whether cached result contains all required numeric scores."""
    scores = result.get("scores")
    if not isinstance(scores, dict):
        return False
    for dim in EVAL_DIMENSIONS:
        val = scores.get(dim)
        if not isinstance(val, (int, float)):
            return False
    return True


def evaluate_single_item(item, args, client):
    """Evaluate a single MME-CoF item."""
    idx = item["idx"]
    base_name = os.path.splitext(item["image"])[0]
    video_path = os.path.join(args.video_dir, f"{base_name}.mp4")

    suffix = f"_run{args.run_id}" if args.run_id else ""
    result_dir = os.path.join(args.video_dir, f"eval_per_item{suffix}")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{idx}.json")

    if not args.no_skip and os.path.exists(result_path):
        cached_result = None
        try:
            with open(result_path, 'r') as f:
                cached_result = json.load(f)
        except Exception:
            # Corrupted per-item cache: fall through and re-evaluate.
            cached_result = None

        if cached_result is not None:
            cached_status = cached_result.get("status")
            cached_video_path = cached_result.get("video_path", video_path)
            video_exists = os.path.exists(video_path) or os.path.exists(cached_video_path)

            # Resume policy:
            # 1) keep only samples with complete scores;
            # 2) keep missing_video cache only when video is still absent;
            # 3) otherwise re-evaluate this sample to fill missing scores.
            if has_complete_scores(cached_result):
                return cached_result

            if cached_status == "missing_video" and not video_exists:
                return cached_result

    if not os.path.exists(video_path):
        result = {
            "idx": idx, "image": item["image"], "category": item.get("category", ""),
            "question": item.get("question", ""), "reasoning_prompt": item.get("reasoning_prompt", ""),
            "video_path": video_path, "status": "missing_video", "scores": None,
        }
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    frame_b64_list = extract_frames_base64(video_path, args.sample_frames)
    if not frame_b64_list:
        result = {
            "idx": idx, "image": item["image"], "category": item.get("category", ""),
            "question": item.get("question", ""), "reasoning_prompt": item.get("reasoning_prompt", ""),
            "video_path": video_path, "status": "frame_extraction_failed", "scores": None,
        }
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    prompt = item.get("reasoning_prompt", "")
    # Same prompt layout as evaluate_video in the official MME-CoF genai_client.py.
    eval_prompt = (
        f"{TEXT_PROMPT_1}\n\n"
        "Video frames are provided as images.\n\n"
        f"Description is:\n{prompt}\n\n"
        f"{TEXT_PROMPT_2}"
    )

    scores = None
    raw_response = ""
    for attempt in range(3):
        try:
            raw_response = chat_with_frames(
                client, eval_prompt, frame_b64_list,
                args.eval_model, args.max_tokens
            )
            scores = parse_eval_response(raw_response)
            if scores:
                break
        except Exception as e:
            tqdm.write(f"  [idx={idx}] Retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    result = {
        "idx": idx, "image": item["image"], "category": item.get("category", ""),
        "question": item.get("question", ""), "reasoning_prompt": prompt,
        "video_path": video_path,
        "status": "success" if scores else "eval_failed",
        "scores": scores,
        "raw_response": raw_response,
    }
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def compute_summary(results):
    """Compute per-category and overall summary statistics."""
    cat_scores = {cat: {dim: [] for dim in EVAL_DIMENSIONS} for cat in CATEGORIES}
    all_scores = {dim: [] for dim in EVAL_DIMENSIONS}

    for r in results:
        if r["status"] != "success" or not r["scores"]:
            continue
        cat = r["category"]
        for dim in EVAL_DIMENSIONS:
            val = r["scores"].get(dim)
            if val is not None:
                all_scores[dim].append(val)
                if cat in cat_scores:
                    cat_scores[cat][dim].append(val)

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    summary = {"per_category": {}, "overall": {}}

    for cat in CATEGORIES:
        cat_summary = {}
        for dim in EVAL_DIMENSIONS:
            cat_summary[dim] = avg(cat_scores[cat][dim])
        dim_vals = [cat_summary[d] for d in EVAL_DIMENSIONS]
        cat_summary["average"] = round(sum(dim_vals) / len(dim_vals), 3)
        cat_summary["count"] = len(cat_scores[cat][EVAL_DIMENSIONS[0]])
        summary["per_category"][cat] = cat_summary

    for dim in EVAL_DIMENSIONS:
        summary["overall"][dim] = avg(all_scores[dim])
    overall_vals = [summary["overall"][d] for d in EVAL_DIMENSIONS]
    summary["overall"]["average"] = round(sum(overall_vals) / len(overall_vals), 3)
    summary["overall"]["total_evaluated"] = len([r for r in results if r["status"] == "success"])
    summary["overall"]["total_items"] = len(results)

    return summary


def main():
    init()
    args = parse_args()

    if not args.output_json:
        suffix = f"_run{args.run_id}" if args.run_id else ""
        args.output_json = os.path.join(args.video_dir, f"eval_results{suffix}.json")

    api_key = args.api_key or os.environ.get("API_KEY", "")
    api_base_url = args.api_base_url or os.environ.get("API_BASE_URL", "")
    if not api_key or not api_base_url:
        print(f"{Fore.RED}Error: Need --api_base_url and --api_key (or env API_BASE_URL, API_KEY).{Style.RESET_ALL}")
        return

    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    run_label = f" (run {args.run_id})" if args.run_id else ""
    print(f"{Fore.BLUE}MME-CoF Benchmark Evaluation{run_label}{Style.RESET_ALL}")
    print(f"  Data: {args.data_file}")
    print(f"  Videos: {args.video_dir}")
    print(f"  Model: {args.eval_model} @ {api_base_url}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Output: {args.output_json}")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items_to_eval = [item for item in data if "image" in item]
    print(f"Loaded {len(data)} items, {len(items_to_eval)} to evaluate")

    def eval_one(item):
        return evaluate_single_item(item, args, client)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(
            executor.map(eval_one, items_to_eval),
            total=len(items_to_eval),
            desc="Evaluating",
        ))

    summary = compute_summary(results)

    output = {"summary": summary, "results": results}
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Evaluation Complete{Style.RESET_ALL}")
    print(f"  Evaluated: {summary['overall']['total_evaluated']}/{summary['overall']['total_items']}")
    print(f"  Overall Average: {summary['overall']['average']:.3f} / 4.0")
    print(f"\n  Per-dimension:")
    for dim in EVAL_DIMENSIONS:
        print(f"    {dim}: {summary['overall'][dim]:.3f}")
    print(f"\n  Per-category averages:")
    for cat in CATEGORIES:
        cat_data = summary['per_category'].get(cat, {})
        print(f"    {cat}: {cat_data.get('average', 0):.3f} (n={cat_data.get('count', 0)})")
    print(f"\n  Results saved: {args.output_json}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
