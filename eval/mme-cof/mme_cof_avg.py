# Copyright (c) 2025 Ziyu Guo (MME-CoF authors)
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# This file integrates / is adapted from the MME-CoF benchmark (MIT License) (https://video-cof.github.io/).
# Modifications by ByteDance Ltd. and/or its affiliates.
"""
Average MME-CoF evaluation scores across multiple runs.

Reads eval_results_run{N}.json for each run, averages per-item scores,
recomputes per-category and overall summary, and saves eval_results_avg.json.
"""
import os
import json
import argparse
from collections import defaultdict

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
    parser = argparse.ArgumentParser(
        description="Average MME-CoF eval scores across multiple runs")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--run_ids", type=int, nargs="+", required=True,
                        help="Run IDs to average, e.g. 1 2 3")
    parser.add_argument("--output_json", type=str, default="",
                        help="Output path (default: video_dir/eval_results_avg.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output_json or os.path.join(
        args.video_dir, "eval_results_avg.json")

    # Load all runs
    run_data = {}
    for rid in args.run_ids:
        path = os.path.join(args.video_dir, f"eval_results_run{rid}.json")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping run {rid}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            run_data[rid] = json.load(f)

    if not run_data:
        print("ERROR: No run result files found. Nothing to average.")
        return

    available_runs = sorted(run_data.keys())
    print(f"Averaging runs: {available_runs}")

    # Index per-item scores by idx across runs
    idx_scores = defaultdict(lambda: {dim: [] for dim in EVAL_DIMENSIONS})
    idx_meta = {}
    for rid in available_runs:
        for item in run_data[rid].get("results", []):
            idx = item["idx"]
            if idx not in idx_meta:
                idx_meta[idx] = {
                    "image": item.get("image", ""),
                    "category": item.get("category", ""),
                    "question": item.get("question", ""),
                    "reasoning_prompt": item.get("reasoning_prompt", ""),
                    "video_path": item.get("video_path", ""),
                }
            if item.get("status") != "success" or not item.get("scores"):
                continue
            for dim in EVAL_DIMENSIONS:
                val = item["scores"].get(dim)
                if val is not None:
                    idx_scores[idx][dim].append(val)

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else None

    # Build averaged results list
    averaged_results = []
    for idx in sorted(idx_meta.keys()):
        meta = idx_meta[idx]
        dim_avgs = {}
        has_any = False
        for dim in EVAL_DIMENSIONS:
            v = avg(idx_scores[idx][dim])
            dim_avgs[dim] = v
            if v is not None:
                has_any = True

        per_run_scores = {}
        for rid in available_runs:
            for item in run_data[rid].get("results", []):
                if item["idx"] == idx and item.get("scores"):
                    per_run_scores[f"run{rid}"] = item["scores"]
                    break

        averaged_results.append({
            "idx": idx,
            **meta,
            "status": "success" if has_any else "no_successful_runs",
            "scores": dim_avgs if has_any else None,
            "per_run_scores": per_run_scores,
            "num_runs_averaged": len(idx_scores[idx][EVAL_DIMENSIONS[0]]),
        })

    # Compute summary from averaged per-item scores
    cat_scores = {cat: {dim: [] for dim in EVAL_DIMENSIONS} for cat in CATEGORIES}
    all_scores = {dim: [] for dim in EVAL_DIMENSIONS}

    for r in averaged_results:
        if r["status"] != "success" or not r["scores"]:
            continue
        cat = r["category"]
        for dim in EVAL_DIMENSIONS:
            val = r["scores"].get(dim)
            if val is not None:
                all_scores[dim].append(val)
                if cat in cat_scores:
                    cat_scores[cat][dim].append(val)

    summary = {"per_category": {}, "overall": {}}

    for cat in CATEGORIES:
        cat_summary = {}
        for dim in EVAL_DIMENSIONS:
            cat_summary[dim] = avg(cat_scores[cat][dim]) or 0.0
        dim_vals = [cat_summary[d] for d in EVAL_DIMENSIONS]
        cat_summary["average"] = round(sum(dim_vals) / len(dim_vals), 3)
        cat_summary["count"] = len(cat_scores[cat][EVAL_DIMENSIONS[0]])
        summary["per_category"][cat] = cat_summary

    for dim in EVAL_DIMENSIONS:
        summary["overall"][dim] = avg(all_scores[dim]) or 0.0
    overall_vals = [summary["overall"][d] for d in EVAL_DIMENSIONS]
    summary["overall"]["average"] = round(sum(overall_vals) / len(overall_vals), 3)
    summary["overall"]["total_evaluated"] = len(
        [r for r in averaged_results if r["status"] == "success"])
    summary["overall"]["total_items"] = len(averaged_results)
    summary["overall"]["runs_averaged"] = available_runs

    output = {"summary": summary, "results": averaged_results}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"MME-CoF Averaged Evaluation (runs {available_runs})")
    print(f"  Evaluated: {summary['overall']['total_evaluated']}/{summary['overall']['total_items']}")
    print(f"  Overall Average: {summary['overall']['average']:.3f} / 4.0")
    print(f"\n  Per-dimension:")
    for dim in EVAL_DIMENSIONS:
        print(f"    {dim}: {summary['overall'][dim]:.3f}")
    print(f"\n  Per-category averages:")
    for cat in CATEGORIES:
        cat_data = summary["per_category"].get(cat, {})
        print(f"    {cat}: {cat_data.get('average', 0):.3f} (n={cat_data.get('count', 0)})")
    print(f"\n  Results saved: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
