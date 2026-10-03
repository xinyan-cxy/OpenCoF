# Copyright (c) 2025 Xuming He (RULER-Bench authors)
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# This file integrates / is adapted from the RULER-Bench benchmark (MIT License).
# Modifications by ByteDance Ltd. and/or its affiliates.
import json
import os
import argparse
import shutil
import tempfile
from tqdm import tqdm
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

SCORE_MAP = {"Good": 2, "Medium": 1, "Poor": 0}

CATEGORY = {
    "science": ["chemistry", "physics", "biology", "earth", "math", "medicine", "life"],
    "game": ["chess", "puzzle", "gomoku", "sudoku", "maze", "minesweeper", "number_sliding_puzzle", "sticks", "xiangqi", "Go"],
    "semantic": ["idioms", "metaphors", "definition"],
    "hypothetical": ["subjective_change", "objective_change"],
    "humanity": ["transportation", "sport", "social", "safety", "festival", "dress", "food", "emotion"],
    "vision rule": ["anomaly", "color", "count", "direction", "position", "shape", "size", "style", "view", "motion"],
}

DIMENSIONS = ["Instruction Following", "Visual Consistency", "Visual Fidelity", "Rule Coherence"]


def parse_args():
    parser = argparse.ArgumentParser(description="RULER-Bench Accuracy Calculation")
    parser.add_argument("--eval_result_dir", type=str, required=True,
                        help="Directory containing per-sample eval results (e.g., video_dir/eval_results/model_name/)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Directory to save summary (default: eval_result_dir/../summary/)")
    parser.add_argument("--model_name", type=str, default="",
                        help="Model name for output file naming")
    return parser.parse_args()


def cal_score_per_instance(eval_result_dir):
    """Calculate per-instance dimension averages from raw eval results."""
    instance_scores = []
    errors = 0

    for task_name in sorted(os.listdir(eval_result_dir)):
        task_folder = os.path.join(eval_result_dir, task_name)
        if not os.path.isdir(task_folder):
            continue

        for filename in sorted(os.listdir(task_folder)):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(task_folder, filename)
            try:
                with open(filepath, "r", encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                errors += 1
                continue

            eval_results = data.get("eval_results", [])
            checklist = data.get("checklist", [])

            if len(eval_results) != len(checklist):
                errors += 1
                continue

            dim_scores = defaultdict(list)
            for idx in range(len(eval_results)):
                key = list(checklist[idx].keys())[0]
                ans = eval_results[idx]
                if ans in SCORE_MAP:
                    dim_scores[key].append(SCORE_MAP[ans])

            avg_scores = {}
            for dim in DIMENSIONS:
                vals = dim_scores.get(dim, [])
                avg_scores[dim] = sum(vals) / len(vals) if vals else 0.0

            instance_scores.append({
                "task_name": data.get("task_name", task_name),
                "index": data.get("index", filename.replace(".json", "")),
                "category": data.get("category", ""),
                "avg_scores": avg_scores,
            })

    if errors > 0:
        print(f"Warning: {errors} files had parsing errors")

    return instance_scores


def compute_report(instance_scores):
    """Compute per-category and overall scores from instance scores."""
    dimension_answers = {dim: [] for dim in DIMENSIONS}
    category_answers = {cat: [] for cat in CATEGORY}
    category_dimension = {cat: {dim: [] for dim in DIMENSIONS} for cat in CATEGORY}
    all_scores = []

    for inst in instance_scores:
        category = inst["category"]
        avg_scores = inst["avg_scores"]

        for dim in DIMENSIONS:
            if not (dim == "Instruction Following" and category == "vision rule"):
                dimension_answers[dim].append(avg_scores[dim])
                if category in category_dimension:
                    category_dimension[category][dim].append(avg_scores[dim])

        if category != "vision rule":
            avg_val = sum(avg_scores[d] for d in DIMENSIONS) / 4.0
            if category in category_answers:
                category_answers[category].append(avg_val)
            all_scores.append(avg_val)
        else:
            non_if_dims = [d for d in DIMENSIONS if d != "Instruction Following"]
            avg_val = sum(avg_scores[d] for d in non_if_dims) / 3.0
            if category in category_answers:
                category_answers[category].append(avg_val)
            all_scores.append(avg_val)

    def to_100(vals):
        if not vals:
            return 0.0
        return round(sum(vals) / len(vals) / 2.0 * 100.0, 2)

    cat_results = {}
    for cat in CATEGORY:
        cat_results[cat] = {
            "average": to_100(category_answers.get(cat, [])),
            "count": len(category_answers.get(cat, [])),
        }
        for dim in DIMENSIONS:
            vals = category_dimension.get(cat, {}).get(dim, [])
            cat_results[cat][dim] = to_100(vals)

    avg_dims = {}
    for dim in DIMENSIONS:
        cat_vals = []
        for cat in CATEGORY:
            v = category_dimension.get(cat, {}).get(dim, [])
            if v:
                cat_vals.append(to_100(v))
        avg_dims[dim] = round(sum(cat_vals) / len(cat_vals), 2) if cat_vals else 0.0

    overall_avg = round(sum(avg_dims.values()) / len(avg_dims), 2)

    return {
        "per_category": cat_results,
        "overall_per_dimension": avg_dims,
        "overall_average": overall_avg,
        "total_instances": len(instance_scores),
    }


def save_excel(report, output_path):
    """Save report as Excel file."""
    if not PANDAS_AVAILABLE:
        print("pandas not available, skipping Excel output")
        return

    rows = []
    for cat, dims in report["per_category"].items():
        cat_display = cat.title()
        if cat != "vision rule":
            rows.append({"Category": cat_display, "Dimension": "IF", "Score": dims.get("Instruction Following", 0)})
            rows.append({"Category": "", "Dimension": "VC", "Score": dims.get("Visual Consistency", 0)})
            rows.append({"Category": "", "Dimension": "VF", "Score": dims.get("Visual Fidelity", 0)})
            rows.append({"Category": "", "Dimension": "RC", "Score": dims.get("Rule Coherence", 0)})
            rows.append({"Category": "", "Dimension": "Avg", "Score": dims.get("average", 0)})
        else:
            rows.append({"Category": cat_display, "Dimension": "VC", "Score": dims.get("Visual Consistency", 0)})
            rows.append({"Category": "", "Dimension": "VF", "Score": dims.get("Visual Fidelity", 0)})
            rows.append({"Category": "", "Dimension": "RC", "Score": dims.get("Rule Coherence", 0)})
            rows.append({"Category": "", "Dimension": "Avg", "Score": dims.get("average", 0)})

    overall = report["overall_per_dimension"]
    rows.append({"Category": "Overall", "Dimension": "IF", "Score": overall.get("Instruction Following", 0)})
    rows.append({"Category": "", "Dimension": "VC", "Score": overall.get("Visual Consistency", 0)})
    rows.append({"Category": "", "Dimension": "VF", "Score": overall.get("Visual Fidelity", 0)})
    rows.append({"Category": "", "Dimension": "RC", "Score": overall.get("Rule Coherence", 0)})
    rows.append({"Category": "", "Dimension": "Avg", "Score": report["overall_average"]})

    df = pd.DataFrame(rows, columns=["Category", "Dimension", "Score"])
    # Write to local temp file first: xlsxwriter uses zipfile.seek(), which fails on HDFS (Errno 95)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        tmp_path = f.name
    try:
        df.to_excel(tmp_path, index=False)
        shutil.copy2(tmp_path, output_path)
        print(f"Excel summary saved: {output_path}")
    except OSError as e:
        print(f"Warning: could not write Excel to {output_path} ({e}). Wrote to {tmp_path}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def print_report(report):
    """Print human-readable report to stdout."""
    print(f"\n{'='*60}")
    print(f"RULER-Bench Evaluation Summary")
    print(f"Total instances: {report['total_instances']}")
    print(f"{'='*60}")

    print(f"\n{'Category':<20} {'IF':>6} {'VC':>6} {'VF':>6} {'RC':>6} {'Avg':>6} {'N':>4}")
    print("-" * 60)

    for cat in CATEGORY:
        d = report["per_category"].get(cat, {})
        cat_display = cat[:18]
        if cat == "vision rule":
            print(f"{cat_display:<20} {'--':>6} {d.get('Visual Consistency',0):>6.1f} "
                  f"{d.get('Visual Fidelity',0):>6.1f} {d.get('Rule Coherence',0):>6.1f} "
                  f"{d.get('average',0):>6.1f} {d.get('count',0):>4}")
        else:
            print(f"{cat_display:<20} {d.get('Instruction Following',0):>6.1f} "
                  f"{d.get('Visual Consistency',0):>6.1f} {d.get('Visual Fidelity',0):>6.1f} "
                  f"{d.get('Rule Coherence',0):>6.1f} {d.get('average',0):>6.1f} {d.get('count',0):>4}")

    print("-" * 60)
    o = report["overall_per_dimension"]
    print(f"{'Overall':<20} {o.get('Instruction Following',0):>6.1f} "
          f"{o.get('Visual Consistency',0):>6.1f} {o.get('Visual Fidelity',0):>6.1f} "
          f"{o.get('Rule Coherence',0):>6.1f} {report['overall_average']:>6.1f}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(args.eval_result_dir.rstrip("/")), "summary")

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_name or os.path.basename(args.eval_result_dir.rstrip("/"))

    print(f"Computing scores from: {args.eval_result_dir}")
    instance_scores = cal_score_per_instance(args.eval_result_dir)
    print(f"Loaded {len(instance_scores)} instances")

    if not instance_scores:
        print("No valid instances found. Check eval_result_dir.")
        return

    report = compute_report(instance_scores)
    print_report(report)

    json_path = os.path.join(args.output_dir, f"{model_name}_report.json")
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"JSON report saved: {json_path}")

    if PANDAS_AVAILABLE:
        xlsx_path = os.path.join(args.output_dir, f"{model_name}_evaluation_summary.xlsx")
        save_excel(report, xlsx_path)


if __name__ == "__main__":
    main()
