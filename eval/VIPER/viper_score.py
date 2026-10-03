# Copyright (c) 2026 Yifan Li (VIPER authors)
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# This file integrates / is adapted from the VIPER benchmark (MIT License) (https://arxiv.org/abs/2512.24952).
# Modifications by ByteDance Ltd. and/or its affiliates.
"""
VIPER Benchmark Scoring.
Adapted from VIPER/eval/scripts/score.py.
Calculates pass@k accuracy for decision, process_consistency, outcome_consistency.
Groups by domain and task_type. Writes output JSON (fixed from original where it was commented out).
"""
import json
import os
import logging
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Some inference outputs label the ttt (tic-tac-toe) task as "Structure" instead
# of the canonical paper name "Structural". Normalize so they aggregate together.
DOMAIN_ALIAS = {
    "Structure": "Structural",
}


def normalize_domain(domain):
    return DOMAIN_ALIAS.get(domain, domain)


def extract_task_name(item_id):
    """Extract task name from item_id (format: {task}_{n})."""
    parts = item_id.rsplit('_', 1)
    return parts[0] if len(parts) > 1 else item_id


def calculate_pass_at_k(results, k=None):
    """Calculate pass@k across all tasks for 3 dimensions."""
    task_stats = defaultdict(lambda: {"videos": []})
    all_task_ids = set()

    for r in results:
        task_id = r.get('item_id')
        all_task_ids.add(task_id)
        if r.get('success'):
            task_stats[task_id]["videos"].append({
                "idx": r.get('video_idx', 0),
                "decision": r.get('decision', ''),
                "process_consistency": r.get('process_consistency', ''),
                "outcome_consistency": r.get('outcome_consistency', ''),
            })

    for task_id in task_stats:
        task_stats[task_id]["videos"].sort(key=lambda x: x["idx"])

    use_k = k if k is not None else float('inf')

    decision_correct = 0
    process_correct = 0
    goal_correct = 0

    for task_id in all_task_ids:
        if task_id in task_stats:
            vids = task_stats[task_id]["videos"]
            check = vids[:use_k] if use_k != float('inf') else vids
            if any(v["decision"] == "correct" for v in check):
                decision_correct += 1
            if any(v["process_consistency"] == "correct" for v in check):
                process_correct += 1
            if any(v["outcome_consistency"] == "correct" for v in check):
                goal_correct += 1

    total = len(all_task_ids)
    return {
        "total_tasks": total,
        "decision": {"passed_tasks": decision_correct, "accuracy": round(decision_correct / total, 4) if total else 0},
        "process_consistency": {"passed_tasks": process_correct, "accuracy": round(process_correct / total, 4) if total else 0},
        "outcome_consistency": {"passed_tasks": goal_correct, "accuracy": round(goal_correct / total, 4) if total else 0},
    }


def calculate_task_stats(results, k=None):
    """Calculate pass@k grouped by task_type for 3 dimensions."""
    task_item_stats = defaultdict(lambda: defaultdict(lambda: {"videos": []}))
    all_task_items = defaultdict(set)

    for r in results:
        item_id = r.get('item_id')
        task_name = extract_task_name(item_id)
        all_task_items[task_name].add(item_id)
        if r.get('success'):
            task_item_stats[task_name][item_id]["videos"].append({
                "idx": r.get('video_idx', 0),
                "decision": r.get('decision', ''),
                "process_consistency": r.get('process_consistency', ''),
                "outcome_consistency": r.get('outcome_consistency', ''),
            })

    for task_name in task_item_stats:
        for item_id in task_item_stats[task_name]:
            task_item_stats[task_name][item_id]["videos"].sort(key=lambda x: x["idx"])

    use_k = k if k is not None else float('inf')
    task_results = {}

    for task_name in all_task_items:
        d_correct = p_correct = g_correct = 0
        total_items = len(all_task_items[task_name])

        for item_id in all_task_items[task_name]:
            if item_id in task_item_stats[task_name]:
                vids = task_item_stats[task_name][item_id]["videos"]
                check = vids[:use_k] if use_k != float('inf') else vids
                if any(v["decision"] == "correct" for v in check):
                    d_correct += 1
                if any(v["process_consistency"] == "correct" for v in check):
                    p_correct += 1
                if any(v["outcome_consistency"] == "correct" for v in check):
                    g_correct += 1

        task_results[task_name] = {
            "total": total_items,
            "decision": {"passed": d_correct, "accuracy": round(d_correct / total_items, 4) if total_items else 0},
            "process_consistency": {"passed": p_correct, "accuracy": round(p_correct / total_items, 4) if total_items else 0},
            "outcome_consistency": {"passed": g_correct, "accuracy": round(g_correct / total_items, 4) if total_items else 0},
        }

    return task_results


def process_file(input_file, k=None):
    """Process a single eval result file.

    A file may contain records spanning multiple domains (e.g. the official
    `viper_inference_*.json` covers all 309 items across all 7 domains in one
    file). The original implementation took `results[0]['domain']` for the
    whole file, which incorrectly bucketed every task under whichever domain
    happened to appear first. Here we group records by their own per-record
    `domain` field (after alias normalization) and compute stats per domain.

    Returns:
        Dict[str, Dict] mapping normalized domain -> task_stats dict.
    """
    results = read_json(input_file)
    if not results:
        return {}

    by_domain = defaultdict(list)
    for r in results:
        d = normalize_domain(r.get("domain", "unknown"))
        by_domain[d].append(r)

    out = {}
    for domain, sub_results in by_domain.items():
        pass_at_k = calculate_pass_at_k(sub_results, k)
        task_stats = calculate_task_stats(sub_results, k)
        logger.info(
            f"{domain} - {os.path.basename(input_file)}: "
            f"Decision={pass_at_k['decision']['accuracy']*100:.1f}% "
            f"({pass_at_k['decision']['passed_tasks']}/{pass_at_k['total_tasks']})"
        )
        out[domain] = task_stats
    return out


def main():
    parser = ArgumentParser(description="VIPER pass@k Scoring")
    parser.add_argument("--input_path", type=str, required=True,
                        help="JSON file or directory of eval results")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    input_files = []
    if os.path.isfile(args.input_path):
        input_files = [args.input_path]
        default_output_dir = os.path.dirname(args.input_path)
    elif os.path.isdir(args.input_path):
        fps_pattern = f"fps@{args.fps}"
        pass_pattern = f"pass@{args.k}"
        input_files = [
            os.path.join(args.input_path, f)
            for f in sorted(os.listdir(args.input_path))
            if f.endswith('.json')
            and not f.endswith('_scores.json')
            and fps_pattern in f
            and pass_pattern in f
        ]
        default_output_dir = args.input_path
    else:
        logger.error(f"Invalid input_path: {args.input_path}")
        return

    if not input_files:
        logger.error(f"No matching JSON files found in {args.input_path}")
        return

    output_dir = args.output_path or default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    metric_name = f"fps@{args.fps}_pass@{args.k}"

    logger.info(f"{'='*60}")
    logger.info(f"VIPER Scoring")
    logger.info(f"{'='*60}")
    logger.info(f"Metric: {metric_name}")
    logger.info(f"Files: {len(input_files)}")

    domain_results = defaultdict(dict)

    for input_file in input_files:
        try:
            per_domain = process_file(input_file, args.k)
            for domain, task_stats in per_domain.items():
                for task_name, task_stat in task_stats.items():
                    if task_name in domain_results[domain]:
                        logger.warning(
                            f"Task '{task_name}' duplicated in domain '{domain}', overwriting"
                        )
                    domain_results[domain][task_name] = task_stat
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")

    domain_results = dict(domain_results)

    # Compute per-domain averages (macro across that domain's tasks)
    for domain in domain_results:
        task_stats = domain_results[domain]
        if not task_stats:
            continue
        d_accs = [t["decision"]["accuracy"] for t in task_stats.values()]
        p_accs = [t["process_consistency"]["accuracy"] for t in task_stats.values()]
        g_accs = [t["outcome_consistency"]["accuracy"] for t in task_stats.values()]
        task_stats['average'] = {
            "total": sum(t["total"] for t in task_stats.values()),
            "decision": {
                "passed": sum(t["decision"]["passed"] for t in task_stats.values()),
                "accuracy": round(sum(d_accs) / len(d_accs), 4),
            },
            "process_consistency": {
                "passed": sum(t["process_consistency"]["passed"] for t in task_stats.values()),
                "accuracy": round(sum(p_accs) / len(p_accs), 4),
            },
            "outcome_consistency": {
                "passed": sum(t["outcome_consistency"]["passed"] for t in task_stats.values()),
                "accuracy": round(sum(g_accs) / len(g_accs), 4),
            },
        }

    # Compute overall summary across all tasks (macro across the 16 tasks).
    # This matches the paper's Table 2 "average OC / POC / Hack" columns.
    # Hack = OC - POC (per the paper, the rate of outcome-correct-but-process-wrong).
    all_d_accs, all_p_accs, all_g_accs = [], [], []
    sum_total = sum_d_passed = sum_p_passed = sum_g_passed = 0
    for domain, task_stats in domain_results.items():
        for tn, t in task_stats.items():
            if tn == "average":
                continue
            all_d_accs.append(t["decision"]["accuracy"])
            all_p_accs.append(t["process_consistency"]["accuracy"])
            all_g_accs.append(t["outcome_consistency"]["accuracy"])
            sum_total += t["total"]
            sum_d_passed += t["decision"]["passed"]
            sum_p_passed += t["process_consistency"]["passed"]
            sum_g_passed += t["outcome_consistency"]["passed"]

    if all_d_accs:
        n = len(all_d_accs)
        oc_macro = sum(all_g_accs) / n
        poc_macro = sum(all_d_accs) / n
        domain_results['Overall'] = {
            "num_tasks": n,
            "total_items": sum_total,
            "decision": {
                "passed": sum_d_passed,
                "accuracy": round(poc_macro, 4),
            },
            "process_consistency": {
                "passed": sum_p_passed,
                "accuracy": round(sum(all_p_accs) / n, 4),
            },
            "outcome_consistency": {
                "passed": sum_g_passed,
                "accuracy": round(oc_macro, 4),
            },
            "hack": {
                "accuracy": round(oc_macro - poc_macro, 4),
            },
        }

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Summary by Domain and Task")
    logger.info(f"{'='*80}\n")

    for domain in sorted(d for d in domain_results if d != 'Overall'):
        task_stats = domain_results[domain]
        if 'average' not in task_stats:
            continue
        logger.info(f"{domain}:")
        logger.info(f"  {'Task':<30} {'Process':<15} {'Outcome':<15} {'Decision':<15}")
        logger.info(f"  {'-'*75}")

        for task_name in sorted(k for k in task_stats if k != 'average'):
            s = task_stats[task_name]
            logger.info(
                f"  {task_name:<30} "
                f"{s['process_consistency']['accuracy']*100:.1f}%{'':>8} "
                f"{s['outcome_consistency']['accuracy']*100:.1f}%{'':>8} "
                f"{s['decision']['accuracy']*100:.1f}%"
            )

        avg = task_stats['average']
        logger.info(f"  {'-'*75}")
        logger.info(
            f"  {'average':<30} "
            f"{avg['process_consistency']['accuracy']*100:.1f}%{'':>8} "
            f"{avg['outcome_consistency']['accuracy']*100:.1f}%{'':>8} "
            f"{avg['decision']['accuracy']*100:.1f}%"
        )
        logger.info("")

    if 'Overall' in domain_results:
        o = domain_results['Overall']
        logger.info(f"Overall (macro across {o['num_tasks']} tasks, {o['total_items']} items):")
        logger.info(
            f"  PC={o['process_consistency']['accuracy']*100:.1f}%  "
            f"OC={o['outcome_consistency']['accuracy']*100:.1f}%  "
            f"POC={o['decision']['accuracy']*100:.1f}%  "
            f"Hack={o['hack']['accuracy']*100:.1f}%"
        )
        logger.info("")

    # Save JSON output (fixed from original score.py where this was commented out)
    output_file = os.path.join(output_dir, f"{metric_name}_scores.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(domain_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
