# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Export OpenCoF-17k from the Hugging Face Hub into the layout used for training.

    <output_dir>/
        metadata.csv          # columns: video, prompt
        <task>/<name>.mp4

Usage:
    python scripts/prepare_data.py --output_dir /path/to/OpenCoF-17k
"""
import argparse
import csv
import glob
import os

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo_id", default="xy06/OpenCoF-17k")
    parser.add_argument("--parquet_dir", default=None,
                        help="Use already-downloaded parquet files instead of fetching from the Hub.")
    args = parser.parse_args()

    parquet_dir = args.parquet_dir or snapshot_download(
        args.repo_id, repo_type="dataset", allow_patterns=["data/*.parquet"])
    files = sorted(glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files under {parquet_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    with tqdm(total=total, desc="export") as bar:
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["video", "prompt", "video_path"]):
                for row in batch.to_pylist():
                    rel = row["video_path"]
                    dst = os.path.join(args.output_dir, rel)
                    if not os.path.exists(dst):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with open(dst, "wb") as fp:
                            fp.write(row["video"]["bytes"])
                    rows.append((rel, row["prompt"]))
                    bar.update(1)

    with open(os.path.join(args.output_dir, "metadata.csv"), "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["video", "prompt"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {os.path.join(args.output_dir, 'metadata.csv')}")


if __name__ == "__main__":
    main()
