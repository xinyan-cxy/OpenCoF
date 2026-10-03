#!/usr/bin/env python3
# Copyright (c) 2026 Yifan Li (VIPER authors)
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# This file integrates / is adapted from the VIPER benchmark (MIT License) (https://arxiv.org/abs/2512.24952).
# Modifications by ByteDance Ltd. and/or its affiliates.
"""
Materialize VIPER dataset from local dataset.parquet into viper.json + images.

Reads the HuggingFace parquet (with embedded image bytes) and extracts:
  - {data_root}/data/images/{item_id}_image.jpg
  - {data_root}/data/images/{item_id}_ref_{i:04d}.jpg
  - {data_root}/data/viper.json  (image fields replaced with file paths)

Usage:
    python materialize_viper_data.py [--data_root /path/to/VIPER-dataset]
"""

import argparse
import json
import os

import pyarrow.parquet as pq
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Materialize VIPER dataset from parquet")
    parser.add_argument(
        "--data_root", type=str,
        default="/path/to/data/VIPER-dataset",
        help="Root directory containing dataset.parquet",
    )
    args = parser.parse_args()

    parquet_path = os.path.join(args.data_root, "dataset.parquet")
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found")
        return

    data_dir = os.path.join(args.data_root, "data")
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    table = pq.read_table(parquet_path)
    print(f"Loaded {table.num_rows} rows from {parquet_path}")

    out = []
    for i in tqdm(range(table.num_rows), desc="Materializing"):
        row = {c: table.column(c)[i].as_py() for c in table.column_names}
        item_id = row.get("id", f"{i:08d}")

        img_field = row.get("image")
        if isinstance(img_field, dict) and img_field.get("bytes"):
            img_path = os.path.join(images_dir, f"{item_id}_image.jpg")
            if not os.path.exists(img_path):
                with open(img_path, "wb") as f:
                    f.write(img_field["bytes"])
            row["image"] = img_path
        else:
            row["image"] = ""

        refs = row.get("reference_frames") or []
        new_refs = []
        for j, rf in enumerate(refs):
            if isinstance(rf, dict) and rf.get("bytes"):
                rf_path = os.path.join(images_dir, f"{item_id}_ref_{j:04d}.jpg")
                if not os.path.exists(rf_path):
                    with open(rf_path, "wb") as f:
                        f.write(rf["bytes"])
                new_refs.append(rf_path)
            elif isinstance(rf, str):
                new_refs.append(rf)
        row["reference_frames"] = new_refs

        out.append(row)

    json_path = os.path.join(data_dir, "viper.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(out)} items to {json_path}")
    print(f"Images directory: {images_dir}")


if __name__ == "__main__":
    main()
