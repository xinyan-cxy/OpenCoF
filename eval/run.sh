#!/bin/bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# MME-CoF / VIPER / RULER-Bench evaluation pipeline
# Usage:
#   bash run.sh [--stage STAGE] [--model MODEL] [--high_noise_lora LORA] --name NAME
#
# Stages:
#   infer_mme     - MME-CoF inference
#   eval_mme      - MME-CoF judging (averaged over --mme_eval_runs runs)
#   infer_viper   - VIPER inference
#   eval_viper    - VIPER judging + scoring
#   infer_ruler   - RULER-Bench inference
#   eval_ruler    - RULER-Bench judging
#   acc_ruler     - RULER-Bench accuracy
#   infer_all     - inference for all benchmarks
#   eval_all      - judging for all benchmarks
#   all           - everything (default)
#
# Example:
#   # Wan-CoF (the low-noise LoRA is found by replacing high_noise_lora -> low_noise_lora)
#   bash run.sh --stage all \
#     --model /path/to/Wan2.2-I2V-A14B \
#     --high_noise_lora /path/to/Wan-CoF/high_noise_lora/model.safetensors \
#     --name wan_cof \
#     --output_root /path/to/eval_output
#
#   # Baseline (no LoRA)
#   bash run.sh --stage all --model /path/to/Wan2.2-I2V-A14B --name wan2.2_i2v_a14b
#
#   # Judging only (after inference is done)
#   bash run.sh --stage eval_all --name wan_cof --output_root /path/to/eval_output
# =============================================================================

set -e

# ===================== Default Configuration =====================

STAGE="all"
MODEL_NAME_TAG=""
MODEL_PATH="/path/to/Wan2.2-I2V-A14B"
LORA_PATH=""
HIGH_NOISE_LORA_PATH=""
STATE_DICT_PATH=""
LORA_ALPHA=1.0
OUTPUT_ROOT="/path/to/eval_output"

MME_COF_DATA="/path/to/MME-CoF/data.json"
MME_COF_DATA_ROOT="/path/to/MME-CoF"

# VIPER_DATA_ROOT holds the HF dataset.parquet; VIPER_REPO is the VIPER code
# repository, which provides the judge prompts under eval/prompt/.
VIPER_DATA_ROOT="/path/to/VIPER-dataset"
VIPER_REPO="/path/to/VIPER"

# RULER_DATA_ROOT is the HF dataset (data.jsonl + images); the judge system
# prompt comes from the RULER-Bench code repository.
RULER_DATA="/path/to/RULER-Bench-data/data.jsonl"
RULER_DATA_ROOT="/path/to/RULER-Bench-data"
RULER_SYSTEM_PROMPT="/path/to/RULER-Bench/eval/system_prompt.txt"

OPENAI_API_BASE="${OPENAI_API_BASE:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
EVAL_MODEL="gemini-2.5-pro"
MME_EVAL_RUNS=1

NUM_FRAMES=81
SEED=1

RULER_TASK_FILTER=""
RULER_CATEGORY_FILTER=""

# ===================== Parse Arguments =====================

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --lora) LORA_PATH="$2"; shift 2 ;;
        --high_noise_lora) HIGH_NOISE_LORA_PATH="$2"; shift 2 ;;
        --state_dict) STATE_DICT_PATH="$2"; shift 2 ;;
        --lora_alpha) LORA_ALPHA="$2"; shift 2 ;;
        --name) MODEL_NAME_TAG="$2"; shift 2 ;;
        --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
        --mme_data) MME_COF_DATA="$2"; shift 2 ;;
        --mme_data_root) MME_COF_DATA_ROOT="$2"; shift 2 ;;
        --viper_data_root) VIPER_DATA_ROOT="$2"; shift 2 ;;
        --viper_repo) VIPER_REPO="$2"; shift 2 ;;
        --ruler_data) RULER_DATA="$2"; shift 2 ;;
        --ruler_data_root) RULER_DATA_ROOT="$2"; shift 2 ;;
        --ruler_system_prompt) RULER_SYSTEM_PROMPT="$2"; shift 2 ;;
        --openai_api_base) OPENAI_API_BASE="$2"; shift 2 ;;
        --openai_api_key) OPENAI_API_KEY="$2"; shift 2 ;;
        --eval_model) EVAL_MODEL="$2"; shift 2 ;;
        --mme_eval_runs) MME_EVAL_RUNS="$2"; shift 2 ;;
        --num_frames) NUM_FRAMES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --task_filter) RULER_TASK_FILTER="$2"; shift 2 ;;
        --category_filter) RULER_CATEGORY_FILTER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_NAME_TAG" ]; then
    echo "Error: --name is required (e.g., 'wan_cof')"
    exit 1
fi

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"

MME_OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME_TAG}/MME-CoF"
VIPER_OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME_TAG}/VIPER"
RULER_OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME_TAG}/RULER-Bench"

# ===================== Build Common Args =====================

COMMON_MODEL_ARGS="--model_path ${MODEL_PATH}"

if [ -n "$LORA_PATH" ]; then
    COMMON_MODEL_ARGS="${COMMON_MODEL_ARGS} --lora_path ${LORA_PATH} --lora_alpha ${LORA_ALPHA}"
fi

if [ -n "$HIGH_NOISE_LORA_PATH" ]; then
    COMMON_MODEL_ARGS="${COMMON_MODEL_ARGS} --high_noise_lora_path ${HIGH_NOISE_LORA_PATH} --lora_alpha ${LORA_ALPHA}"
fi

if [ -n "$STATE_DICT_PATH" ]; then
    COMMON_MODEL_ARGS="${COMMON_MODEL_ARGS} --state_dict_path ${STATE_DICT_PATH}"
fi

require_api() {
    if [ -z "$OPENAI_API_BASE" ] || [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_BASE and OPENAI_API_KEY must be set for $1 judging."
        echo "Set them in the environment or via --openai_api_base URL --openai_api_key KEY"
        return 1
    fi
}

banner() {
    echo "============================================================"
    echo "[Stage] $1"
    echo "  $2"
    echo "============================================================"
}

# ===================== Stage Functions =====================

infer_mme() {
    banner "MME-CoF Inference" "Output: ${MME_OUTPUT_DIR}"
    python "${EVAL_DIR}/mme-cof/mme_cof_infer.py" \
        --data_file "${MME_COF_DATA}" \
        --data_root "${MME_COF_DATA_ROOT}" \
        --output_dir "${MME_OUTPUT_DIR}" \
        --num_frames ${NUM_FRAMES} \
        --seed ${SEED} \
        ${COMMON_MODEL_ARGS}
    echo "[Done] MME-CoF Inference"
}

eval_mme() {
    banner "MME-CoF Evaluation" "Videos: ${MME_OUTPUT_DIR}  (runs: ${MME_EVAL_RUNS})"
    require_api "MME-CoF" || return 1

    if [ "${MME_EVAL_RUNS}" -le 1 ]; then
        python "${EVAL_DIR}/mme-cof/mme_cof_eval.py" \
            --data_file "${MME_COF_DATA}" \
            --video_dir "${MME_OUTPUT_DIR}" \
            --api_base_url "${OPENAI_API_BASE}" \
            --api_key "${OPENAI_API_KEY}" \
            --eval_model "${EVAL_MODEL}" \
            --output_json "${MME_OUTPUT_DIR}/eval_results.json"
    else
        for run_id in $(seq 1 "${MME_EVAL_RUNS}"); do
            python "${EVAL_DIR}/mme-cof/mme_cof_eval.py" \
                --data_file "${MME_COF_DATA}" \
                --video_dir "${MME_OUTPUT_DIR}" \
                --api_base_url "${OPENAI_API_BASE}" \
                --api_key "${OPENAI_API_KEY}" \
                --eval_model "${EVAL_MODEL}" \
                --run_id "${run_id}"
        done
        python "${EVAL_DIR}/mme-cof/mme_cof_avg.py" \
            --video_dir "${MME_OUTPUT_DIR}" \
            --run_ids $(seq 1 "${MME_EVAL_RUNS}")
    fi
    echo "[Done] MME-CoF Evaluation"
}

infer_viper() {
    banner "VIPER Inference" "Output: ${VIPER_OUTPUT_DIR}"
    if [ ! -f "${VIPER_DATA_ROOT}/data/viper.json" ]; then
        python "${EVAL_DIR}/VIPER/materialize_viper_data.py" --data_root "${VIPER_DATA_ROOT}"
    fi
    python "${EVAL_DIR}/VIPER/viper_infer.py" \
        --data_root "${VIPER_DATA_ROOT}" \
        --output_dir "${VIPER_OUTPUT_DIR}" \
        --num_frames ${NUM_FRAMES} \
        --seed ${SEED} \
        ${COMMON_MODEL_ARGS}
    echo "[Done] VIPER Inference"
}

eval_viper() {
    banner "VIPER Evaluation" "Videos: ${VIPER_OUTPUT_DIR}"
    require_api "VIPER" || return 1

    local EVAL_OUT="${VIPER_OUTPUT_DIR}/evaluation"
    python "${EVAL_DIR}/VIPER/viper_eval.py" \
        --data_file "${VIPER_OUTPUT_DIR}/viper_inference.json" \
        --output_path "${EVAL_OUT}" \
        --system_prompt_path "${VIPER_REPO}/eval/prompt/system_prompt.txt" \
        --domain_prompt_root "${VIPER_REPO}/eval/prompt/domain_prompt" \
        --api_base_url "${OPENAI_API_BASE}" \
        --api_key "${OPENAI_API_KEY}" \
        --eval_model "${EVAL_MODEL}" \
        --fps 1.0 --k 1 \
        --resume
    python "${EVAL_DIR}/VIPER/viper_score.py" \
        --input_path "${EVAL_OUT}" \
        --output_path "${EVAL_OUT}" \
        --fps 1.0 --k 1
    echo "[Done] VIPER Evaluation"
}

infer_ruler() {
    banner "RULER-Bench Inference" "Output: ${RULER_OUTPUT_DIR}"

    EXTRA_ARGS=""
    if [ -n "$RULER_TASK_FILTER" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --task_filter ${RULER_TASK_FILTER}"
    fi
    if [ -n "$RULER_CATEGORY_FILTER" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --category_filter ${RULER_CATEGORY_FILTER}"
    fi

    python "${EVAL_DIR}/ruler_bench/ruler_bench_infer.py" \
        --data_file "${RULER_DATA}" \
        --data_root "${RULER_DATA_ROOT}" \
        --output_dir "${RULER_OUTPUT_DIR}" \
        --num_frames ${NUM_FRAMES} \
        --seed ${SEED} \
        ${COMMON_MODEL_ARGS} \
        ${EXTRA_ARGS}
    echo "[Done] RULER-Bench Inference"
}

eval_ruler() {
    banner "RULER-Bench Evaluation" "Videos: ${RULER_OUTPUT_DIR}"
    require_api "RULER-Bench" || return 1

    python "${EVAL_DIR}/ruler_bench/ruler_bench_eval.py" \
        --data_file "${RULER_DATA}" \
        --data_root "${RULER_DATA_ROOT}" \
        --video_dir "${RULER_OUTPUT_DIR}" \
        --model_name "${MODEL_NAME_TAG}" \
        --api_base_url "${OPENAI_API_BASE}" \
        --api_key "${OPENAI_API_KEY}" \
        --eval_model "${EVAL_MODEL}" \
        --system_prompt_file "${RULER_SYSTEM_PROMPT}"
    echo "[Done] RULER-Bench Evaluation"
}

acc_ruler() {
    banner "RULER-Bench Accuracy Calculation" "Results: ${RULER_OUTPUT_DIR}/eval_results/${MODEL_NAME_TAG}"

    EVAL_RESULT_DIR="${RULER_OUTPUT_DIR}/eval_results/${MODEL_NAME_TAG}"
    if [ ! -d "$EVAL_RESULT_DIR" ]; then
        echo "Error: Eval results not found at ${EVAL_RESULT_DIR}"
        echo "Run eval_ruler stage first."
        return 1
    fi

    python "${EVAL_DIR}/ruler_bench/ruler_bench_cal_acc.py" \
        --eval_result_dir "${EVAL_RESULT_DIR}" \
        --output_dir "${RULER_OUTPUT_DIR}/summary" \
        --model_name "${MODEL_NAME_TAG}"
    echo "[Done] RULER-Bench Accuracy Calculation"
}

# ===================== Execute Stages =====================

echo "============================================================"
echo "Benchmark Evaluation Pipeline"
echo "  Stage: ${STAGE}"
echo "  Model: ${MODEL_PATH}"
echo "  Name:  ${MODEL_NAME_TAG}"
echo "  LoRA:  ${LORA_PATH:-${HIGH_NOISE_LORA_PATH:-None}}"
echo "  Judge: ${EVAL_MODEL}"
echo "============================================================"

case $STAGE in
    infer_mme)   infer_mme ;;
    eval_mme)    eval_mme ;;
    infer_viper) infer_viper ;;
    eval_viper)  eval_viper ;;
    infer_ruler) infer_ruler ;;
    eval_ruler)  eval_ruler ;;
    acc_ruler)   acc_ruler ;;
    infer_all)
        infer_mme
        infer_viper
        infer_ruler
        ;;
    eval_all)
        eval_mme
        eval_viper
        eval_ruler
        acc_ruler
        ;;
    all)
        infer_mme
        infer_viper
        infer_ruler
        eval_mme
        eval_viper
        eval_ruler
        acc_ruler
        ;;
    *)
        echo "Unknown stage: ${STAGE}"
        echo "Valid stages: infer_mme, eval_mme, infer_viper, eval_viper, infer_ruler, eval_ruler, acc_ruler, infer_all, eval_all, all"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "  MME-CoF:     ${MME_OUTPUT_DIR}"
echo "  VIPER:       ${VIPER_OUTPUT_DIR}"
echo "  RULER-Bench: ${RULER_OUTPUT_DIR}"
echo "============================================================"
