#!/bin/bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Shared launcher for LoRA fine-tuning of Wan2.2-I2V-A14B. Sourced by the entry
# scripts in this directory; not meant to be run directly.
#
# Wan2.2-I2V-A14B has two experts, so every method trains two LoRAs:
#   high: high-noise expert (pipe.dit),  timestep range [0, 0.358]
#   low:  low-noise expert  (pipe.dit2), timestep range [0.358, 1]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_DIR="${MODEL_DIR:-/path/to/Wan2.2-I2V-A14B}"
DATA_DIR="${DATA_DIR:-/path/to/OpenCoF-17k}"
METADATA="${METADATA:-$DATA_DIR/metadata.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

WANDB_ARGS=()
if [ -n "$WANDB_PROJECT" ]; then
  WANDB_ARGS=(--use_wandb --wandb_project "$WANDB_PROJECT")
fi

# train_expert <high|low> <run_name> [extra train.py args...]
train_expert() {
  local expert="$1" run_name="$2"
  shift 2

  local boundary_args
  case "$expert" in
    high) boundary_args=(--max_timestep_boundary 0.358 --min_timestep_boundary 0) ;;
    low)  boundary_args=(--max_timestep_boundary 1 --min_timestep_boundary 0.358) ;;
    *) echo "unknown expert: $expert" >&2; return 1 ;;
  esac

  local shards=()
  for i in 1 2 3 4 5 6; do
    shards+=("\"$MODEL_DIR/${expert}_noise_model/diffusion_pytorch_model-0000${i}-of-00006.safetensors\"")
  done
  local model_paths
  model_paths="[[$(IFS=,; echo "${shards[*]}")], \"$MODEL_DIR/Wan2.1_VAE.pth\", \"$MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth\"]"

  echo "[$run_name] training ${expert}-noise LoRA on $NNODES node(s) x $GPUS_PER_NODE GPU(s)"
  torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "$REPO_ROOT/examples/wanvideo/model_training/train.py" \
    --dataset_base_path "$DATA_DIR" \
    --dataset_metadata_path "$METADATA" \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --dataset_repeat 1 \
    --model_paths "$model_paths" \
    --tokenizer_path "$MODEL_DIR/google/umt5-xxl" \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "$OUTPUT_DIR/$run_name/${expert}_noise_lora" \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    --extra_inputs "input_image" \
    "${boundary_args[@]}" \
    --save_state_steps 100 \
    --save_state_total_limit 2 \
    --resume_from_checkpoint \
    "${WANDB_ARGS[@]}" \
    --wandb_run_name "${run_name}_${expert}_noise_lora" \
    "$@"
}
