#!/bin/bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Wan-CoF: LoRA fine-tuning of Wan2.2-I2V-A14B on OpenCoF-17k, no reasoning tokens.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

COF_ARGS=(--num_reasoning_tokens 0 --num_cond_reasoning_tokens 0
          --reasoning_tokens_log_interval 0)

train_expert high wan_cof "${COF_ARGS[@]}"
train_expert low  wan_cof "${COF_ARGS[@]}"
