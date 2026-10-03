#!/bin/bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Visual Reasoning Tokens (VT): 16 learnable tokens prepended to the visual
# token sequence of every DiT block.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

COF_ARGS=(--num_reasoning_tokens 16 --num_cond_reasoning_tokens 0
          --reasoning_tokens_log_interval 100)

train_expert high wan_cof_vt "${COF_ARGS[@]}"
train_expert low  wan_cof_vt "${COF_ARGS[@]}"
