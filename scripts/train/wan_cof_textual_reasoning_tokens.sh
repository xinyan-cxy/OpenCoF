#!/bin/bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Textual Reasoning Tokens (TT): 16 learnable tokens prepended to the encoded
# (post-T5) text sequence consumed by cross-attention.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

COF_ARGS=(--num_reasoning_tokens 0 --num_cond_reasoning_tokens 16
          --reasoning_tokens_log_interval 100 --find_unused_parameters)

train_expert high wan_cof_tt "${COF_ARGS[@]}"
train_expert low  wan_cof_tt "${COF_ARGS[@]}"
