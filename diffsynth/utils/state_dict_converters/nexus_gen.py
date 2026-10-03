# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of DiffSynth-Studio, released under the Apache License 2.0:
# https://github.com/modelscope/DiffSynth-Studio
def NexusGenAutoregressiveModelStateDictConverter(state_dict):
    new_state_dict = {}
    for key in state_dict:
        value = state_dict[key]
        new_state_dict["model." + key] = value
    return new_state_dict