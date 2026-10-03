# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of DiffSynth-Studio, released under the Apache License 2.0:
# https://github.com/modelscope/DiffSynth-Studio
def WanVideoVAEStateDictConverter(state_dict):
    state_dict_ = {}
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    for name in state_dict:
        state_dict_['model.' + name] = state_dict[name]
    return state_dict_