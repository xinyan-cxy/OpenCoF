# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of DiffSynth-Studio, released under the Apache License 2.0:
# https://github.com/modelscope/DiffSynth-Studio
def WanImageEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("textual."):
            continue
        name_ = "model." + name
        state_dict_[name_] = state_dict[name]
    return state_dict_