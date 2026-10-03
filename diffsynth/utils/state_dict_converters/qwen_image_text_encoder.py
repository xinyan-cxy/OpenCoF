# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of DiffSynth-Studio, released under the Apache License 2.0:
# https://github.com/modelscope/DiffSynth-Studio
def QwenImageTextEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for k in state_dict:
        v = state_dict[k]
        if k.startswith("visual."):
            k = "model." + k
        elif k.startswith("model."):
            k = k.replace("model.", "model.language_model.")
        state_dict_[k] = v
    return state_dict_
