# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of DiffSynth-Studio, released under the Apache License 2.0:
# https://github.com/modelscope/DiffSynth-Studio
def WanS2VAudioEncoderStateDictConverter(state_dict):
    rename_dict = {
        "model.wav2vec2.encoder.pos_conv_embed.conv.weight_g": "model.wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        "model.wav2vec2.encoder.pos_conv_embed.conv.weight_v": "model.wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    }
    state_dict_ = {}
    for name in state_dict:
        name_ = "model." + name
        if name_ in rename_dict:
            name_ = rename_dict[name_]
        state_dict_[name_] = state_dict[name]
    return state_dict_
