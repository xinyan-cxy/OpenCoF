# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of DiffSynth-Studio, released under the Apache License 2.0:
# https://github.com/modelscope/DiffSynth-Studio
def FluxTextEncoderT5StateDictConverter(state_dict):
    state_dict_ = {i: state_dict[i] for i in state_dict}
    state_dict_["encoder.embed_tokens.weight"] = state_dict["shared.weight"]
    return state_dict_
