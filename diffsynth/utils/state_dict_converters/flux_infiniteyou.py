# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of DiffSynth-Studio, released under the Apache License 2.0:
# https://github.com/modelscope/DiffSynth-Studio
def FluxInfiniteYouImageProjectorStateDictConverter(state_dict):
    return state_dict['image_proj']