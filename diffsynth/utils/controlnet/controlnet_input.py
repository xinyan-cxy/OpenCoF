# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of DiffSynth-Studio, released under the Apache License 2.0:
# https://github.com/modelscope/DiffSynth-Studio
from dataclasses import dataclass
from PIL import Image


@dataclass
class ControlNetInput:
    controlnet_id: int = 0
    scale: float = 1.0
    start: float = 1.0
    end: float = 0.0
    image: Image.Image = None
    inpaint_mask: Image.Image = None
    processor_id: str = None
