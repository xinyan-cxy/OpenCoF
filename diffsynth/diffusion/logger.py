# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
# The original file is part of DiffSynth-Studio, released under the Apache
# License 2.0: https://github.com/modelscope/DiffSynth-Studio
import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, reasoning_tokens_log_interval=0):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.reasoning_tokens_log_interval = reasoning_tokens_log_interval
        self._reasoning_trace = []
        self.optimizer_steps = 0

    def _append_reasoning_tokens_snapshot(self, accelerator: Accelerator, model: torch.nn.Module):
        if not accelerator.is_main_process:
            return
        pipe = accelerator.unwrap_model(model).pipe
        row = {"optimizer_step": self.optimizer_steps}
        if hasattr(pipe, "dit") and pipe.dit is not None and hasattr(pipe.dit, "reasoning_tokens"):
            row["dit"] = pipe.dit.reasoning_tokens.detach().float().cpu().clone()
        else:
            row["dit"] = None
        if hasattr(pipe, "dit2") and pipe.dit2 is not None and hasattr(pipe.dit2, "reasoning_tokens"):
            row["dit2"] = pipe.dit2.reasoning_tokens.detach().float().cpu().clone()
        else:
            row["dit2"] = None
        self._reasoning_trace.append(row)

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        if accelerator.sync_gradients:
            self.optimizer_steps += 1
            if self.reasoning_tokens_log_interval and self.optimizer_steps % self.reasoning_tokens_log_interval == 0:
                self._append_reasoning_tokens_snapshot(accelerator, model)


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        if self._reasoning_trace and accelerator.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, "reasoning_tokens_trace.pt")
            torch.save(self._reasoning_trace, path)


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

    def state_dict(self):
        return {
            "num_steps": self.num_steps,
            "optimizer_steps": self.optimizer_steps,
        }

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        self.num_steps = int(state_dict.get("num_steps", 0))
        self.optimizer_steps = int(state_dict.get("optimizer_steps", 0))
