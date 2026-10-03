# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
# The original file is part of DiffSynth-Studio, released under the Apache
# License 2.0: https://github.com/modelscope/DiffSynth-Studio
import torch, os, argparse, accelerate, warnings, json, time
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def apply_num_reasoning_tokens(pipe, num: int, torch_dtype, device):
    """Resize/re-init reasoning_tokens on dit/dit2 when CLI count differs from checkpoint (after from_pretrained)."""
    for name in ("dit", "dit2"):
        m = getattr(pipe, name, None)
        if m is None or not hasattr(m, "reasoning_tokens"):
            continue
        if m.reasoning_tokens.shape[1] == num:
            m.num_reasoning_tokens = num
            continue
        dim = m.dim
        with torch.no_grad():
            rt = torch.randn(1, num, dim, device=device, dtype=torch_dtype) / (dim ** 0.5)
        m.num_reasoning_tokens = num
        m.reasoning_tokens = torch.nn.Parameter(rt, requires_grad=True)


def apply_num_cond_reasoning_emb(pipe, num: int, torch_dtype, device):
    """Resize/re-init the textual reasoning tokens (`cond_reasoning_emb`) on dit/dit2.

    Mirrors `apply_num_reasoning_tokens`. The WanModel ctor creates the param
    with N=0 by default; this helper grows it to the CLI-requested N right
    after `from_pretrained`. Init: per-row N(0, 1/sqrt(dim)).
    """
    for name in ("dit", "dit2"):
        m = getattr(pipe, name, None)
        if m is None or not hasattr(m, "cond_reasoning_emb"):
            continue
        if m.cond_reasoning_emb.shape[1] == num:
            m.num_cond_reasoning_emb = num
            continue
        dim = m.dim
        with torch.no_grad():
            cre = torch.randn(1, num, dim, device=device, dtype=torch_dtype) / (dim ** 0.5)
        m.num_cond_reasoning_emb = num
        m.cond_reasoning_emb = torch.nn.Parameter(cre, requires_grad=(num > 0))


def _get_reasoning_tokens_signature(pipe):
    signature = {}
    for name in ("dit", "dit2"):
        m = getattr(pipe, name, None)
        if m is None or not hasattr(m, "reasoning_tokens"):
            signature[name] = None
            continue
        p = m.reasoning_tokens
        signature[name] = {
            "id": id(p),
            "shape": list(p.shape),
            "device": str(p.device),
            "dtype": str(p.dtype),
            "requires_grad": bool(p.requires_grad),
        }
        # Textual reasoning tokens (`cond_reasoning_emb`). Lives on each
        # DiT alongside `reasoning_tokens`. Tracking shape + abs_mean here lets
        # the prepare-pre vs prepare-post comparison detect any DDP-induced
        # parameter rebuilds that would silently desync the value across ranks.
        if hasattr(m, "cond_reasoning_emb"):
            cre = m.cond_reasoning_emb
            try:
                # `cre.detach()` may live on meta device when `--initialize_model_on_cpu`
                # is set; `.float().abs().mean()` would fail there, so guard it.
                if cre.device.type != "meta" and cre.numel() > 0:
                    abs_mean = float(cre.detach().float().abs().mean().item())
                else:
                    abs_mean = None
            except Exception:
                abs_mean = None
            signature[name + "_cond_reasoning_emb"] = {
                "id": id(cre),
                "shape": list(cre.shape),
                "device": str(cre.device),
                "dtype": str(cre.dtype),
                "requires_grad": bool(cre.requires_grad),
                "abs_mean": abs_mean,
            }
        else:
            signature[name + "_cond_reasoning_emb"] = None
    return signature


def _write_rank_log(output_path, rank, event, data):
    if not output_path:
        return
    try:
        os.makedirs(output_path, exist_ok=True)
        payload = {
            "ts": int(time.time() * 1000),
            "rank": rank,
            "event": event,
            "data": data,
        }
        path = os.path.join(output_path, f"rank-{rank}.log")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        num_reasoning_tokens=16,
        num_cond_reasoning_tokens=0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        # audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        # self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            num_cond_reasoning_tokens=num_cond_reasoning_tokens,
        )
        apply_num_reasoning_tokens(self.pipe, num_reasoning_tokens, self.pipe.torch_dtype, self.pipe.device)
        apply_num_cond_reasoning_emb(self.pipe, num_cond_reasoning_tokens, self.pipe.torch_dtype, self.pipe.device)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # === Unfreeze Reasoning Tokens (they are not LoRA params, so freeze_except would freeze them) ===
        if hasattr(self.pipe, 'dit') and self.pipe.dit is not None and hasattr(self.pipe.dit, 'reasoning_tokens'):
            # import ipdb; ipdb.set_trace()
            self.pipe.dit.reasoning_tokens.requires_grad_(True)
            if self.pipe.dit.reasoning_tokens.device.type == 'meta':
                with torch.no_grad():
                    real_tensor = torch.randn(1, self.pipe.dit.num_reasoning_tokens, self.pipe.dit.dim, device=self.pipe.device, dtype=self.pipe.torch_dtype) / (self.pipe.dit.dim ** 0.5)
                    self.pipe.dit.reasoning_tokens = torch.nn.Parameter(real_tensor, requires_grad=True)
            self.pipe.dit.reasoning_tokens.register_hook(lambda grad: grad.clone(memory_format=torch.contiguous_format) if grad is not None else grad)
        if hasattr(self.pipe, 'dit2') and self.pipe.dit2 is not None and hasattr(self.pipe.dit2, 'reasoning_tokens'):
            self.pipe.dit2.reasoning_tokens.requires_grad_(True)
            if self.pipe.dit2.reasoning_tokens.device.type == 'meta':
                with torch.no_grad():
                    real_tensor = torch.randn(1, self.pipe.dit2.num_reasoning_tokens, self.pipe.dit2.dim, device=self.pipe.device, dtype=self.pipe.torch_dtype) / (self.pipe.dit2.dim ** 0.5)
                    self.pipe.dit2.reasoning_tokens = torch.nn.Parameter(real_tensor, requires_grad=True)
            self.pipe.dit2.reasoning_tokens.register_hook(lambda grad: grad.clone(memory_format=torch.contiguous_format) if grad is not None else grad)
        # ========================================================

        # === Unfreeze Textual Reasoning Tokens ===
        # Mirror of the `reasoning_tokens` unfreeze block above. With N=0 the
        # param has shape (1, 0, dim) and stays frozen.
        if num_cond_reasoning_tokens > 0:
            for dit_attr in ("dit", "dit2"):
                m = getattr(self.pipe, dit_attr, None)
                if m is None or not hasattr(m, "cond_reasoning_emb"):
                    continue
                m.cond_reasoning_emb.requires_grad_(True)
                if m.cond_reasoning_emb.device.type == "meta":
                    with torch.no_grad():
                        real_tensor = torch.randn(1, m.num_cond_reasoning_emb, m.dim, device=self.pipe.device, dtype=self.pipe.torch_dtype) / (m.dim ** 0.5)
                        m.cond_reasoning_emb = torch.nn.Parameter(real_tensor, requires_grad=True)
                m.cond_reasoning_emb.register_hook(
                    lambda grad: grad.clone(memory_format=torch.contiguous_format) if grad is not None else grad
                )
        # ===============================================================
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
    
    @staticmethod
    def _normalize_text(value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            if value != value:
                return ""
        except Exception:
            pass
        return str(value)

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": self._normalize_text(data.get("prompt", ""))}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--num_reasoning_tokens", type=int, default=16, help="Number of learnable reasoning tokens prepended in Wan DiT (dit/dit2).")
    parser.add_argument("--num_cond_reasoning_tokens", type=int, default=0, help="Number of learnable textual reasoning tokens prepended to the encoded text sequence in Wan DiT (dit/dit2).")
    parser.add_argument("--reasoning_tokens_log_interval", type=int, default=100, help="Every N optimizer steps save reasoning token tensors; 0 disables. Written to reasoning_tokens_trace.pt at end.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_cond_reasoning_tokens=args.num_cond_reasoning_tokens,
        num_reasoning_tokens=args.num_reasoning_tokens,
    )
    pre_sig = _get_reasoning_tokens_signature(model.pipe)
    model._reasoning_tokens_signature_pre = pre_sig
    _write_rank_log(
        args.output_path,
        accelerator.process_index,
        "reasoning_tokens_pre_prepare_train_py",
        pre_sig,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        reasoning_tokens_log_interval=args.reasoning_tokens_log_interval,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
