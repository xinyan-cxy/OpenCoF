# Copyright (c) 2023 Zhongjie Duan and the ModelScope Community
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
# The original file is part of DiffSynth-Studio, released under the Apache
# License 2.0: https://github.com/modelscope/DiffSynth-Studio
import hashlib
import json, os, random, re, shutil, time, torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    save_state_steps: int = None,
    save_state_total_limit: int = 2,
    resume_from_checkpoint: str = None,
    num_epochs: int = 1,
    args = None,
):
    def _debug_log(run_id, hypothesis_id, message, data):
        # No-op: internal debug logging removed for the open-source release.
        return

    def _rank_log(event, data=None):
        try:
            output_path = getattr(model_logger, "output_path", None)
            if not output_path:
                return
            os.makedirs(output_path, exist_ok=True)
            payload = {
                "ts": int(time.time() * 1000),
                "rank": accelerator.process_index,
                "event": event,
                "step": getattr(model_logger, "num_steps", None),
                "optimizer_step": getattr(model_logger, "optimizer_steps", None),
                "data": data or {},
            }
            path = os.path.join(output_path, f"rank-{accelerator.process_index}.log")
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            pass

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
        return signature

    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        save_state_steps = getattr(args, "save_state_steps", save_state_steps)
        save_state_total_limit = getattr(args, "save_state_total_limit", save_state_total_limit)
        resume_from_checkpoint = getattr(args, "resume_from_checkpoint", resume_from_checkpoint)
        num_epochs = args.num_epochs
        seed = getattr(args, "seed", 42)
    else:
        seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # region agent log
    _debug_log(
        "resume_debug",
        "H1",
        "args_resolved",
        {
            "resume_from_checkpoint": resume_from_checkpoint,
            "save_state_steps": save_state_steps,
            "save_state_total_limit": save_state_total_limit,
            "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", None),
            "output_path": getattr(model_logger, "output_path", None),
            "seed": seed,
            "num_processes": accelerator.num_processes,
            "process_index": accelerator.process_index,
            "distributed_type": str(getattr(accelerator, "distributed_type", None)),
        },
    )
    # endregion
    use_wandb = getattr(args, "use_wandb", False)
    if use_wandb and accelerator.is_main_process:
        try:
            import wandb
            wandb_project = getattr(args, "wandb_project", "diffsynth")
            wandb_run_name = getattr(args, "wandb_run_name", None)
            wandb_run_id = getattr(args, "wandb_run_id", None) or os.environ.get("WANDB_RUN_ID")
            wandb_kwargs = {
                "project": wandb_project,
                "name": wandb_run_name,
                "config": vars(args) if args else None,
            }
            if resume_from_checkpoint:
                if not wandb_run_id and wandb_run_name:
                    safe_run_name = re.sub(r"[^A-Za-z0-9_-]", "_", str(wandb_run_name))
                    if len(safe_run_name) > 64:
                        digest = hashlib.sha1(safe_run_name.encode("utf-8")).hexdigest()[:12]
                        prefix_len = max(1, 64 - 1 - len(digest))
                        wandb_run_id = f"{safe_run_name[:prefix_len]}-{digest}"
                        print(
                            "Info: wandb_run_name too long, using shortened wandb_run_id for resume: "
                            f"{wandb_run_id}"
                        )
                    else:
                        wandb_run_id = safe_run_name
                        print(f"Info: using wandb_run_name as wandb_run_id for resume: {wandb_run_id}")
                if wandb_run_id:
                    wandb_kwargs.update({"id": wandb_run_id, "resume": "allow"})
                else:
                    print("Warning: resume requested but WANDB_RUN_ID/--wandb_run_id not set. Starting a new wandb run.")
            # region agent log
            _debug_log(
                "resume_debug",
                "H2",
                "wandb_resume_decision",
                {
                    "resume_requested": bool(resume_from_checkpoint),
                    "wandb_run_id_set": bool(wandb_run_id),
                    "wandb_run_name": wandb_run_name,
                    "resume_mode": wandb_kwargs.get("resume"),
                },
            )
            # endregion
            wandb.init(**wandb_kwargs)
        except ImportError:
            print("Warning: wandb is not installed. Please run `pip install wandb`.")
            use_wandb = False

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    pre_sig = _get_reasoning_tokens_signature(model.pipe)
    model._reasoning_tokens_signature_pre = pre_sig
    _rank_log("reasoning_tokens_pre_prepare_runner", pre_sig)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    unwrapped_model = accelerator.unwrap_model(model)
    post_sig = _get_reasoning_tokens_signature(unwrapped_model.pipe)
    _rank_log("reasoning_tokens_post_prepare_runner", post_sig)
    pre_sig = getattr(unwrapped_model, "_reasoning_tokens_signature_pre", None)
    if pre_sig:
        for name in ("dit", "dit2"):
            before = pre_sig.get(name)
            after = post_sig.get(name)
            if before and after and before.get("id") != after.get("id"):
                _rank_log(
                    "reasoning_tokens_signature_mismatch",
                    {"name": name, "before": before, "after": after},
                )
                raise RuntimeError(
                    f"Reasoning tokens parameter '{name}' was re-created after DDP init on "
                    f"rank {accelerator.process_index}. This can desync buckets and cause NCCL hangs. "
                    "Ensure reasoning_tokens / cond-side embedding are finalized before accelerator.prepare()."
                )
    accelerator.register_for_checkpointing(model_logger)
    if save_state_steps is not None and save_state_steps <= 0:
        save_state_steps = None
    if save_state_total_limit is not None and save_state_total_limit <= 0:
        save_state_total_limit = None

    def _list_resume_checkpoints(root_path):
        if not os.path.isdir(root_path):
            return []
        checkpoints = []
        for name in os.listdir(root_path):
            if not name.startswith("step-"):
                continue
            path = os.path.join(root_path, name)
            if not os.path.isdir(path):
                continue
            try:
                step = int(name.split("-", 1)[1])
            except ValueError:
                continue
            checkpoints.append((step, path))
        checkpoints.sort(key=lambda item: item[0])
        return checkpoints

    def _resolve_resume_candidates(path_value):
        if not path_value:
            return [], None
        if path_value == "latest":
            resume_root = os.path.join(model_logger.output_path, "resume_ckpt")
            checkpoints = _list_resume_checkpoints(resume_root)
            if not checkpoints:
                return [], resume_root
            candidates = [path for _, path in reversed(checkpoints)]
            # region agent log
            _debug_log(
                "resume_debug",
                "H3",
                "resume_candidates",
                {
                    "resume_root": resume_root,
                    "count": len(candidates),
                    "steps_tail": [os.path.basename(p) for p in candidates[:5]],
                },
            )
            # endregion
            return candidates, resume_root
        return [path_value], None

    def _summarize_checkpoint(path_value):
        summary = {
            "exists": os.path.isdir(path_value),
            "entries_count": 0,
            "entries_sample": [],
            "file_sizes": {},
        }
        if not summary["exists"]:
            return summary
        try:
            entries = os.listdir(path_value)
            summary["entries_count"] = len(entries)
            summary["entries_sample"] = entries[:20]
            for name in entries[:20]:
                file_path = os.path.join(path_value, name)
                if os.path.isfile(file_path):
                    summary["file_sizes"][name] = os.path.getsize(file_path)
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary

    def _prune_resume_checkpoints(root_path):
        if save_state_total_limit is None:
            return
        checkpoints = _list_resume_checkpoints(root_path)
        if len(checkpoints) <= save_state_total_limit:
            return
        for _, path in checkpoints[:-save_state_total_limit]:
            shutil.rmtree(path)

    def _save_resume_checkpoint(step):
        resume_root = os.path.join(model_logger.output_path, "resume_ckpt")
        final_dir = os.path.join(resume_root, f"step-{step}")
        tmp_dir = f"{final_dir}.tmp"
        if accelerator.is_main_process:
            os.makedirs(resume_root, exist_ok=True)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
        accelerator.wait_for_everyone()
        # safetensors drops zero-sized parameters (e.g. unused reasoning tokens) as
        # "shared" tensors, which makes the checkpoint fail to load strictly.
        accelerator.save_state(tmp_dir, safe_serialization=False)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            os.replace(tmp_dir, final_dir)
            _prune_resume_checkpoints(resume_root)
        accelerator.wait_for_everyone()

    def _make_worker_init_fn(epoch_id):
        def _seed_worker(worker_id):
            worker_seed = seed + epoch_id * 1000 + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        return _seed_worker

    def _build_dataloader(epoch_id):
        collate_fn = lambda x: x[0]
        worker_init_fn = _make_worker_init_fn(epoch_id)
        generator = torch.Generator()
        generator.manual_seed(seed + epoch_id)
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            generator=generator,
            collate_fn=collate_fn,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    def _build_prepared_dataloader(epoch_id):
        dataloader = _build_dataloader(epoch_id)
        return accelerator.prepare(dataloader)

    steps_per_epoch = len(_build_prepared_dataloader(0))

    resume_candidates, resume_root = _resolve_resume_candidates(resume_from_checkpoint)
    resume_path = None
    if resume_from_checkpoint and not resume_candidates and accelerator.is_main_process:
        if resume_root:
            print(f"Warning: --resume_from_checkpoint requested but no checkpoints found under {resume_root}.")
        else:
            print(f"Warning: --resume_from_checkpoint requested but checkpoint not found: {resume_from_checkpoint}")
    for candidate in resume_candidates:
        if accelerator.is_main_process:
            print(f"Resume: trying checkpoint {candidate}")
        try:
            # region agent log
            _debug_log(
                "resume_debug",
                "H4",
                "resume_attempt",
                {
                    "candidate": candidate,
                    "checkpoint_summary": _summarize_checkpoint(candidate),
                },
            )
            # endregion
            accelerator.load_state(candidate)
        except Exception as exc:
            if accelerator.is_main_process:
                print(f"Resume: failed to load {candidate}: {type(exc).__name__}: {exc}")
            # region agent log
            _debug_log(
                "resume_debug",
                "H4",
                "resume_failed",
                {"candidate": candidate, "error": f"{type(exc).__name__}: {exc}"},
            )
            # endregion
            if resume_from_checkpoint != "latest":
                raise
            continue
        resume_path = candidate
        if accelerator.is_main_process:
            lr_value = scheduler.get_last_lr()[0] if scheduler else None
            print(
                "Resume: loaded checkpoint "
                f"{candidate} (optimizer_steps={model_logger.optimizer_steps}, lr={lr_value})"
            )
        # region agent log
        _debug_log(
            "resume_debug",
            "H5",
            "resume_loaded",
            {
                "candidate": candidate,
                "optimizer_steps": model_logger.optimizer_steps,
                "optimizer_state_len": len(optimizer.state),
                "optimizer_lr": optimizer.param_groups[0].get("lr") if optimizer.param_groups else None,
                "scheduler_lr": scheduler.get_last_lr()[0] if scheduler else None,
            },
        )
        # endregion
        break

    resume_epoch = 0
    resume_step_in_epoch = 0
    if resume_path and steps_per_epoch > 0:
        resume_total_steps = model_logger.num_steps
        resume_epoch = resume_total_steps // steps_per_epoch
        resume_step_in_epoch = resume_total_steps % steps_per_epoch
        if accelerator.is_main_process:
            lr_value = scheduler.get_last_lr()[0] if scheduler else None
            print(
                "Resume: data position "
                f"epoch={resume_epoch} step={resume_step_in_epoch}/{steps_per_epoch} seed={seed} lr={lr_value}"
            )
        # region agent log
        _debug_log(
            "resume_debug",
            "H6",
            "resume_data_position",
            {
                "resume_total_steps": resume_total_steps,
                "resume_epoch": resume_epoch,
                "resume_step_in_epoch": resume_step_in_epoch,
                "steps_per_epoch": steps_per_epoch,
                "seed": seed,
            },
        )
        # endregion
    start_epoch = resume_epoch if resume_path else 0

    for epoch_id in range(start_epoch, num_epochs):
        dataloader = _build_prepared_dataloader(epoch_id)
        if resume_path and epoch_id == resume_epoch and resume_step_in_epoch > 0:
            dataloader = accelerator.skip_first_batches(dataloader, resume_step_in_epoch)
            if accelerator.is_main_process:
                print(f"Resume: skipping {resume_step_in_epoch} batches in epoch {epoch_id}")
            # region agent log
            _debug_log(
                "resume_debug",
                "H6",
                "resume_skip_batches",
                {
                    "epoch_id": epoch_id,
                    "skipped_batches": resume_step_in_epoch,
                },
            )
            # endregion
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                _rank_log(
                    "forward_begin",
                    {"epoch": epoch_id, "global_step": model_logger.num_steps + 1},
                )
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                _rank_log(
                    "forward_end",
                    {"epoch": epoch_id, "global_step": model_logger.num_steps + 1},
                )
                accelerator.backward(loss)
                _rank_log(
                    "backward_end",
                    {
                        "epoch": epoch_id,
                        "global_step": model_logger.num_steps + 1,
                        "sync_gradients": bool(accelerator.sync_gradients),
                    },
                )
                
                rt_stats = {}
                if use_wandb and accelerator.is_main_process:
                    unwrapped_model_module = accelerator.unwrap_model(model)
                    unwrapped_model = unwrapped_model_module.pipe
                    for dit_name in ["dit", "dit2"]:
                        m = getattr(unwrapped_model, dit_name, None)
                        if m is not None and hasattr(m, "reasoning_tokens"):
                            with torch.no_grad():
                                rt = m.reasoning_tokens.float()
                                rt_count = int(rt.shape[1]) if rt.ndim >= 2 else int(rt.numel())
                                rt_stats[f"train/{dit_name}_reasoning_tokens_count"] = rt_count
                                if rt.numel() > 0:
                                    rt_stats[f"train/{dit_name}_reasoning_tokens_norm"] = rt.norm().item()
                                    rt_stats[f"train/{dit_name}_reasoning_tokens_abs_mean"] = rt.abs().mean().item()
                                    rt_stats[f"train/{dit_name}_reasoning_tokens_max"] = rt.max().item()
                                    rt_stats[f"train/{dit_name}_reasoning_tokens_min"] = rt.min().item()
                                else:
                                    rt_stats[f"train/{dit_name}_reasoning_tokens_norm"] = 0.0
                                    rt_stats[f"train/{dit_name}_reasoning_tokens_abs_mean"] = 0.0
                                    rt_stats[f"train/{dit_name}_reasoning_tokens_max"] = 0.0
                                    rt_stats[f"train/{dit_name}_reasoning_tokens_min"] = 0.0
                                
                                if m.reasoning_tokens.grad is not None and m.reasoning_tokens.grad.numel() > 0:
                                    rt_stats[f"train/{dit_name}_reasoning_tokens_grad_norm"] = m.reasoning_tokens.grad.float().norm().item()
                                else:
                                    rt_stats[f"train/{dit_name}_reasoning_tokens_grad_norm"] = 0.0
                        # Textual reasoning tokens (`cond_reasoning_emb`): same stats as above.
                        if m is not None and hasattr(m, "cond_reasoning_emb"):
                            with torch.no_grad():
                                cre = m.cond_reasoning_emb.float()
                                cre_count = int(cre.shape[1]) if cre.ndim >= 2 else int(cre.numel())
                                rt_stats[f"train/{dit_name}_cond_reasoning_emb_count"] = cre_count
                                if cre.numel() > 0:
                                    rt_stats[f"train/{dit_name}_cond_reasoning_emb_norm"] = cre.norm().item()
                                    rt_stats[f"train/{dit_name}_cond_reasoning_emb_abs_mean"] = cre.abs().mean().item()
                                    rt_stats[f"train/{dit_name}_cond_reasoning_emb_max"] = cre.max().item()
                                    rt_stats[f"train/{dit_name}_cond_reasoning_emb_min"] = cre.min().item()
                                else:
                                    rt_stats[f"train/{dit_name}_cond_reasoning_emb_norm"] = 0.0
                                    rt_stats[f"train/{dit_name}_cond_reasoning_emb_abs_mean"] = 0.0
                                    rt_stats[f"train/{dit_name}_cond_reasoning_emb_max"] = 0.0
                                    rt_stats[f"train/{dit_name}_cond_reasoning_emb_min"] = 0.0
                                if m.cond_reasoning_emb.grad is not None and m.cond_reasoning_emb.grad.numel() > 0:
                                    rt_stats[f"train/{dit_name}_cond_reasoning_emb_grad_norm"] = m.cond_reasoning_emb.grad.float().norm().item()
                                else:
                                    rt_stats[f"train/{dit_name}_cond_reasoning_emb_grad_norm"] = 0.0

                optimizer.step()
                _rank_log(
                    "optimizer_step_end",
                    {
                        "epoch": epoch_id,
                        "global_step": model_logger.num_steps + 1,
                        "sync_gradients": bool(accelerator.sync_gradients),
                    },
                )
                
                if use_wandb and accelerator.is_main_process:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch_id,
                        "train/global_step": model_logger.num_steps + 1,
                    }
                    log_dict.update(rt_stats)
                    wandb.log(log_dict)
                
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
                if save_state_steps is not None and accelerator.sync_gradients:
                    current_step = model_logger.optimizer_steps
                    if current_step % save_state_steps == 0:
                        _save_resume_checkpoint(current_step)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    if save_state_steps is not None:
        last_step = model_logger.optimizer_steps
        if last_step > 0 and last_step % save_state_steps != 0:
            _save_resume_checkpoint(last_step)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
