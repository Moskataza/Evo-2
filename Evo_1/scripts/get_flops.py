import argparse
import json
import os
import sys
import time

import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo1 import EVO1


def count_parameters(module: torch.nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def build_model_config(args):
    config = {}
    if args.config_path is not None:
        if not os.path.exists(args.config_path):
            raise FileNotFoundError(f"Missing config file: {args.config_path}")
        with open(args.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    cli_config = {
        "device": args.device,
        "run_name": args.run_name,
        "vlm_name": args.vlm_name,
        "action_head": args.action_head,
        "return_cls_only": args.return_cls_only,
        "disable_wandb": args.disable_wandb,
        "dataset_type": args.dataset_type,
        "data_paths": args.data_paths,
        "dataset_config_path": args.dataset_config_path,
        "image_size": args.image_size,
        "binarize_gripper": args.binarize_gripper,
        "use_augmentation": args.use_augmentation,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "grad_clip_norm": args.grad_clip_norm,
        "weight_decay": args.weight_decay,
        "log_interval": args.log_interval,
        "ckpt_interval": args.ckpt_interval,
        "save_dir": args.save_dir,
        "resume": args.resume,
        "resume_path": args.resume_path,
        "resume_pretrain": args.resume_pretrain,
        "finetune_vlm": args.finetune_vlm,
        "finetune_action_head": args.finetune_action_head,
        "per_action_dim": args.per_action_dim,
        "state_dim": args.state_dim,
        "horizon": args.horizon,
        "num_layers": args.num_layers,
        "num_workers": args.num_workers,
        "dropout": args.dropout,
    }
    config.update(cli_config)

    if args.num_inference_timesteps is not None:
        config["num_inference_timesteps"] = args.num_inference_timesteps
    else:
        config.setdefault("num_inference_timesteps", 32)

    config.setdefault("embed_dim", 896)
    config.setdefault("state_dim", 24)
    return config


def build_dummy_inputs(args):
    image_size = args.image_size
    images = [
        torch.randint(0, 256, (3, image_size, image_size), dtype=torch.uint8)
        for _ in range(3)
    ]

    if args.image_mask_mode == "metaworld":
        image_mask = torch.tensor([1, 0, 0], dtype=torch.int32, device=args.device)
    elif args.image_mask_mode == "libero":
        image_mask = torch.tensor([1, 1, 0], dtype=torch.int32, device=args.device)
    elif args.image_mask_mode == "all":
        image_mask = torch.tensor([1, 1, 1], dtype=torch.int32, device=args.device)
    else:
        raise ValueError(f"Unknown image_mask_mode: {args.image_mask_mode}")

    state = torch.randn(args.state_dim, dtype=torch.float32, device=args.device)

    action_mask = torch.zeros(1, args.per_action_dim, dtype=torch.int32, device=args.device)
    if args.action_mask_mode == "metaworld":
        action_mask[:, : min(4, args.per_action_dim)] = 1
    elif args.action_mask_mode == "libero":
        action_mask[:, : min(7, args.per_action_dim)] = 1
    elif args.action_mask_mode == "all":
        action_mask[:, :] = 1
    else:
        raise ValueError(f"Unknown action_mask_mode: {args.action_mask_mode}")

    prompt = args.prompt
    return images, image_mask, state, action_mask, prompt


def build_action_head_inputs(model: EVO1, config: dict, args):
    device = args.device
    dtype = next(model.parameters()).dtype
    embed_dim = config.get("embed_dim", 896)

    fused_tokens = torch.randn(1, args.seq_len, embed_dim, dtype=dtype, device=device)
    state = torch.randn(1, args.state_dim, dtype=torch.float32, device=device)

    action_mask = torch.zeros(1, args.per_action_dim, dtype=torch.int32, device=device)
    if args.action_mask_mode == "metaworld":
        action_mask[:, : min(4, args.per_action_dim)] = 1
    elif args.action_mask_mode == "libero":
        action_mask[:, : min(7, args.per_action_dim)] = 1
    elif args.action_mask_mode == "all":
        action_mask[:, :] = 1
    else:
        raise ValueError(f"Unknown action_mask_mode: {args.action_mask_mode}")

    return fused_tokens, state, action_mask


def run_single_inference(model: EVO1, images, image_mask, prompt, state, action_mask):
    return model.run_inference(
        images=images,
        image_mask=image_mask,
        prompt=prompt,
        state_input=state,
        action_mask=action_mask,
    )


def run_action_head_inference(model: EVO1, fused_tokens, state, action_mask):
    return model.predict_action(
        fused_tokens=fused_tokens,
        state=state,
        action_mask=action_mask,
    )


def measure_full_latency_ms(model: EVO1, images, image_mask, prompt, state, action_mask, device: str, warmup_runs: int):
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = run_single_inference(model, images, image_mask, prompt, state, action_mask)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start_event.record()
            output = run_single_inference(model, images, image_mask, prompt, state, action_mask)
            end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
    else:
        start_time = time.perf_counter()
        with torch.no_grad():
            output = run_single_inference(model, images, image_mask, prompt, state, action_mask)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

    return latency_ms, output


def measure_action_head_latency_ms(model: EVO1, fused_tokens, state, action_mask, device: str, warmup_runs: int):
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = run_action_head_inference(model, fused_tokens, state, action_mask)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start_event.record()
            output = run_action_head_inference(model, fused_tokens, state, action_mask)
            end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
    else:
        start_time = time.perf_counter()
        with torch.no_grad():
            output = run_action_head_inference(model, fused_tokens, state, action_mask)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

    return latency_ms, output


def main():
    parser = argparse.ArgumentParser(description="Measure EVO-1 single-inference latency with random initialization.")
    parser.add_argument("--config_path", type=str, default=None, help="Optional config.json path used as a base config")

    # Basic config, aligned with train.py
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--vlm_name", type=str, default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--action_head", type=str, default="flowmatching", choices=["flowmatching", "blockbottleneck"])
    parser.add_argument("--return_cls_only", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="lerobot")
    parser.add_argument("--data_paths", type=str, required=False)
    parser.add_argument("--dataset_config_path", type=str, required=False, default=None)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--binarize_gripper", action="store_true", default=False)
    parser.add_argument("--use_augmentation", action="store_true")

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # Resume
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_pretrain", action="store_true")

    # Finetuning
    parser.add_argument("--finetune_vlm", action="store_true")
    parser.add_argument("--finetune_action_head", action="store_true")

    # Misc
    parser.add_argument("--per_action_dim", type=int, default=7)
    parser.add_argument("--state_dim", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Inference timing only
    parser.add_argument("--num_inference_timesteps", type=int, default=None)
    parser.add_argument("--profile_mode", type=str, default="full", choices=["full", "action_head"])
    parser.add_argument("--image_mask_mode", type=str, default="metaworld", choices=["metaworld", "libero", "all"])
    parser.add_argument("--action_mask_mode", type=str, default="metaworld", choices=["metaworld", "libero", "all"])
    parser.add_argument("--prompt", type=str, default="pick up the object and place it at the target position")
    parser.add_argument("--seq_len", type=int, default=1024, help="Used only in action_head mode")
    parser.add_argument("--warmup_runs", type=int, default=1, help="Warmup runs before the single timed inference")
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("This script measures one EVO1 inference at a time, so --batch_size must be 1.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device points to CUDA.")

    torch.manual_seed(0)

    config_source = args.config_path if args.config_path is not None else "CLI arguments"
    print(f"Building randomly initialized EVO-1 model from: {config_source}")
    config = build_model_config(args)
    model = EVO1(config).eval().to(args.device)

    total_params, trainable_params = count_parameters(model)
    action_head_params, _ = count_parameters(model.action_head)
    if args.profile_mode == "full":
        images, image_mask, state, action_mask, prompt = build_dummy_inputs(args)
        latency_ms, output = measure_full_latency_ms(
            model=model,
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state=state,
            action_mask=action_mask,
            device=args.device,
            warmup_runs=args.warmup_runs,
        )
        prompt_length = len(prompt)
        extra_lines = [
            f"image_size           : {args.image_size}",
            f"image_mask_mode      : {args.image_mask_mode}",
            f"prompt_length        : {prompt_length}",
        ]
    else:
        fused_tokens, state, action_mask = build_action_head_inputs(model, config, args)
        latency_ms, output = measure_action_head_latency_ms(
            model=model,
            fused_tokens=fused_tokens,
            state=state,
            action_mask=action_mask,
            device=args.device,
            warmup_runs=args.warmup_runs,
        )
        extra_lines = [
            f"seq_len              : {args.seq_len}",
            f"embed_dim            : {config.get('embed_dim', 896)}",
        ]

    output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else "unknown"

    print("\n===== EVO-1 Single Inference Timing =====")
    print(f"config_source        : {config_source}")
    print("weights              : random initialization")
    print(f"profile_mode         : {args.profile_mode}")
    print(f"device               : {args.device}")
    print(f"action_head          : {config.get('action_head', 'flowmatching')}")
    print(f"vlm_name             : {config.get('vlm_name')}")
    print(f"action_mask_mode     : {args.action_mask_mode}")
    print(f"state_dim            : {args.state_dim}")
    print(f"horizon              : {args.horizon}")
    print(f"per_action_dim       : {args.per_action_dim}")
    print(f"num_layers           : {args.num_layers}")
    print(f"num_inference_steps  : {config.get('num_inference_timesteps')}")
    print(f"warmup_runs          : {args.warmup_runs}")
    for line in extra_lines:
        print(line)
    print(f"total_params         : {total_params}")
    print(f"trainable_params     : {trainable_params}")
    print(f"action_head_params   : {action_head_params}")
    print(f"single_latency_ms    : {latency_ms:.4f}")
    if latency_ms > 0:
        print(f"single_frequency_hz  : {1000.0 / latency_ms:.4f}")
    print(f"output_shape         : {output_shape}")


if __name__ == "__main__":
    main()
