"""End-to-end benchmarking for the CS336 Transformer language model.

Times the forward pass, backward pass, and full training step (forward +
backward + optimizer update) of the assignment-1 ``BasicsTransformerLM`` model,
reporting the mean and standard deviation of per-step timings.

Examples:
    uv run python cs336_systems/benchmark.py --size small --phase fwd
    uv run python cs336_systems/benchmark.py --size medium --phase fwd-bwd
    uv run python cs336_systems/benchmark.py --size small --phase fwd-bwd-opt --bf16
"""

from __future__ import annotations

import argparse
import statistics
import timeit
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.common import BATCH, CONTEXT_LEN, VOCAB_LEN

# Resolve the assignment-1 dataset relative to this file. The repo layout is:
#   <repo>/assignment1-basics/data/tinystories_small/...  and
#   <repo>/assignment2-systems/cs336_systems/benchmark.py
# so `parents[2]` is the shared `<repo>` directory that holds both.
DATA_PATH = Path(__file__).resolve().parents[2] / "assignment1-basics" / "data" / "tinystories_small" / "tinystories_small-train-tokens.npy"
WARM_UP = 5
MEASURE_STEPS = 10

# Model-size presets from Section 2.1.2 (Table 1) of the assignment handout.
MODEL_CONFIGS = {
    # name -> d_model, d_ff, num_layers, num_heads
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10B": (4608, 12288, 50, 36),
}


def get_autocast_context(use_bf16: bool, device: str):
    """Return a BF16 autocast context for mixed precision, or a no-op."""
    if not use_bf16:
        return nullcontext()
    device_type = "cuda" if "cuda" in device else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


def run_one_step(
    phase: str,
    model: BasicsTransformerLM,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    autocast_context,
) -> None:
    """Execute a single forward / forward+backward / full training step."""
    if phase != "fwd":
        optimizer.zero_grad(set_to_none=True)

    with autocast_context:
        logits = model(x)
        if phase != "fwd":
            loss = cross_entropy(logits, y)

    if phase == "fwd":
        return

    loss.backward()
    if phase == "fwd-bwd-opt":
        optimizer.step()

    if "cuda" in device:
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the CS336 Transformer LM.")
    parser.add_argument(
        "--size",
        choices=sorted(MODEL_CONFIGS),
        default=None,
        help="Model-size preset from Table 1 (overrides the default 'small' config).",
    )
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LEN)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--warmup-steps", type=int, default=WARM_UP)
    parser.add_argument("--measure-steps", type=int, default=MEASURE_STEPS)
    parser.add_argument(
        "--phase",
        choices=["fwd", "fwd-bwd", "fwd-bwd-opt"],
        default="fwd-bwd",
        help="Which part of training to time.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Run with BF16 autocast mixed precision.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to .npy token data (defaults to DATA_PATH if it exists).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (defaults to cuda if available, else cpu).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data generation.")
    return parser.parse_args()


def resolve_model_config(args: argparse.Namespace) -> dict:
    """Resolve model hyperparameters from the ``--size`` preset and explicit flags."""
    d_model, d_ff, num_layers, num_heads = MODEL_CONFIGS[args.size] if args.size else MODEL_CONFIGS["small"]
    config = {"d_model": d_model, "d_ff": d_ff, "num_layers": num_layers, "num_heads": num_heads}
    for key in ("d_model", "d_ff", "num_layers", "num_heads"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def run() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    config = resolve_model_config(args)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device)
    optimizer = AdamW(model.parameters())
    autocast_context = get_autocast_context(args.bf16, device)

    data_path = args.data_path or DATA_PATH
    dataset = np.load(data_path, mmap_mode="r")
    x, y = get_batch(dataset, args.batch_size, args.context_length, device)

    print(
        f"device={device} phase={args.phase} bf16={args.bf16} "
        f"d_model={config['d_model']} d_ff={config['d_ff']} "
        f"num_layers={config['num_layers']} num_heads={config['num_heads']} "
        f"context_length={args.context_length} batch_size={args.batch_size} "
        f"params={model.get_num_params() / 1e6:.2f}M"
    )

    # Warm-up steps: run without measuring to trigger lazy init / cuBLAS autotuning.
    for _ in range(args.warmup_steps):
        run_one_step(args.phase, model, optimizer, x, y, autocast_context)

    # Measurement steps.
    timings: list[float] = []
    for _ in range(args.measure_steps):
        start = timeit.default_timer()
        run_one_step(args.phase, model, optimizer, x, y, autocast_context)

        end = timeit.default_timer()
        timings.append(end - start)

    mean = statistics.mean(timings)
    stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
    tokens_per_step = args.batch_size * args.context_length

    print(
        f"mean={mean * 1e3:.3f} ms/step  stdev={stdev * 1e3:.3f} ms/step  "
        f"({tokens_per_step / mean:.1f} tokens/s)"
    )


if __name__ == "__main__":
    run()
