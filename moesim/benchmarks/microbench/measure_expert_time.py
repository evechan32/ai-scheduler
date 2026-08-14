#!/usr/bin/env python3
"""Measure per-expert FFN time on GPU and CPU for scheduler calibration."""
import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as a plain script without installing moesim.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def time_fn(fn, repeats=20) -> float:
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000.0  # ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--expert-mb", type=float, default=340.0)
    parser.add_argument("--out", type=str, default="benchmarks/microbench/out/profiles.json")
    args = parser.parse_args()

    import torch

    from moesim.executor.cpu_kernels import expert_ffn

    assert torch.cuda.is_available(), "requires a CUDA GPU"
    x_gpu = torch.randn(1, args.hidden, dtype=torch.float16, device="cuda")
    w1 = torch.randn(args.intermediate, args.hidden, dtype=torch.float16, device="cuda")
    w2 = torch.randn(args.hidden, args.intermediate, dtype=torch.float16, device="cuda")

    gpu_ms = time_fn(lambda: torch.nn.functional.gelu(x_gpu @ w1.t()) @ w2.t())

    x_cpu = x_gpu.cpu()
    w1_cpu = w1.cpu()
    w2_cpu = w2.cpu()
    cpu_ms = time_fn(lambda: expert_ffn(x_cpu, w1_cpu, w2_cpu))

    profiles = [
        {
            "expert_id": f"e{i}",
            "size_mb": args.expert_mb,
            "gpu_exec_ms": round(gpu_ms, 4),
            "cpu_exec_ms": round(cpu_ms, 4),
        }
        for i in range(args.num_experts)
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(profiles, indent=2))
    print(f"GPU {gpu_ms:.3f}ms / CPU {cpu_ms:.3f}ms per expert -> {args.out}")


if __name__ == "__main__":
    main()
