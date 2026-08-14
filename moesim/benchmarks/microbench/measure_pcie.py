#!/usr/bin/env python3
"""Measure effective PCIe bandwidth (GPU <-> CPU) for calibration."""
import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=float, default=340.0)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", type=str, default="benchmarks/microbench/out/pcie.json")
    args = parser.parse_args()

    import torch

    assert torch.cuda.is_available(), "requires a CUDA GPU"
    size = int(args.size_mb * 1024 * 1024 // 2)  # fp16 elements
    src = torch.randn(size, dtype=torch.float16, device="cuda")
    dst = torch.empty(size, dtype=torch.float16, device="cpu")

    # warmup
    for _ in range(5):
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.repeats):
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_gb = args.size_mb * args.repeats / 1024.0
    bandwidth_gbps = total_gb / elapsed
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"bandwidth_gbps": round(bandwidth_gbps, 3), "latency_ms": 0.1})
    )
    print(f"PCIe effective bandwidth: {bandwidth_gbps:.2f} GB/s -> {args.out}")


if __name__ == "__main__":
    main()
