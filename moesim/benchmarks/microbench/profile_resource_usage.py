#!/usr/bin/env python3
"""Profile real-machine resource usage under compute- and bandwidth-bound loads.

Runs two torch workloads while a background monitor samples GPU util, GPU mem
util, CPU util and system mem util; optionally wraps in NCU for precise DRAM
bandwidth utilization and SM active throughput. Records everything to
`benchmarks/microbench/out/resource_usage.json`.

Usage (in a torch/CUDA environment, e.g. vllm-build):
  python benchmarks/microbench/profile_resource_usage.py            # smi sampling
  python benchmarks/microbench/profile_resource_usage.py --with-ncu # + NCU metrics
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.microbench.resource_monitor import (
    DmonSampler,
    ResourceMonitor,
    print_summary,
    run_ncu,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_FILE = OUT_DIR / "resource_usage.json"


def compute_bound_workload(iterations: int = 60) -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    start = time.perf_counter()
    for _ in range(iterations):
        a = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    flops = 2 * 4096**3 * iterations
    print(f"compute-bound matmul: {iterations}x (4096)^3  -> {flops / elapsed / 1e12:.2f} TFLOPS")
    return elapsed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ncu", action="store_true",
                        help="wrap in NSight Compute for DRAM bw / SM active metrics")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--theoretical-bw-gbs", type=float, default=448.0,
                        help="GPU theoretical DRAM bandwidth (RTX 5070 Laptop GDDR7 ~448)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(interval_s=args.interval)
    dmon = DmonSampler()
    monitor.start()
    dmon.start()
    compute_bound_workload()
    bw_gbps = measure_bandwidth()
    monitor.stop()
    dmon.stop()

    ncu = run_ncu(["python", str(Path(__file__).resolve())]) if args.with_ncu else None

    summary = monitor.summary()
    dmon_summary = dmon.summary()
    bw_util_pct = 100.0 * bw_gbps / args.theoretical_bw_gbs if bw_gbps else 0.0

    print(f"\n=== resource usage (sampled {len(monitor.samples)}x smi + {len(dmon.samples)}x dmon) ===")
    print_summary(summary, ncu)
    print(f"\ndriver-level (1s granularity):")
    for key, agg in dmon_summary.items():
        label = "sm_active proxy" if key == "dmon_sm" else "DRAM bw-util proxy"
        print(f"{label:28s} {agg['mean']:8.1f} {agg['max']:8.1f} {agg['p95']:8.1f}")
    print(f"\nmeasured DRAM bandwidth: {bw_gbps:.0f} GB/s (r+w) "
          f"= {bw_util_pct:.1f}% of theoretical {args.theoretical_bw_gbs:.0f} GB/s")

    payload = {
        "gpu": "RTX 5070 Laptop (8GiB)",
        "theoretical_bw_gbs": args.theoretical_bw_gbs,
        "measured_bw_gbs": bw_gbps,
        "measured_bw_util_pct": bw_util_pct,
        "sampled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples_smi": len(monitor.samples),
        "n_samples_dmon": len(dmon.samples),
        "ncu": ncu or {},
        "summary": summary,
        "dmon_summary": dmon_summary,
        "samples": monitor.samples,
        "dmon_samples": dmon.samples,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nrecorded: {OUT_FILE}")


def measure_bandwidth(gib: float = 1.0, rounds: int = 5) -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    n = int(gib * 1024**3 / 4)
    src = torch.randn(n, device="cuda")
    dst = torch.empty_like(src)
    torch.cuda.synchronize()
    dst.copy_(src)
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        dst.copy_(src)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return 2 * n * 4 / min(times) / 1e9


if __name__ == "__main__":
    main()
