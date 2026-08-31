#!/usr/bin/env python3
"""vLLM CPU-offload integration: load OLMoE-1B-7B (bf16 14G) on an 8GiB GPU
via UVA offload, sweep cpu_offload_gb. Measures throughput + hardware.

This is the "real integration" smoke test: does vLLM's built-in CPU offload let
an over-capacity MoE model run, and at what throughput/memory cost?
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.microbench.resource_monitor import ResourceMonitor

OUT_DIR = Path(__file__).resolve().parent / "out"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/qyw/models/olmoe-1b-7b")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--offload-gb", default="0,4,8")
    parser.add_argument("--gpu-util", type=float, default=0.85)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    rows = []
    for gb_str in args.offload_gb.split(","):
        gb = float(gb_str)
        monitor = ResourceMonitor(interval_s=0.2)
        monitor.start()
        t_load = time.perf_counter()
        try:
            llm = LLM(model=args.model, cpu_offload_gb=gb, enforce_eager=True,
                      gpu_memory_utilization=args.gpu_util, max_model_len=512)
            load_s = time.perf_counter() - t_load
            params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                    ignore_eos=True)
            tpots = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                out = llm.generate([args.prompt], params, use_tqdm=False)
                el = time.perf_counter() - t0
                n = len(out[0].outputs[0].token_ids)
                tpots.append(el / max(1, n))
            tpot = sum(tpots) / len(tpots)
            del llm
        except Exception as e:
            load_s = time.perf_counter() - t_load
            monitor.stop()
            print(f"cpu_offload_gb={gb}: FAILED {type(e).__name__}: {str(e)[:120]}")
            continue
        monitor.stop()
        s = monitor.summary()
        rows.append({
            "cpu_offload_gb": gb,
            "load_s": round(load_s, 1),
            "tpot_ms": round(tpot * 1000.0, 2),
            "throughput_tok_s": round(1.0 / tpot, 1),
            "gpu_util_mean": round(s["gpu_util"]["mean"], 1),
            "gpu_mem_used_mib": round(s["gpu_mem_used_mib"]["max"], 0),
            "cpu_util_mean": round(s["cpu_util"]["mean"], 1),
            "sys_mem_util": round(s["sys_mem_util"]["mean"], 1),
        })

    print(f"\n=== vLLM CPU-offload integration (OLMoE-1B-7B bf16 14G, 8GiB GPU) ===")
    print(f"{'offload_gb':>10s} {'load(s)':>8s} {'TPOT(ms)':>9s} {'tok/s':>8s} "
          f"{'GPUutil':>8s} {'GPUmem':>8s} {'CPUutil':>8s} {'sysmem':>7s}")
    for r in rows:
        print(f"{r['cpu_offload_gb']:10.0f} {r['load_s']:8.1f} {r['tpot_ms']:9.2f} "
              f"{r['throughput_tok_s']:8.1f} {r['gpu_util_mean']:7.1f}% "
              f"{r['gpu_mem_used_mib']:8.0f} {r['cpu_util_mean']:7.1f}% "
              f"{r['sys_mem_util']:6.1f}%")

    out = OUT_DIR / "vllm_offload_integration.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nrecorded: {out}")


if __name__ == "__main__":
    main()
