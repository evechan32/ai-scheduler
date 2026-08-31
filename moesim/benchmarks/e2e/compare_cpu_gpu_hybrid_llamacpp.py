#!/usr/bin/env python3
"""Same-model, same-input framework comparison: pure CPU / pure GPU / hybrid.

llama.cpp (CUDA build) on OLMoE-1B-7B GGUF, sweeping n_gpu_layers:
  - n_gpu_layers=0   -> pure CPU
  - n_gpu_layers=99  -> pure GPU (3.6GB Q3_K_L fits 8GiB)
  - n_gpu_layers=N   -> layer-granularity hybrid (first N layers GPU, rest CPU)
Records throughput + GPU/CPU utilization per config.

Run in vllm-build env (CUDA llama-cpp-python 0.3.35).
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


def bench_one(llm, prompt, max_tokens, runs):
    tpots = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = llm(prompt, max_tokens=max_tokens)
        elapsed = time.perf_counter() - t0
        n = out["usage"]["completion_tokens"]
        tpots.append(elapsed / max(1, n))
    return sum(tpots) / len(tpots)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/qyw/models/olmoe-1b-7b-gguf/OLMoE-1B-7B-0125-Q3_K_L.gguf")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--layers", default="0,8,16,99")
    args = parser.parse_args()

    from llama_cpp import Llama

    rows = []
    for layer_str in args.layers.split(","):
        n_gpu = int(layer_str)
        monitor = ResourceMonitor(interval_s=0.2)
        monitor.start()
        t_load = time.perf_counter()
        llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_gpu_layers=n_gpu,
                    verbose=False)
        load_s = time.perf_counter() - t_load
        tpot = bench_one(llm, args.prompt, args.max_tokens, args.runs)
        monitor.stop()
        s = monitor.summary()
        rows.append({
            "n_gpu_layers": n_gpu,
            "load_s": round(load_s, 1),
            "tpot_ms": round(tpot * 1000.0, 2),
            "throughput_tok_s": round(1.0 / tpot, 1),
            "gpu_util_mean": round(s["gpu_util"]["mean"], 1),
            "gpu_util_max": round(s["gpu_util"]["max"], 1),
            "gpu_mem_used_mib": round(s["gpu_mem_used_mib"]["max"], 0),
            "cpu_util_mean": round(s["cpu_util"]["mean"], 1),
        })

    print(f"=== llama.cpp (CUDA) on OLMoE-1B-7B Q3_K_L — same prompt ===")
    print(f"{'n_gpu_layers':>12s} {'load(s)':>8s} {'TPOT(ms)':>9s} {'tok/s':>8s} "
          f"{'GPUutil':>8s} {'GPUmax':>7s} {'GPUmem':>8s} {'CPUutil':>8s}")
    for r in rows:
        label = {0: "pure CPU", 99: "pure GPU"}.get(r["n_gpu_layers"], "hybrid")
        print(f"{label:>12s} {r['load_s']:8.1f} {r['tpot_ms']:9.2f} "
              f"{r['throughput_tok_s']:8.1f} {r['gpu_util_mean']:7.1f}% "
              f"{r['gpu_util_max']:6.0f}% {r['gpu_mem_used_mib']:8.0f} "
              f"{r['cpu_util_mean']:7.1f}%")

    out = OUT_DIR / "llamacpp_cpu_gpu_hybrid.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nrecorded: {out}")


if __name__ == "__main__":
    main()
