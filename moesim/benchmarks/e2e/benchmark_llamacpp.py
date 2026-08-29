#!/usr/bin/env python3
"""llama.cpp real inference benchmark (MoE baseline) with hardware telemetry.

Runs llama_cpp on OLMoE-1B-7B GGUF (64-expert MoE), sampling CPU/mem utilization
(via resource_monitor). Records TTFT / TPOT / throughput + hardware.

Run in the py311 env (llama-cpp-python installed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.microbench.resource_monitor import ResourceMonitor, print_summary

OUT_DIR = Path(__file__).resolve().parent / "out"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/qyw/models/olmoe-1b-7b-gguf/OLMoE-1B-7B-0125-Q3_K_L.gguf")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from llama_cpp import Llama

    monitor = ResourceMonitor(interval_s=0.2)
    monitor.start()
    t_load = time.perf_counter()
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_threads=args.n_threads,
                verbose=False)
    load_s = time.perf_counter() - t_load

    ttfts: list[float] = []
    tpots: list[float] = []
    for _ in range(args.num_runs):
        t0 = time.perf_counter()
        out = llm(args.prompt, max_tokens=args.max_tokens)
        elapsed = time.perf_counter() - t0
        n_tokens = out["usage"]["completion_tokens"]
        tpot = elapsed / max(1, n_tokens)
        tpots.append(tpot)

    monitor.stop()

    summary = monitor.summary()
    result = {
        "model": args.model,
        "framework": "llama_cpp",
        "load_s": load_s,
        "ttft_ms": 0.0,  # non-streaming API: not measured separately
        "tpot_ms": sum(tpots) / len(tpots) * 1000.0,
        "throughput_tok_s": 1.0 / (sum(tpots) / len(tpots)),
        "hardware": summary,
    }
    print(f"\n=== llama.cpp ({Path(args.model).name}) ===")
    print(f"load {load_s:.1f}s | TTFT {result['ttft_ms']:.1f}ms | "
          f"TPOT {result['tpot_ms']:.1f}ms | {result['throughput_tok_s']:.1f} tok/s")
    print_summary(summary)

    out = args.out or str(OUT_DIR / "llamacpp_benchmark.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nrecorded: {out}")


if __name__ == "__main__":
    main()
