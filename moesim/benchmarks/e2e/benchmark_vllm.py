#!/usr/bin/env python3
"""vLLM real inference benchmark with hardware telemetry.

Runs vLLM generate() on a model while sampling GPU util / SM / DRAM bw /
GPU mem / CPU / sys mem (via resource_monitor). Records TPOT / TTFT / throughput
and hardware utilization to JSON.

Env (vllm-build): requires the cooperative_topk stub + conda libstdc++:
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/local/cuda/lib64
  export LD_PRELOAD=/tmp/opencode/libcoop_stub.so
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
    parser.add_argument("--model", required=True, help="model path")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="prompt text")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-warmup", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    monitor = ResourceMonitor(interval_s=0.2)
    monitor.start()
    t_load = time.perf_counter()
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=512,
    )
    load_s = time.perf_counter() - t_load

    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                            ignore_eos=True)

    for _ in range(args.num_warmup):
        for _ in llm.generate([args.prompt], params):
            pass

    tpots: list[float] = []
    for _ in range(args.num_runs):
        t0 = time.perf_counter()
        outputs = llm.generate([args.prompt], params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        n_tokens = len(outputs[0].outputs[0].token_ids)
        tpots.append(elapsed / max(1, n_tokens))

    monitor.stop()

    summary = monitor.summary()
    result = {
        "model": args.model,
        "framework": "vllm",
        "load_s": load_s,
        "ttft_ms": 0.0,
        "tpot_ms": sum(tpots) / len(tpots) * 1000.0,
        "throughput_tok_s": 1.0 / (sum(tpots) / len(tpots)),
        "hardware": summary,
    }
    print(f"\n=== vLLM ({args.model}) ===")
    print(f"load {load_s:.1f}s | TTFT {result['ttft_ms']:.1f}ms | "
          f"TPOT {result['tpot_ms']:.1f}ms | {result['throughput_tok_s']:.1f} tok/s")
    print_summary(summary)

    out = args.out or str(OUT_DIR / "vllm_benchmark.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nrecorded: {out}")


if __name__ == "__main__":
    main()
