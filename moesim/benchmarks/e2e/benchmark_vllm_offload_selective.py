#!/usr/bin/env python3
"""2.1 local verification: does vLLM cpu_offload_params selectively offload?

Loads Qwen3.5-2B (4.5G, fits 8GiB) and verifies that specifying parameter
segments via cpu_offload_params really moves those weights off-GPU (VRAM drops)
— proving the moesim->vLLM offload link works end-to-end. OLMoE 14G can't load
on this box (7.6G RAM), so mechanism is verified on the small model.

Run (vllm-build env, see vllm-runtime-environment-fixes.md for env vars).
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


def run_case(args, offload_params, offload_group_size, offload_num_in_group, label):
    from vllm import LLM, SamplingParams

    monitor = ResourceMonitor(interval_s=0.2)
    monitor.start()
    t0 = time.perf_counter()
    llm = LLM(model=args.model,
              offload_params=offload_params,
              offload_group_size=offload_group_size,
              offload_num_in_group=offload_num_in_group,
              enforce_eager=True,
              gpu_memory_utilization=0.85, max_model_len=512)
    load_s = time.perf_counter() - t0

    params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    tpots = []
    for _ in range(2):
        t0 = time.perf_counter()
        out = llm.generate([args.prompt], params, use_tqdm=False)
        el = time.perf_counter() - t0
        n = len(out[0].outputs[0].token_ids)
        tpots.append(el / max(1, n))
    monitor.stop()
    s = monitor.summary()

    result = {
        "case": label,
        "offload_params": list(offload_params),
        "offload_group_size": offload_group_size,
        "load_s": round(load_s, 1),
        "tpot_ms": round(sum(tpots) / len(tpots) * 1000.0, 2),
        "gpu_mem_used_mib": round(s["gpu_mem_used_mib"]["max"], 0),
        "gpu_util_mean": round(s["gpu_util"]["mean"], 1),
    }
    print(f"{label:28s} load={result['load_s']:5.1f}s "
          f"TPOT={result['tpot_ms']:6.2f}ms GPUmem={result['gpu_mem_used_mib']:5.0f}MiB "
          f"GPUutil={result['gpu_util_mean']:4.1f}%")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/qyw/models/Qwen3.5-2B")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--num-offload-layers", type=int, default=12)
    args = parser.parse_args()

    rows = []
    # Case A: full GPU baseline.
    rows.append(run_case(args, set(), 0, 1, "full GPU (baseline)"))

    # Case B: prefetch offload — group all 24 layers, offload last N layers' mlp.
    rows.append(run_case(args, {"mlp"}, 24, args.num_offload_layers,
                         f"offload last {args.num_offload_layers} layers mlp"))

    out = OUT_DIR / "vllm_offload_selective.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nrecorded: {out}")


if __name__ == "__main__":
    main()
