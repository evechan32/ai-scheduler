#!/usr/bin/env python3
"""Step 1: generate the expert offload plan from moesim.

Outputs offload_plan.json = the set of cold experts (layer + expert index) that
moesim's scheduling decides to keep on CPU, given a GPU memory budget and the
expert activation-frequency distribution.

Run (numpy-only, no GPU needed):
    python scripts/moesim_vllm_offload_plan.py --experts 64 --layers 16 \
        --expert-size-mb 12.6 --gpu-budget-mb 4096 --out offload_plan.json
"""
from __future__ import annotations

import argparse
import json
import math

# MoE expert activation is skewed (few hot, many cold) — MoE-Infinity /
# PowerInfer consensus. We model frequency as Zipf; real frequency must be
# measured from live inference (router trace) and can be substituted here.
DEFAULT_HOT_EXPERTS = 8  # top-8 per layer dominate activation


def zipf_freq(rank: int, n: int, s: float = 1.0) -> float:
    """Zipf frequency for expert ranked `rank` (0 = hottest)."""
    norm = sum(1.0 / (i ** s) for i in range(1, n + 1))
    return (1.0 / ((rank + 1) ** s)) / norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--expert-size-mb", type=float, default=12.6,
                        help="bf16 expert size (3*1024*2048*2 bytes for OLMoE)")
    parser.add_argument("--gpu-budget-mb", type=float, default=4096.0,
                        help="GPU memory reserved for expert weights")
    parser.add_argument("--zipf-s", type=float, default=1.0)
    parser.add_argument("--out", default="offload_plan.json")
    args = parser.parse_args()

    total_experts = args.layers * args.experts
    keep_capacity = int(args.gpu_budget_mb / args.expert_size_mb)
    keep_per_layer = max(1, keep_capacity // args.layers)

    # Rank experts per layer by frequency; keep top `keep_per_layer` on GPU.
    freqs = [zipf_freq(r, args.experts, args.zipf_s) for r in range(args.experts)]
    ranked = sorted(range(args.experts), key=lambda j: -freqs[j])
    hot = set(ranked[:keep_per_layer])

    cold_experts = []
    for layer in range(args.layers):
        for expert in range(args.experts):
            if expert not in hot:
                cold_experts.append(f"layers.{layer}.experts.{expert}")

    plan = {
        "num_layers": args.layers,
        "num_experts": args.experts,
        "expert_size_mb": args.expert_size_mb,
        "gpu_budget_mb": args.gpu_budget_mb,
        "keep_per_layer": keep_per_layer,
        "total_experts": total_experts,
        "cold_expert_count": len(cold_experts),
        "cold_experts": cold_experts,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print(f"total {total_experts} experts, keep {keep_capacity} on GPU "
          f"({keep_per_layer}/layer), offload {len(cold_experts)} cold experts")
    print(f"written: {args.out}")
    print(f"sample cpu_offload_params: {cold_experts[:5]} ...")


if __name__ == "__main__":
    main()
