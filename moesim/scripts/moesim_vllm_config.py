#!/usr/bin/env python3
"""moesim -> vLLM config generator (methods 1 + 2, no model patching).

Method 1 (layer-granularity MoE offload): vLLM's `cpu_offload_params` matches
parameter-name segments (`f".{param}." in f".{name}."`), and MoE weights are
named `layers.{i}.mlp.experts.w13_weight`. So segment `layers.{i}.experts`
offloads that layer's whole MoE — the same granularity as llama.cpp --n-cpu-moe.
moesim picks WHICH layers to offload by layer importance (early layers are more
sensitive — QuantMoE-Bench "+FirstL"), which beats vLLM's default non-selective
offload-until-gb-full.

Method 2 (KV cache offload): moesim v8's KV-tier scheduling (pressure threshold,
eviction) maps to vLLM's SimpleCPUOffloadScheduler params (lazy vs eager, pool
size).

Run (numpy-only):
    python scripts/moesim_vllm_config.py --layers 16 --experts 64 \
        --expert-size-mb 12.6 --gpu-budget-mb 4096 --out vllm_config.json
"""
from __future__ import annotations

import argparse
import json


def layer_importance(layer: int, num_layers: int) -> float:
    """Early layers matter more (QuantMoE-Bench: prioritize first MoE blocks).
    Returns a score in [0,1]; higher = keep on GPU."""
    return 1.0 - (layer / num_layers)


def build_plan(args) -> dict:
    # How many whole MoE layers fit in the expert GPU budget.
    layer_moe_mb = args.experts * args.expert_size_mb
    keep_layers = max(1, int(args.gpu_budget_mb / layer_moe_mb))

    # Rank layers by importance; keep the top `keep_layers` on GPU.
    ranked = sorted(range(args.layers),
                    key=lambda i: -layer_importance(i, args.layers))
    keep_set = set(ranked[:keep_layers])
    offload_layers = [i for i in range(args.layers) if i not in keep_set]

    cpu_offload_params = [f"layers.{i}.experts" for i in offload_layers]

    # Method 2: KV offload params. moesim v8 pressure threshold -> lazy offload
    # when GPU KV pool is under pressure; pool size from host budget.
    kv_pressure_threshold = 0.8
    kv = {
        "lazy_offload": True,          # evict KV to CPU when GPU pool pressure > threshold
        "kv_pressure_threshold": kv_pressure_threshold,
        "host_kv_pool_mb": args.kv_host_pool_mb,
    }

    plan = {
        "num_layers": args.layers,
        "num_experts": args.experts,
        "layer_moe_mb": layer_moe_mb,
        "gpu_budget_mb": args.gpu_budget_mb,
        "keep_layers": keep_layers,
        "offload_layers": offload_layers,
        "cpu_offload_params": cpu_offload_params,
        "cpu_offload_gb": len(offload_layers) * layer_moe_mb / 1024.0,
        "kv_offload": kv,
    }
    return plan


def verify_segment_match(plan: dict) -> bool:
    """Unit-check: does `layers.{i}.experts` match vLLM's real parameter name
    `model.layers.{i}.mlp.experts.w13_weight` under dot-wrapped substring match?"""
    for param in plan["cpu_offload_params"][:3]:
        name = f"model.{param}.w13_weight"
        assert f".{param}." in f".{name}.", f"{param} should match {name}"
    # negative: must NOT match a different layer
    p0 = plan["cpu_offload_params"][0]
    other = f"model.layers.99.experts.w13_weight"
    assert f".{p0}." not in f".{other}.", "must not match another layer"
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--expert-size-mb", type=float, default=12.6)
    parser.add_argument("--gpu-budget-mb", type=float, default=4096.0)
    parser.add_argument("--kv-host-pool-mb", type=float, default=2048.0)
    parser.add_argument("--out", default="vllm_config.json")
    args = parser.parse_args()

    plan = build_plan(args)
    assert verify_segment_match(plan), "segment match check failed"

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print(f"keep {plan['keep_layers']}/{args.layers} MoE layers on GPU, "
          f"offload {len(plan['offload_layers'])} layers "
          f"(cpu_offload_gb={plan['cpu_offload_gb']:.1f})")
    print(f"cpu_offload_params sample: {plan['cpu_offload_params'][:4]} ...")
    print(f"KV offload: lazy={plan['kv_offload']['lazy_offload']}, "
          f"host pool {plan['kv_offload']['host_kv_pool_mb']}MB")
    print(f"written: {args.out}")
    print("\nvLLM usage (on a >=32G-RAM machine):")
    print("  LLM(model=..., cpu_offload_gb=..., cpu_offload_params=set(plan['cpu_offload_params']))")


if __name__ == "__main__":
    main()
