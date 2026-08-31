#!/usr/bin/env python3
"""Step 3: load the patched model in vLLM with moesim's offload plan.

Combines Step 1 (offload_plan.json) + Step 2 (per-expert model) into a vLLM
run: pass cpu_offload_params with the cold experts, and compare against
vLLM's default non-selective offload.

Requires >=32G host RAM (14G bf16 + 8G offload + vLLM runtime). The 7.6G dev
box OOMs on this (documented), so this runs on a bigger machine.

    python scripts/moesim_vllm_run.py --model <patched-model> \
        --plan offload_plan.json
"""
from __future__ import annotations

import argparse
import json
import time


def build_cpu_offload_params(plan: dict) -> set[str]:
    """offload_plan cold_experts -> cpu_offload_params segments.

    vLLM matches `f".{param}." in f".{name}."`; our per-expert names are
    `...experts.{j}.w1.weight`, so segment `layers.{i}.experts.{j}` matches
    only that expert (not w1_weight_scale, thanks to dot-wrapping).
    """
    return set(plan["cold_experts"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="patched model path")
    parser.add_argument("--plan", default="offload_plan.json")
    parser.add_argument("--cpu-offload-gb", type=float, default=8.0)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    with open(args.plan, encoding="utf-8") as f:
        plan = json.load(f)
    cpu_offload_params = build_cpu_offload_params(plan)

    from vllm import LLM, SamplingParams

    print(f"loading {args.model} with {len(cpu_offload_params)} cold experts "
          f"offloaded (cpu_offload_gb={args.cpu_offload_gb})")

    t0 = time.perf_counter()
    llm = LLM(model=args.model, cpu_offload_gb=args.cpu_offload_gb,
              cpu_offload_params=cpu_offload_params, enforce_eager=True,
              gpu_memory_utilization=0.85, max_model_len=512)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                            ignore_eos=True)
    t0 = time.perf_counter()
    out = llm.generate([args.prompt], params, use_tqdm=False)
    elapsed = time.perf_counter() - t0
    n = len(out[0].outputs[0].token_ids)
    print(f"TPOT {elapsed / max(1, n) * 1000:.2f}ms  "
          f"throughput {n / elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()
