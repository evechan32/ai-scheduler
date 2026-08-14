#!/usr/bin/env python3
"""Compare moesim policies against llama.cpp --n-cpu-moe on Qwen3-30B-A3B."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# Allow running as a plain script (python benchmarks/e2e/...py) without install:
# the script's own directory is sys.path[0], not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moesim.sim.calibrate import load_profiles
from moesim.scheduler.cost_model import profiles_from_dicts
from moesim.sim.sweep import compare_policies


def build_trace(profiles, num_steps=100, hot=None, cold=None):
    if hot is None:
        hot = sorted(profiles)[:2]
    if cold is None:
        cold = sorted(profiles)[-2:]
    trace = []
    for i in range(num_steps):
        if i % 5 == 4:
            trace.append(list(hot) + list(cold))
        else:
            trace.append(list(hot))
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=str,
                        default="benchmarks/microbench/out/profiles.json")
    parser.add_argument("--pcie", type=str,
                        default="benchmarks/microbench/out/pcie.json")
    parser.add_argument("--llama-cpp-bin", type=str, default="llama-bench")
    parser.add_argument("--model-gguf", type=str, default="")
    args = parser.parse_args()

    profiles = profiles_from_dicts(load_profiles(args.profiles))
    pcie_params = json.loads(Path(args.pcie).read_text())
    trace = build_trace(profiles)

    print("=== moesim simulation (calibrated) ===")
    results = compare_policies(profiles=profiles, steps=trace,
                               pcie_params=pcie_params, gpu_capacity_mb=12000.0)
    for name, m in results.items():
        print(f"{name:16s} TPOT={m.tpot_ms():8.3f}ms  "
              f"tput={m.throughput_tok_s():8.3f} tok/s  hit={m.hit_rate():.3f}")

    if args.llama_cpp_bin and args.model_gguf:
        print("\n=== llama.cpp --n-cpu-moe baseline ===")
        out = subprocess.run(
            [args.llama_cpp_bin, "-m", args.model_gguf, "--n-cpu-moe", "8", "--cpu-moe"],
            capture_output=True, text=True,
        )
        print(out.stdout[-2000:])


if __name__ == "__main__":
    main()
