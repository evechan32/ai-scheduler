#!/usr/bin/env python3
"""v9: request concurrency comparison — queuing impact on TTFT/JCT.

Same request set under different GPU concurrency (1 vs 4 vs 8): shows how
prefill queuing and decode contention grow TTFT/JCT — Kairos's point that
queuing dominates latency (77-98% of P95 TTFT).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.overlap import OverlapAwarePolicy
from moesim.sim.request_sim import Request, RequestSimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def build_requests(n=8, arrival_interval=2.0, prompt=64, output=32):
    return [Request(i, float(i) * arrival_interval, prompt, output) for i in range(n)]


def profiles(n=8, size_mb=24.0, gpu_exec=1.0, cpu_exec=6.0):
    return {f"e{i}": ExpertProfile(f"e{i}", size_mb, gpu_exec, cpu_exec,
                                   activation_freq=0.9 if i < 2 else 0.1)
            for i in range(n)}


def run_concurrency(concurrency, requests, prof, pcie_params, kv_capacity_mb):
    pcie = BandwidthResource(**pcie_params)
    sim = RequestSimulation(
        scheduler=OverlapAwarePolicy(pcie=pcie, prefetch_n=2),
        profiles=prof, gpu_capacity_mb=12 * 1024.0, pcie=pcie,
        gpu=ComputeResource(concurrency=concurrency, per_unit_ms=1.0),
        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
        prefill_per_token_ms=0.5, experts_per_token=2,
        kv_per_token_mb=10.0, kv_gpu_capacity_mb=kv_capacity_mb,
        kv_host_capacity_mb=16 * 1024.0,
    )
    return sim.run(requests)


def main() -> None:
    reqs = build_requests()
    prof = profiles()
    pcie_params = {"bandwidth_gbps": 8.0}

    print("=== v9 request concurrency (8 requests, prompt 64, output 32, "
          "arrival every 2ms) ===")
    print(f"{'gpu_slots':>9s} {'TTFT_avg':>9s} {'TTFT_p95':>9s} "
          f"{'TPOT_avg':>9s} {'JCT_avg':>9s} {'prefill_q%':>10s} {'tput':>7s}")
    for concurrency in (1, 4, 8):
        stats = run_concurrency(concurrency, reqs, prof, pcie_params, 4 * 1024.0)
        ttfts = sorted(s.ttft_ms for s in stats)
        ttft_avg = sum(ttfts) / len(ttfts)
        ttft_p95 = ttfts[min(len(ttfts) - 1, int(0.95 * len(ttfts)))]
        tpot_avg = sum(s.tpot_avg_ms for s in stats) / len(stats)
        jct_avg = sum(s.jct_ms for s in stats) / len(stats)
        q_share = sum(s.prefill_queuing_ms for s in stats) / sum(s.ttft_ms for s in stats)
        total_tokens = sum(r.output_tokens for r in reqs)
        wall = max(s.jct_ms for s in stats)
        tput = total_tokens / wall * 1000.0
        print(f"{concurrency:9d} {ttft_avg:9.2f} {ttft_p95:9.2f} "
              f"{tpot_avg:9.2f} {jct_avg:9.2f} {q_share * 100:9.1f}% {tput:7.1f}")

    print("\nread: lower concurrency => more prefill queuing (larger prefill_q%) "
          "and higher TTFT/JCT; higher concurrency trades per-token latency for "
          "throughput. Queuing share of TTFT grows as GPU saturates (Kairos).")


if __name__ == "__main__":
    main()
