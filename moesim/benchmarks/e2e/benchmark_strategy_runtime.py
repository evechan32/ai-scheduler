#!/usr/bin/env python3
"""2.3 verification: scheduler decision value in REAL inference (not simulation).

Drives a multi-layer MoE model through MoEForwardHook with different scheduling
policies (all-GPU, all-CPU, cost_model, DiskTierPolicy) and measures the real
forward time on CUDA. Validates that moesim's placement decisions change
execution — mixed placement beats all-CPU (and all-GPU on memory-bound models).

Uses a w1/w2 expert structure the hook + CPU kernel support (the real
transformers OlmoeExperts layout changed in transformers 5.x, out of scope here).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from moesim.executor.backends.forward_hook import MoEForwardHook
from moesim.executor.backends.transformers import TransformersMoEExecutor
from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.policies.disk_tier import DiskTierPolicy
from moesim.sim.resources import BandwidthResource


class AllGpuPolicy(Scheduler):
    def decide(self, state, clock):
        return [Action(kind="execute_gpu", expert_ids=(eid,)) for eid in state.requested]


class AllCpuPolicy(Scheduler):
    def decide(self, state, clock):
        return [Action(kind="execute_cpu", expert_ids=(eid,)) for eid in state.requested]


class Expert(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.gelu(self.w1(x)))


class SimpleMoE(nn.Module):
    def __init__(self, hidden, intermediate, num_experts):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(hidden, intermediate)
                                      for _ in range(num_experts)])

    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=-1)
        out = torch.zeros_like(x)
        for i, e in enumerate(self.experts):
            out += w[:, i:i + 1] * e(x)
        return out


class MultiLayerMoE(nn.Module):
    def __init__(self, num_layers, hidden, intermediate, num_experts):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(num_layers)])
        for layer in self.model.layers:
            layer.mlp = SimpleMoE(hidden, intermediate, num_experts)


def bench(scheduler, profiles, pcie, device, hidden, num_runs=10):
    torch.manual_seed(0)
    model = MultiLayerMoE(4, hidden, 1024, 8).to(device=device, dtype=torch.float16)
    executor = TransformersMoEExecutor(model, device=device)
    hook = MoEForwardHook(executor=executor, scheduler=scheduler, profiles=profiles,
                          pcie=pcie, device=device)
    hook.install(model)
    model.eval()
    x = torch.randn(2, 32, hidden, dtype=torch.float16, device=device)

    def step(x):
        for layer in model.model.layers:
            out = layer.mlp(x)
            x = out[0] if isinstance(out, tuple) else out
        return x

    with torch.no_grad():
        for _ in range(3):
            x = step(x)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            x = torch.randn(2, 32, hidden, dtype=torch.float16, device=device)
            x = step(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / num_runs * 1000.0
    hook.uninstall(model)
    return elapsed


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = 512
    profiles = {f"{i}": ExpertProfile(f"{i}", size_mb=2.0, gpu_exec_ms=0.1,
                                      cpu_exec_ms=0.6, activation_freq=0.9 if i < 2 else 0.05)
                for i in range(8)}
    pcie = BandwidthResource(bandwidth_gbps=4.3)

    policies = {
        "all-GPU": AllGpuPolicy(),
        "all-CPU": AllCpuPolicy(),
        "cost_model": CostModelPolicy(pcie=pcie, prefetch_n=0),
        "disk_tier": DiskTierPolicy(pcie=pcie, prefetch_n=0, disk_budget_mb=8.0),
    }

    print(f"=== 2.3 scheduler value in real inference (device={device}) ===")
    print(f"{'policy':14s} {'ms/forward':>11s}")
    results = {}
    for name, policy in policies.items():
        ms = bench(policy, profiles, pcie, device, hidden)
        results[name] = ms
        print(f"{name:14s} {ms:11.3f}")

    cpu, gpu = results["all-CPU"], results["all-GPU"]
    mixed = min(results["cost_model"], results["disk_tier"])
    print(f"\nread: mixed ({mixed:.3f}ms) vs all-CPU ({cpu:.3f}ms) vs all-GPU "
          f"({gpu:.3f}ms). Mixed beats all-CPU when CPU is bottleneck; all-GPU "
          f"wins only when the model fits VRAM — the exact tradeoff moesim's "
          f"decide() navigates per-expert.")


if __name__ == "__main__":
    main()
