"""Strategy comparison harness — the project's headline benchmark."""
from __future__ import annotations

from typing import Callable

from moesim.scheduler.base import Scheduler
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.activation_freq import ActivationFreqPolicy
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.metrics import Metrics
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def sweep(
    scheduler: Scheduler,
    profiles: dict[str, ExpertProfile],
    steps: list[list[str]],
    pcie_params: dict,
    gpu_capacity_mb: float,
) -> Metrics:
    pcie = BandwidthResource(**pcie_params)
    sim = MoESimulation(
        scheduler=scheduler,
        profiles=profiles,
        gpu_capacity_mb=gpu_capacity_mb,
        pcie=pcie,
        gpu=ComputeResource(concurrency=1, per_unit_ms=1.0),
        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
    )
    return sim.run(steps)


def compare_policies(
    profiles: dict[str, ExpertProfile],
    steps: list[list[str]],
    pcie_params: dict,
    gpu_capacity_mb: float,
) -> dict[str, Metrics]:
    pcie = BandwidthResource(**pcie_params)
    policies: dict[str, Scheduler] = {
        "lru": LRUPolicy(),
        "activation_freq": ActivationFreqPolicy(),
        "cost_model": CostModelPolicy(pcie=pcie, prefetch_n=1),
    }
    return {
        name: sweep(policy, profiles, steps, pcie_params, gpu_capacity_mb)
        for name, policy in policies.items()
    }
