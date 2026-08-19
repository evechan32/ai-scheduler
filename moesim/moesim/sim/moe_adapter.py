"""MoE domain adapter: drive the scheduler inside the simulator."""
from __future__ import annotations

from moesim.scheduler.base import Scheduler, apply_actions
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.state import ScheduleState
from moesim.sim.metrics import Metrics
from moesim.sim.resources import BandwidthResource, ComputeResource


class MoESimulation:
    def __init__(
        self,
        scheduler: Scheduler,
        profiles: dict[str, ExpertProfile],
        gpu_capacity_mb: float,
        pcie: BandwidthResource,
        gpu: ComputeResource | None = None,
        cpu: ComputeResource | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.profiles = profiles
        self.pcie = pcie
        self.gpu = gpu or ComputeResource(concurrency=1, per_unit_ms=1.0)
        self.cpu = cpu
        self._clock = 0.0
        self._state = ScheduleState(
            profiles=profiles, resident=set(), gpu_capacity_mb=gpu_capacity_mb
        )
        self._metrics = Metrics()

    def feed(self, step_experts: list[str], token_count: int = 1) -> None:
        """Record one decode step's requested experts and account its time."""
        self._step(step_experts, token_count)

    def run(self, steps: list[list[str]]) -> Metrics:
        for step in steps:
            self._step(step)
        self._metrics.cache_hits = self._state.cache_hits
        self._metrics.cache_misses = self._state.cache_misses
        return self._metrics

    def _step(self, experts: list[str], token_count: int = 1) -> None:
        self._state.requested = tuple(experts)
        for eid in experts:
            self._state.mark_access(eid)
        actions = self.scheduler.decide(self._state, self._clock)
        apply_actions(self._state, actions)

        # Execution model: an expert's execution can only start AFTER its PCIe
        # load completes (serial dependency). Experts with no explicit execute
        # action default to GPU execution (cache-management-only policies like
        # LRU never emit execute actions).
        cpu_ids = [eid for a in actions if a.kind == "execute_cpu" for eid in a.expert_ids]
        gpu_ids = [eid for a in actions if a.kind == "execute_gpu" for eid in a.expert_ids]
        decided = set(cpu_ids) | set(gpu_ids)
        default_gpu = [eid for eid in experts if eid not in decided]

        # Feed queue-length feedback back to the scheduler: how many experts
        # execute on GPU vs CPU in this step.
        self._state.gpu_queue_len = len(gpu_ids) + len(default_gpu)
        self._state.cpu_queue_len = len(cpu_ids)

        # Feed residency-benefit feedback: each load is a migration whose PCIe
        # cost the expert's residency saves on future hits.
        for action in actions:
            if action.kind == "load":
                for eid in action.expert_ids:
                    self._state.record_load(
                        eid, self.pcie.transfer_time_ms(self.profiles[eid].size_mb)
                    )

        # Schedule PCIe loads first; remember each expert's load completion time.
        load_times: dict[str, float] = {}
        for action in actions:
            if action.kind == "load":
                for eid in action.expert_ids:
                    load_times[eid] = self.pcie.reserve(self._clock, self.profiles[eid].size_mb)

        completions: list[float] = []
        # GPU executions start at load completion (or now if already resident).
        for eid in list(gpu_ids) + default_gpu:
            start = load_times.get(eid, self._clock)
            completions.append(self.gpu.schedule(start, self.profiles[eid].gpu_exec_ms))
        # CPU executions run in parallel with GPU/PCIe work (separate resource).
        for eid in cpu_ids:
            if self.cpu is None:
                raise RuntimeError("execute_cpu requested but no CPU resource configured")
            completions.append(self.cpu.schedule(self._clock, self.profiles[eid].cpu_exec_ms))

        # KV evict/fetch transfers run over PCIe concurrently with expert loads.
        kv_times: list[float] = []
        for action in actions:
            if action.kind == "evict_kv":
                for eid in action.expert_ids:
                    kv_times.append(self.pcie.reserve(self._clock, self.profiles[eid].size_mb))
            elif action.kind == "fetch_kv":
                for eid in action.expert_ids:
                    kv_times.append(self.pcie.reserve(self._clock, self.profiles[eid].size_mb))
        # include KV transfer completions in the step's completion time
        completions.extend(kv_times)

        step_completion = max(completions, default=self._clock)
        step_time = step_completion - self._clock
        self._clock = step_completion
        # Each step is one decode step (activating multiple experts); it
        # contributes token_count (default 1) tokens.
        self._metrics.record_completion(tokens=token_count, time_ms=step_time)
