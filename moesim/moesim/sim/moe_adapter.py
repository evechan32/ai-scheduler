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
        kv_per_token_mb: float = 0.0,
        kv_gpu_capacity_mb: float = 0.0,
        kv_host_capacity_mb: float = 0.0,
        kv_disk_capacity_mb: float = 0.0,
        disk_read_gbps: float = 2.0,
        disk_latency_ms: float = 5.0,
    ) -> None:
        self.scheduler = scheduler
        self.profiles = profiles
        self.pcie = pcie
        self.gpu = gpu or ComputeResource(concurrency=1, per_unit_ms=1.0)
        self.cpu = cpu
        self._clock = 0.0
        self._state = ScheduleState(
            profiles=profiles, resident=set(), gpu_capacity_mb=gpu_capacity_mb,
            kv_per_token_mb=kv_per_token_mb,
            kv_gpu_capacity_mb=kv_gpu_capacity_mb,
            kv_host_capacity_mb=kv_host_capacity_mb,
            kv_disk_capacity_mb=kv_disk_capacity_mb,
            disk_read_gbps=disk_read_gbps,
            disk_latency_ms=disk_latency_ms,
        )
        self._kv_host_capacity_mb = kv_host_capacity_mb
        self._kv_disk_capacity_mb = kv_disk_capacity_mb
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
        self._prune_pending_loads()
        self._account_kv(token_count)
        self._snapshot_feedback(experts)
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

        # Feed residency-benefit feedback: each booked transfer is a migration
        # whose PCIe cost the expert's residency saves on future hits.
        for action in actions:
            if action.kind in ("load", "prefetch"):
                for eid in action.expert_ids:
                    self._state.record_load(
                        eid, self.pcie.transfer_time_ms(self.profiles[eid].size_mb)
                    )

        # Book PCIe transfers. In-flight transfers (from an earlier prefetch)
        # are reused instead of re-booked, so the bus is never double-charged.
        load_times: dict[str, float] = {}
        for action in actions:
            if action.kind == "load":
                for eid in action.expert_ids:
                    load_times[eid] = self._book_transfer(eid, critical=True)
        for action in actions:
            if action.kind == "prefetch":
                for eid in action.expert_ids:
                    self._book_transfer(eid, critical=False)
        for action in actions:
            if action.kind == "prefetch_from_disk":
                for eid in action.expert_ids:
                    self._prefetch_from_disk(eid)

        completions: list[float] = []
        # GPU executions start at load completion (or now if already resident).
        for eid in list(gpu_ids) + default_gpu:
            start = max(
                self._clock,
                load_times.get(eid, self._clock),
                self._state.pending_loads.get(eid, self._clock),
            )
            completions.append(self.gpu.schedule(start, self.profiles[eid].gpu_exec_ms))
        # CPU executions run in parallel with GPU/PCIe work (separate resource).
        # Quantized variants execute faster on CPU (HOBBIT mixed precision).
        for eid in cpu_ids:
            if self.cpu is None:
                raise RuntimeError("execute_cpu requested but no CPU resource configured")
            start = self._clock
            if eid in self._state.disk_experts:
                size = self.profiles[eid].size_mb
                start += size / self._state.disk_read_gbps + self._state.disk_latency_ms
            completions.append(
                self.cpu.schedule(start, self.profiles[eid].quantized_cpu_exec_ms())
            )

        # KV evict/fetch transfers run over PCIe concurrently with expert loads.
        kv_times: list[float] = []
        for action in actions:
            if action.kind == "evict_kv":
                for eid in action.expert_ids:
                    kv_times.append(self._kv_transfer(eid))
            elif action.kind == "fetch_kv":
                for eid in action.expert_ids:
                    kv_times.append(self._kv_transfer(eid))
        # include KV transfer completions in the step's completion time
        completions.extend(kv_times)

        step_completion = max(completions, default=self._clock)
        step_time = step_completion - self._clock
        self._clock = step_completion
        # Each step is one decode step (activating multiple experts); it
        # contributes token_count (default 1) tokens.
        self._metrics.record_completion(tokens=token_count, time_ms=step_time)

    def _prune_pending_loads(self) -> None:
        self._state.pending_loads = {
            eid: completion
            for eid, completion in self._state.pending_loads.items()
            if completion > self._clock
        }

    def _account_kv(self, token_count: int) -> None:
        kv_per_token = self._state.kv_per_token_mb
        if kv_per_token <= 0.0:
            return
        self._state.kv_gpu_mb += token_count * kv_per_token

        gpu_capacity = self._state.kv_gpu_capacity_mb
        if gpu_capacity > 0.0 and self._state.kv_gpu_mb > gpu_capacity:
            excess = self._state.kv_gpu_mb - gpu_capacity
            self._state.kv_gpu_mb = gpu_capacity
            self._state.kv_host_mb += excess
            self._metrics.record_kv_offload(excess)
            completion = self.pcie.reserve(self._clock, excess)
            transfer_ms = self.pcie.transfer_time_ms(excess)
            self._metrics.record_transfer_wait(
                max(0.0, completion - transfer_ms - self._clock)
            )

        host_capacity = self._state.kv_host_capacity_mb
        if host_capacity > 0.0 and self._state.kv_host_mb > host_capacity:
            disk_excess = self._state.kv_host_mb - host_capacity
            self._state.kv_host_mb = host_capacity
            self._state.kv_disk_mb += disk_excess

        if gpu_capacity > 0.0:
            self._state.kv_pressure = self._state.kv_gpu_mb / gpu_capacity
            self._metrics.record_kv_sample("gpu", self._state.kv_gpu_mb / gpu_capacity)
            if self._kv_host_capacity_mb > 0.0:
                self._metrics.record_kv_sample(
                    "host", min(1.0, self._state.kv_host_mb / self._kv_host_capacity_mb)
                )
            if self._kv_disk_capacity_mb > 0.0:
                self._metrics.record_kv_sample(
                    "disk", min(1.0, self._state.kv_disk_mb / self._kv_disk_capacity_mb)
                )

    def _snapshot_feedback(self, experts: list[str]) -> None:
        self._state.pcie_queue_len = self.pcie.queue_depth(self._clock)
        self._state.gpu_queue_len = self.gpu.queue_depth(self._clock)
        self._state.cpu_queue_len = (
            self.cpu.queue_depth(self._clock) if self.cpu is not None else 0
        )
        self._state.pcie_utilization = self.pcie.utilization(self._clock) if self._clock > 0 else 0.0
        self._state.gpu_utilization = self.gpu.utilization(self._clock) if self._clock > 0 else 0.0
        self._state.cpu_utilization = (
            self.cpu.utilization(self._clock) if (self.cpu is not None and self._clock > 0) else 0.0
        )
        if experts:
            profile = self.profiles[experts[0]]
            self._state.gpu_wait_ms = self.gpu.wait_time_ms(self._clock, profile.gpu_exec_ms) \
                - self.gpu.process_time_ms(profile.gpu_exec_ms)
            if self.cpu is not None:
                self._state.cpu_wait_ms = self.cpu.wait_time_ms(self._clock, profile.cpu_exec_ms) \
                    - self.cpu.process_time_ms(profile.cpu_exec_ms)
            else:
                self._state.cpu_wait_ms = 0.0
            self._state.pcie_wait_ms = self.pcie.wait_time_ms(self._clock, profile.size_mb) \
                - self.pcie.transfer_time_ms(profile.size_mb)
        self._metrics.record_queue_sample("pcie", self._state.pcie_queue_len)
        self._metrics.record_queue_sample("gpu", self._state.gpu_queue_len)
        self._metrics.record_queue_sample("cpu", self._state.cpu_queue_len)
        self._metrics.record_utilization("pcie", self._state.pcie_utilization)
        self._metrics.record_utilization("gpu", self._state.gpu_utilization)
        self._metrics.record_utilization("cpu", self._state.cpu_utilization)

    def _book_transfer(self, eid: str, critical: bool) -> float:
        pending = self._state.pending_loads.get(eid)
        if pending is not None:
            return pending
        profile = self.profiles[eid]
        size_mb = profile.size_mb if critical else profile.quantized_size_mb()
        completion = self.pcie.reserve(self._clock, size_mb)
        self._state.pending_loads[eid] = completion
        transfer_ms = self.pcie.transfer_time_ms(size_mb)
        self._metrics.record_transfer(transfer_ms)
        if critical:
            self._metrics.record_transfer_wait(max(0.0, completion - transfer_ms - self._clock))
        else:
            self._metrics.record_prefetch()
            self._metrics.record_hidden_transfer(transfer_ms)
        return completion

    def _kv_transfer(self, eid: str) -> float:
        completion = self.pcie.reserve(self._clock, self.profiles[eid].size_mb)
        transfer_ms = self.pcie.transfer_time_ms(self.profiles[eid].size_mb)
        self._metrics.record_transfer(transfer_ms)
        return completion

    def _prefetch_from_disk(self, eid: str) -> None:
        """Background disk->DRAM read for a predicted expert (MoE-Infinity).

        Runs off the step critical path (overlapped with compute); once done,
        the expert is promoted out of the disk tier so its later activation no
        longer pays the slow SSD read.
        """
        size = self.profiles[eid].size_mb
        read_ms = size / self._state.disk_read_gbps + self._state.disk_latency_ms
        self._metrics.record_hidden_transfer(read_ms)
        self._metrics.record_transfer(read_ms)
        self._state.disk_experts.discard(eid)
