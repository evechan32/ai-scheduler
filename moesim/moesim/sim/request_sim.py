"""v9: request-level concurrent simulation.

Models a set of requests sharing GPU/CPU/PCIe resources:
- prefill: GPU compute block (prompt_tokens x prefill_per_token_ms), FIFO queued
- decode: per-token steps round-robin across active requests (continuous
  batching), sharing the v8 resource/state machinery
- latency breakdown per request: prefill queuing / prefill exec / decode /
  total (TTFT, JCT) — DistServe-style five-stage decomposition (simplified).

Deterministic: request trace and resource competition are pure arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from moesim.scheduler.base import Scheduler, apply_actions
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.state import ScheduleState
from moesim.sim.metrics import Metrics
from moesim.sim.resources import BandwidthResource, ComputeResource


@dataclass
class Request:
    req_id: int
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int


@dataclass
class RequestStats:
    req_id: int
    ttft_ms: float
    prefill_queuing_ms: float
    prefill_exec_ms: float
    tpot_avg_ms: float
    jct_ms: float
    kv_offload_mb: float = 0.0


class RequestSimulation:
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
        prefill_per_token_ms: float = 0.5,
        experts_per_token: int = 2,
        expert_trace: callable | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.profiles = profiles
        self.pcie = pcie
        self.gpu = gpu or ComputeResource(concurrency=1, per_unit_ms=1.0)
        self.cpu = cpu
        self.prefill_per_token_ms = prefill_per_token_ms
        self.experts_per_token = experts_per_token
        self.expert_trace = expert_trace
        self._clock = 0.0
        self._state = ScheduleState(
            profiles=profiles, resident=set(), gpu_capacity_mb=gpu_capacity_mb,
            kv_per_token_mb=kv_per_token_mb,
            kv_gpu_capacity_mb=kv_gpu_capacity_mb,
        )
        self._kv_host_capacity_mb = kv_host_capacity_mb
        self._metrics = Metrics()
        self._req_kv_gpu: dict[int, float] = {}
        self._req_kv_host: dict[int, float] = {}

    def _experts_for(self, req: Request, token_idx: int) -> list[str]:
        if self.expert_trace is not None:
            return list(self.expert_trace(req.req_id, token_idx))
        ids = sorted(self.profiles)
        k = min(self.experts_per_token, len(ids))
        return [ids[(token_idx + i) % len(ids)] for i in range(k)]

    def run(self, requests: list[Request]) -> list[RequestStats]:
        stats: list[RequestStats] = []
        prefill_done: list[tuple[float, Request]] = []
        for req in sorted(requests, key=lambda r: r.arrival_ms):
            start = max(self._clock, req.arrival_ms)
            completion = self.gpu.schedule(
                start, req.prompt_tokens * self.prefill_per_token_ms
            )
            queuing = max(0.0, start - req.arrival_ms)
            prefill_exec = completion - start
            self._account_kv(req.prompt_tokens, req.req_id)
            prefill_done.append((completion, req))
            stats.append(RequestStats(
                req_id=req.req_id, ttft_ms=completion - req.arrival_ms,
                prefill_queuing_ms=queuing, prefill_exec_ms=prefill_exec,
                tpot_avg_ms=0.0, jct_ms=0.0,
            ))
            self._clock = completion

        tokens_done: dict[int, int] = {}
        active = list(requests)
        decode_start: dict[int, float] = {r.req_id: self._clock for r in requests}
        while active:
            round_completions: list[float] = []
            for req in active:
                idx = tokens_done.get(req.req_id, 0)
                experts = self._experts_for(req, idx)
                completion = self._decode_step(experts, self._clock, req.req_id)
                round_completions.append(completion)
                tokens_done[req.req_id] = idx + 1
            self._clock = max(round_completions)
            for req in list(active):
                if tokens_done.get(req.req_id, 0) >= req.output_tokens:
                    self._release_kv(req.req_id)
                    active.remove(req)
        self._metrics.cache_hits = self._state.cache_hits
        self._metrics.cache_misses = self._state.cache_misses

        for s in stats:
            req = next(r for r in requests if r.req_id == s.req_id)
            decode_elapsed = self._clock - decode_start[req.req_id]
            s.tpot_avg_ms = decode_elapsed / req.output_tokens
            s.jct_ms = s.ttft_ms + decode_elapsed
            s.kv_offload_mb = self._metrics.kv_offload_bytes
        return stats

    def _decode_step(self, experts: list[str], clock: float, req_id: int | None = None) -> float:
        state = self._state
        state.requested = tuple(experts)
        for eid in experts:
            state.mark_access(eid)
        self._prune_pending_loads()
        self._account_kv(1, req_id)
        self._snapshot_feedback(experts)
        actions = self.scheduler.decide(state, clock)
        apply_actions(state, actions)

        cpu_ids = [eid for a in actions if a.kind == "execute_cpu" for eid in a.expert_ids]
        gpu_ids = [eid for a in actions if a.kind == "execute_gpu" for eid in a.expert_ids]
        decided = set(cpu_ids) | set(gpu_ids)
        default_gpu = [eid for eid in experts if eid not in decided]

        for action in actions:
            if action.kind in ("load", "prefetch"):
                for eid in action.expert_ids:
                    state.record_load(
                        eid, self.pcie.transfer_time_ms(self.profiles[eid].size_mb)
                    )

        load_times: dict[str, float] = {}
        for action in actions:
            if action.kind == "load":
                for eid in action.expert_ids:
                    load_times[eid] = self._book_transfer(eid, critical=True)
        for action in actions:
            if action.kind == "prefetch":
                for eid in action.expert_ids:
                    self._book_transfer(eid, critical=False)

        completions: list[float] = []
        for eid in list(gpu_ids) + default_gpu:
            start = max(
                clock,
                load_times.get(eid, clock),
                state.pending_loads.get(eid, clock),
            )
            completions.append(self.gpu.schedule(start, self.profiles[eid].gpu_exec_ms))
        for eid in cpu_ids:
            if self.cpu is None:
                raise RuntimeError("execute_cpu requested but no CPU resource configured")
            completions.append(
                self.cpu.schedule(clock, self.profiles[eid].quantized_cpu_exec_ms())
            )

        kv_times: list[float] = []
        for action in actions:
            if action.kind == "evict_kv":
                for eid in action.expert_ids:
                    kv_times.append(self._kv_transfer(eid))
            elif action.kind == "fetch_kv":
                for eid in action.expert_ids:
                    kv_times.append(self._kv_transfer(eid))
        completions.extend(kv_times)

        completion = max(completions, default=clock)
        self._metrics.record_completion(tokens=1, time_ms=completion - clock)
        return completion

    def _prune_pending_loads(self) -> None:
        self._state.pending_loads = {
            eid: c for eid, c in self._state.pending_loads.items() if c > self._clock
        }

    def _account_kv(self, token_count: int, req_id: int | None = None) -> None:
        kv_per_token = self._state.kv_per_token_mb
        if kv_per_token <= 0.0:
            return
        added = token_count * kv_per_token
        self._state.kv_gpu_mb += added
        if req_id is not None:
            self._req_kv_gpu[req_id] = self._req_kv_gpu.get(req_id, 0.0) + added
        capacity = self._state.kv_gpu_capacity_mb
        if capacity > 0.0 and self._state.kv_gpu_mb > capacity:
            excess = self._state.kv_gpu_mb - capacity
            self._state.kv_gpu_mb = capacity
            self._state.kv_host_mb += excess
            if req_id is not None:
                self._req_kv_host[req_id] = self._req_kv_host.get(req_id, 0.0) + excess
            self._metrics.record_kv_offload(excess)
            completion = self.pcie.reserve(self._clock, excess)
            transfer_ms = self.pcie.transfer_time_ms(excess)
            self._metrics.record_transfer_wait(
                max(0.0, completion - transfer_ms - self._clock)
            )
        if capacity > 0.0:
            self._state.kv_pressure = self._state.kv_gpu_mb / capacity
            self._metrics.record_kv_sample("gpu", self._state.kv_gpu_mb / capacity)
            self._metrics.record_kv_peak(self._state.kv_gpu_mb)
            if self._kv_host_capacity_mb > 0.0:
                self._metrics.record_kv_sample(
                    "host", min(1.0, self._state.kv_host_mb / self._kv_host_capacity_mb)
                )

    def _release_kv(self, req_id: int) -> None:
        gpu = self._req_kv_gpu.pop(req_id, 0.0)
        host = self._req_kv_host.pop(req_id, 0.0)
        self._state.kv_gpu_mb = max(0.0, self._state.kv_gpu_mb - gpu)
        self._state.kv_host_mb = max(0.0, self._state.kv_host_mb - host)

    def _snapshot_feedback(self, experts: list[str]) -> None:
        self._state.pcie_queue_len = self.pcie.queue_depth(self._clock)
        self._state.gpu_queue_len = self.gpu.queue_depth(self._clock)
        self._state.cpu_queue_len = (
            self.cpu.queue_depth(self._clock) if self.cpu is not None else 0
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
