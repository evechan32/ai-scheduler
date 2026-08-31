"""Aggregated simulation metrics."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metrics:
    total_tokens: int = 0
    total_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    pcie_queue_depth_avg: float = 0.0
    pcie_queue_depth_max: int = 0
    gpu_queue_depth_avg: float = 0.0
    gpu_queue_depth_max: int = 0
    cpu_queue_depth_avg: float = 0.0
    cpu_queue_depth_max: int = 0
    pcie_utilization: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    transfer_wait_ms: float = 0.0
    total_transfer_ms: float = 0.0
    hidden_transfer_ms: float = 0.0
    prefetch_count: int = 0
    kv_gpu_utilization: dict[str, float] = field(default_factory=lambda: {"mean": 0.0, "max": 0.0, "p95": 0.0})
    kv_host_utilization: dict[str, float] = field(default_factory=lambda: {"mean": 0.0, "max": 0.0, "p95": 0.0})
    kv_offload_bytes: float = 0.0
    kv_peak_mb: float = 0.0
    _kv_samples: dict[str, list[float]] = field(default_factory=dict)
    _queue_samples: dict[str, list[int]] = field(default_factory=dict)
    _utilization_samples: dict[str, list[float]] = field(default_factory=dict)

    def record_completion(self, tokens: int, time_ms: float) -> None:
        self.total_tokens += tokens
        self.total_time_ms += time_ms

    def record_access(self, hit: bool) -> None:
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_queue_sample(self, which: str, depth: int) -> None:
        samples = self._queue_samples.setdefault(which, [])
        samples.append(depth)
        setattr(self, f"{which}_queue_depth_max",
                max(getattr(self, f"{which}_queue_depth_max"), depth))
        setattr(self, f"{which}_queue_depth_avg", sum(samples) / len(samples))

    def record_kv_sample(self, which: str, util: float) -> None:
        samples = self._kv_samples.setdefault(which, [])
        samples.append(util)
        sorted_v = sorted(samples)
        p95 = sorted_v[min(len(sorted_v) - 1, int(0.95 * len(sorted_v)))]
        setattr(self, f"kv_{which}_utilization", {
            "mean": sum(samples) / len(samples),
            "max": max(samples),
            "p95": p95,
        })

    def record_kv_offload(self, mb: float) -> None:
        self.kv_offload_bytes += mb

    def record_kv_peak(self, mb: float) -> None:
        self.kv_peak_mb = max(self.kv_peak_mb, mb)

    def record_utilization(self, which: str, util: float) -> None:
        samples = self._utilization_samples.setdefault(which, [])
        samples.append(util)
        setattr(self, f"{which}_utilization", sum(samples) / len(samples))

    def record_transfer_wait(self, ms: float) -> None:
        self.transfer_wait_ms += ms

    def record_transfer(self, ms: float) -> None:
        self.total_transfer_ms += ms

    def record_hidden_transfer(self, ms: float) -> None:
        self.hidden_transfer_ms += ms

    def record_prefetch(self) -> None:
        self.prefetch_count += 1

    def tpot_ms(self) -> float:
        return self.total_time_ms / self.total_tokens if self.total_tokens else 0.0

    def throughput_tok_s(self) -> float:
        return self.total_tokens / (self.total_time_ms / 1000.0) if self.total_time_ms else 0.0

    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def overlap_ratio(self) -> float:
        if self.total_transfer_ms <= 0.0:
            return 0.0
        return min(1.0, self.hidden_transfer_ms / self.total_transfer_ms)
