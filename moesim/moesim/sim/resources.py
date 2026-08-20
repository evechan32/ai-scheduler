"""Domain-agnostic resource models with explicit queueing behavior."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class BandwidthResource:
    """A serialized transfer channel (e.g., PCIe, DRAM bus).

    Queueing behavior is explicit: every reservation is recorded as a
    (start, completion) tuple so the scheduler and metrics can observe
    queue depth, utilization, and waiting time. ``reserve`` semantics are
    unchanged (FIFO serialization on ``_busy_until``).
    """

    def __init__(self, bandwidth_gbps: float, latency_ms: float = 0.0) -> None:
        self.bandwidth_gbps = bandwidth_gbps
        self.latency_ms = latency_ms
        self._busy_until = 0.0
        self._reservations: list[tuple[float, float]] = []  # (start, completion)

    def transfer_time_ms(self, size_mb: float) -> float:
        return size_mb / self.bandwidth_gbps + self.latency_ms

    def reserve(self, now: float, size_mb: float) -> float:
        start = max(now, self._busy_until)
        completion = start + self.transfer_time_ms(size_mb)
        self._busy_until = completion
        self._reservations.append((start, completion))
        return completion

    def queue_depth(self, now: float) -> int:
        """Number of transfers still in flight or queued at ``now``."""
        return sum(1 for _, completion in self._reservations if completion > now)

    def utilization(self, until: float) -> float:
        """Fraction of the window [0, until] the channel was busy."""
        if until <= 0.0:
            return 0.0
        busy = sum(
            (min(completion, until) - min(start, until))
            for start, completion in self._reservations
            if start < until
        )
        return min(1.0, busy / until)

    def wait_time_ms(self, now: float, size_mb: float) -> float:
        """Peek: time a new transfer of ``size_mb`` would take to complete.

        Does NOT mutate the resource (used for scheduling decisions).
        """
        return max(now, self._busy_until) + self.transfer_time_ms(size_mb) - now


class ComputeResource:
    """A compute pool with limited concurrency.

    Each reservation is recorded as (slot, start, completion) so queue
    depth, utilization, and peek wait time are observable. ``schedule``
    semantics are unchanged (least-loaded-slot assignment).
    """

    def __init__(self, concurrency: int = 1, per_unit_ms: float = 1.0) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.concurrency = concurrency
        self.per_unit_ms = per_unit_ms
        self._slots: list[float] = [0.0] * concurrency  # each slot's next-free time
        self._reservations: list[tuple[int, float, float]] = []  # (slot, start, completion)

    def process_time_ms(self, units: float) -> float:
        return units * self.per_unit_ms

    def schedule(self, now: float, units: float) -> float:
        slot = min(range(self.concurrency), key=lambda i: self._slots[i])
        start = max(now, self._slots[slot])
        completion = start + self.process_time_ms(units)
        self._slots[slot] = completion
        self._reservations.append((slot, start, completion))
        return completion

    def queue_depth(self, now: float) -> int:
        """Number of jobs still in flight or queued at ``now``."""
        return sum(1 for _, _, completion in self._reservations if completion > now)

    def utilization(self, until: float) -> float:
        """Fraction of slots busy over the window [0, until]."""
        if until <= 0.0:
            return 0.0
        busy = sum(
            (min(completion, until) - min(start, until))
            for _, start, completion in self._reservations
            if start < until
        )
        return min(1.0, busy / (until * self.concurrency))

    def wait_time_ms(self, now: float, units: float) -> float:
        """Peek: earliest completion time for a new job of ``units`` minus ``now``.

        Does NOT mutate the resource (used for scheduling decisions).
        """
        earliest_slot = min(self._slots)
        return max(now, earliest_slot) + self.process_time_ms(units) - now


@dataclass
class StorageResource:
    capacity_mb: float
    used_mb: float = 0.0

    def fits(self, size_mb: float) -> bool:
        return self.used_mb + size_mb <= self.capacity_mb + 1e-9

    def insert(self, size_mb: float) -> None:
        if not self.fits(size_mb):
            raise ValueError(
                f"capacity {self.capacity_mb}MB exceeded: used {self.used_mb}MB + {size_mb}MB"
            )
        self.used_mb += size_mb

    def remove(self, size_mb: float) -> None:
        if size_mb > self.used_mb + 1e-9:
            raise ValueError(f"cannot remove {size_mb}MB from used {self.used_mb}MB")
        self.used_mb -= size_mb


class MultiGPUCluster:
    """Multiple GPU nodes with a pairwise bandwidth matrix (GB/s)."""

    def __init__(self, gpu_capacities_mb: list[float], bandwidth_matrix: np.ndarray) -> None:
        assert bandwidth_matrix.shape == (len(gpu_capacities_mb),) * 2
        self.gpu_capacities_mb = list(gpu_capacities_mb)
        self.bandwidth_matrix = bandwidth_matrix

    def transfer_time_ms(self, src: int, dst: int, size_mb: float) -> float:
        if src == dst:
            return 0.0
        return size_mb / float(self.bandwidth_matrix[src, dst])
