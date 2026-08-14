"""Domain-agnostic resource models with explicit queueing behavior."""
from __future__ import annotations

from dataclasses import dataclass, field


class BandwidthResource:
    """A serialized transfer channel (e.g., PCIe, DRAM bus)."""

    def __init__(self, bandwidth_gbps: float, latency_ms: float = 0.0) -> None:
        self.bandwidth_gbps = bandwidth_gbps
        self.latency_ms = latency_ms
        self._busy_until = 0.0

    def transfer_time_ms(self, size_mb: float) -> float:
        return size_mb / self.bandwidth_gbps + self.latency_ms

    def reserve(self, now: float, size_mb: float) -> float:
        start = max(now, self._busy_until)
        completion = start + self.transfer_time_ms(size_mb)
        self._busy_until = completion
        return completion


class ComputeResource:
    """A compute pool with limited concurrency."""

    def __init__(self, concurrency: int = 1, per_unit_ms: float = 1.0) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.concurrency = concurrency
        self.per_unit_ms = per_unit_ms
        self._slots: list[float] = [0.0] * concurrency  # each slot's next-free time

    def process_time_ms(self, units: float) -> float:
        return units * self.per_unit_ms

    def schedule(self, now: float, units: float) -> float:
        slot = min(range(self.concurrency), key=lambda i: self._slots[i])
        start = max(now, self._slots[slot])
        completion = start + self.process_time_ms(units)
        self._slots[slot] = completion
        return completion


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
