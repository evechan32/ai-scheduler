"""Aggregated simulation metrics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    total_tokens: int = 0
    total_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    def record_completion(self, tokens: int, time_ms: float) -> None:
        self.total_tokens += tokens
        self.total_time_ms += time_ms

    def record_access(self, hit: bool) -> None:
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def tpot_ms(self) -> float:
        return self.total_time_ms / self.total_tokens if self.total_tokens else 0.0

    def throughput_tok_s(self) -> float:
        return self.total_tokens / (self.total_time_ms / 1000.0) if self.total_time_ms else 0.0

    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0
