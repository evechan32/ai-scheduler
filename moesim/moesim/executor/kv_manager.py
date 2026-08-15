"""KV cache tier accounting: GPU pool / host pool (v1 skeleton)."""
from __future__ import annotations


class KVTierManager:
    def __init__(self, gpu_pool_mb: float, host_pool_mb: float) -> None:
        self.gpu_pool_mb = gpu_pool_mb
        self.host_pool_mb = host_pool_mb
        self.gpu_used_mb = 0.0
        self.host_used_mb = 0.0

    def can_allocate(self, mb: float) -> bool:
        return self.gpu_used_mb + mb <= self.gpu_pool_mb + 1e-9

    def allocate_gpu(self, mb: float) -> None:
        if self.gpu_used_mb + mb > self.gpu_pool_mb + 1e-9:
            raise ValueError(f"GPU pool overflow: {self.gpu_used_mb}+{mb}>{self.gpu_pool_mb}")
        self.gpu_used_mb += mb

    def allocate_host(self, mb: float) -> None:
        if self.host_used_mb + mb > self.host_pool_mb + 1e-9:
            raise ValueError(f"host pool overflow: {self.host_used_mb}+{mb}>{self.host_pool_mb}")
        self.host_used_mb += mb

    def transfer_gpu_to_host(self, mb: float) -> None:
        if mb > self.gpu_used_mb + 1e-9:
            raise ValueError(f"cannot transfer {mb}MB, only {self.gpu_used_mb}MB on GPU")
        self.gpu_used_mb -= mb
        self.host_used_mb += mb

    def free(self, mb: float) -> None:
        if mb > self.gpu_used_mb + self.host_used_mb + 1e-9:
            raise ValueError("free amount exceeds total usage")
        gpu_freed = min(mb, self.gpu_used_mb)
        self.gpu_used_mb -= gpu_freed
        self.host_used_mb -= mb - gpu_freed

    def total_kv_mb(self) -> float:
        return self.gpu_used_mb + self.host_used_mb

    def pressure(self) -> float:
        return self.gpu_used_mb / self.gpu_pool_mb

    def transfer_host_to_gpu(self, mb: float) -> None:
        if mb > self.host_used_mb + 1e-9:
            raise ValueError(f"cannot transfer {mb}MB, only {self.host_used_mb}MB on host")
        self.host_used_mb -= mb
        self.gpu_used_mb += mb
