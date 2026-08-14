import pytest
from moesim.executor.kv_manager import KVTierManager


def test_tier_accounting():
    m = KVTierManager(gpu_pool_mb=100.0, host_pool_mb=200.0)
    m.allocate_gpu(60.0)
    assert m.gpu_used_mb == 60.0
    m.transfer_gpu_to_host(60.0)
    assert m.gpu_used_mb == 0.0
    assert m.host_used_mb == 60.0
    m.free(60.0)
    assert m.host_used_mb == 0.0


def test_tier_over_allocation_raises():
    m = KVTierManager(gpu_pool_mb=100.0, host_pool_mb=200.0)
    with pytest.raises(ValueError):
        m.allocate_gpu(150.0)
