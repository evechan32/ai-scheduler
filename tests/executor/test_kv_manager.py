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


def test_transfer_host_to_gpu_and_total():
    m = KVTierManager(gpu_pool_mb=100.0, host_pool_mb=200.0)
    m.allocate_gpu(30.0)
    m.transfer_gpu_to_host(10.0)
    assert m.gpu_used_mb == pytest.approx(20.0)
    assert m.host_used_mb == pytest.approx(10.0)
    assert m.total_kv_mb() == pytest.approx(30.0)
    m.transfer_host_to_gpu(5.0)
    assert m.gpu_used_mb == pytest.approx(25.0)
    assert m.host_used_mb == pytest.approx(5.0)


def test_pressure_ratio():
    m = KVTierManager(gpu_pool_mb=100.0, host_pool_mb=200.0)
    assert m.pressure() == pytest.approx(0.0)
    m.allocate_gpu(90.0)
    assert m.pressure() == pytest.approx(0.9)
