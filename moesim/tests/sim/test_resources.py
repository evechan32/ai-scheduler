import pytest
from moesim.sim.resources import BandwidthResource, ComputeResource, StorageResource


def test_bandwidth_transfer_time():
    bw = BandwidthResource(bandwidth_gbps=12.0, latency_ms=1.0)
    # 120 MB at 12 GB/s => 10 ms + 1 ms latency
    assert bw.transfer_time_ms(120.0) == pytest.approx(11.0)


def test_bandwidth_serializes_concurrent_transfers():
    bw = BandwidthResource(bandwidth_gbps=12.0)
    first = bw.reserve(0.0, 120.0)   # finishes at 10 ms
    second = bw.reserve(0.0, 60.0)   # queued behind first => 10 + 5 = 15 ms
    assert first == pytest.approx(10.0)
    assert second == pytest.approx(15.0)


def test_compute_concurrency_limit():
    comp = ComputeResource(concurrency=2, per_unit_ms=1.0)
    a = comp.schedule(0.0, 10.0)   # starts at 0, finishes 10
    b = comp.schedule(0.0, 10.0)   # starts at 0, finishes 10
    c = comp.schedule(0.0, 10.0)   # must wait for one slot => finishes 20
    assert a == pytest.approx(10.0)
    assert b == pytest.approx(10.0)
    assert c == pytest.approx(20.0)


def test_storage_capacity():
    st = StorageResource(capacity_mb=100.0)
    assert st.fits(50.0)
    st.insert(50.0)
    assert st.used_mb == 50.0
    with pytest.raises(ValueError):
        st.insert(60.0)
    st.remove(50.0)
    assert st.used_mb == 0.0
