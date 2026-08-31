"""v6: resource queue visibility — queue depth, utilization, peek wait time."""
import pytest

from moesim.sim.resources import BandwidthResource, ComputeResource


def test_bandwidth_queue_depth_and_wait():
    bw = BandwidthResource(bandwidth_gbps=10.0)  # 100MB => 10ms
    bw.reserve(0.0, 100.0)
    bw.reserve(0.0, 100.0)  # queued behind first => starts at 10
    assert bw.queue_depth(0.0) == 2
    assert bw.queue_depth(20.0) == 0  # both finished at 10 and 20
    # peek: at t=5 the bus is busy until 20; a new 100MB (10ms) finishes at 30
    assert bw.wait_time_ms(5.0, 100.0) == 25.0
    assert bw.wait_time_ms(30.0, 100.0) == 10.0  # idle at 30 => just transfer
    assert bw.utilization(20.0) == pytest.approx(1.0)  # busy 20 of 20 ms
    assert bw.utilization(40.0) == pytest.approx(0.5)  # busy 20 of 40 ms


def test_bandwidth_wait_peek_does_not_mutate():
    bw = BandwidthResource(bandwidth_gbps=10.0)
    bw.reserve(0.0, 100.0)
    before = bw.queue_depth(0.0)
    bw.wait_time_ms(0.0, 100.0)  # peek must not schedule anything
    assert bw.queue_depth(0.0) == before
    assert bw.wait_time_ms(0.0, 100.0) == 20.0  # still only the one transfer


def test_compute_queue_depth_utilization_wait():
    comp = ComputeResource(concurrency=2, per_unit_ms=1.0)
    comp.schedule(0.0, 10.0)
    comp.schedule(0.0, 10.0)
    assert comp.queue_depth(0.0) == 2
    # both run in parallel, finish at 10; a third 10-unit job waits 10ms
    assert comp.wait_time_ms(0.0, 10.0) == 20.0  # earliest slot busy until 10 => 10+10
    assert comp.utilization(10.0) == pytest.approx(1.0)  # 2 slots busy 10/10
    assert comp.utilization(20.0) == pytest.approx(0.5)


def test_compute_concurrency_limited_queue_depth():
    comp = ComputeResource(concurrency=1, per_unit_ms=1.0)
    comp.schedule(0.0, 5.0)
    comp.schedule(0.0, 5.0)  # serialized => finishes 10
    assert comp.queue_depth(0.0) == 2
    assert comp.queue_depth(6.0) == 1
    assert comp.queue_depth(10.0) == 0


def test_compute_wait_peek_does_not_mutate():
    comp = ComputeResource(concurrency=1, per_unit_ms=1.0)
    comp.schedule(0.0, 10.0)
    before = comp.queue_depth(0.0)
    assert comp.wait_time_ms(0.0, 10.0) == 20.0
    assert comp.queue_depth(0.0) == before  # peek must not occupy a slot
