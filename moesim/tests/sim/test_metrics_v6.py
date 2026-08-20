"""v6: queueing and overlap metrics."""
from moesim.sim.metrics import Metrics


def test_metrics_defaults():
    m = Metrics()
    assert m.pcie_queue_depth_avg == 0.0
    assert m.pcie_queue_depth_max == 0
    assert m.pcie_utilization == 0.0
    assert m.hidden_transfer_ms == 0.0
    assert m.prefetch_count == 0
    assert m.overlap_ratio() == 0.0


def test_record_queue_samples_aggregates():
    m = Metrics()
    m.record_queue_sample("pcie", 1)
    m.record_queue_sample("pcie", 3)
    m.record_queue_sample("pcie", 3)
    assert m.pcie_queue_depth_avg == pytest.approx(7 / 3)
    assert m.pcie_queue_depth_max == 3


def test_record_utilization_averages_per_step():
    m = Metrics()
    m.record_utilization("cpu", 0.5)
    m.record_utilization("cpu", 0.7)
    assert m.cpu_utilization == pytest.approx(0.6)
    assert m.pcie_utilization == 0.0


def test_overlap_ratio():
    m = Metrics()
    m.record_transfer(10.0)          # 10ms total transfer
    m.record_hidden_transfer(6.0)    # 6ms hidden behind compute
    m.record_prefetch()
    assert m.hidden_transfer_ms == 6.0
    assert m.prefetch_count == 1
    assert m.overlap_ratio() == pytest.approx(0.6)


def test_overlap_ratio_no_transfer():
    assert Metrics().overlap_ratio() == 0.0


def test_existing_metrics_untouched():
    m = Metrics()
    m.record_completion(tokens=2, time_ms=100.0)
    m.record_access(hit=True)
    assert m.total_tokens == 2
    assert m.total_time_ms == 100.0
    assert m.cache_hits == 1
    assert m.tpot_ms() == 50.0
    assert m.throughput_tok_s() == 20.0
