from moesim.sim.metrics import Metrics


def test_tpot_and_throughput():
    m = Metrics(total_tokens=100, total_time_ms=5000.0)
    assert m.tpot_ms() == 50.0
    assert m.throughput_tok_s() == 20.0


def test_hit_rate():
    m = Metrics()
    m.record_access(hit=True)
    m.record_access(hit=True)
    m.record_access(hit=False)
    assert m.hit_rate() == 2 / 3


def test_zero_tokens_safe():
    m = Metrics()
    assert m.tpot_ms() == 0.0
    assert m.throughput_tok_s() == 0.0
    assert m.hit_rate() == 0.0
