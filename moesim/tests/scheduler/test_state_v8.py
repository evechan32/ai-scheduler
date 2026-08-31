"""v8: KV tiering state fields and metrics defaults."""
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.state import ScheduleState
from moesim.sim.metrics import Metrics


def test_v8_state_defaults():
    profiles = {"e0": ExpertProfile("e0", 10.0, 1.0, 5.0)}
    s = ScheduleState(profiles=profiles, resident=set(), gpu_capacity_mb=100.0)
    assert s.kv_per_token_mb == 0.0
    assert s.kv_pressure == 0.0
    assert s.kv_evict_count == 0
    assert s.kv_fetch_count == 0


def test_v8_metrics_defaults_and_recording():
    m = Metrics()
    assert m.kv_gpu_utilization == {"mean": 0.0, "max": 0.0, "p95": 0.0}
    assert m.kv_offload_bytes == 0.0
    m.record_kv_sample("gpu", 0.5)
    m.record_kv_sample("gpu", 0.9)
    assert m.kv_gpu_utilization["mean"] == 0.7
    assert m.kv_gpu_utilization["max"] == 0.9
    m.record_kv_offload(12.5)
    assert m.kv_offload_bytes == 12.5
