"""v9: request-level concurrent simulation — prefill/decode, FIFO queuing, latency breakdown."""
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.request_sim import Request, RequestSimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _profiles(n=4):
    return {f"e{i}": ExpertProfile(f"e{i}", 10.0, 1.0, 5.0) for i in range(n)}


def _sim(gpu_concurrency=1, prefill_per_token_ms=0.5, kv_per_token_mb=0.0):
    pcie = BandwidthResource(bandwidth_gbps=10.0)
    return RequestSimulation(
        scheduler=LRUPolicy(),
        profiles=_profiles(),
        gpu_capacity_mb=100.0,
        pcie=pcie,
        gpu=ComputeResource(concurrency=gpu_concurrency, per_unit_ms=1.0),
        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
        prefill_per_token_ms=prefill_per_token_ms,
        kv_per_token_mb=kv_per_token_mb,
        kv_gpu_capacity_mb=1000.0,
    )


def test_ttft_includes_prefill_execution():
    sim = _sim()
    stats = sim.run([Request(0, 0.0, prompt_tokens=10, output_tokens=5)])
    r = stats[0]
    # prefill: 10 tokens x 0.5ms = 5ms; no queuing (idle GPU)
    assert r.prefill_queuing_ms == 0.0
    assert r.prefill_exec_ms == 5.0
    assert r.ttft_ms == 5.0


def test_prefill_queues_behind_busy_gpu():
    sim = _sim()
    stats = sim.run([
        Request(0, 0.0, prompt_tokens=10, output_tokens=1),
        Request(1, 0.0, prompt_tokens=10, output_tokens=1),
    ])
    # req0 prefill 0->5ms; req1 prefill queues behind => 5->10ms
    assert stats[1].prefill_queuing_ms == 5.0
    assert stats[1].ttft_ms == 10.0


def test_concurrent_decode_shares_gpu_slots():
    serial = _sim(gpu_concurrency=1)
    parallel = _sim(gpu_concurrency=2)
    reqs = [Request(0, 0.0, 5, 10), Request(1, 0.0, 5, 10)]
    m_serial = serial.run(reqs)
    m_parallel = parallel.run(reqs)
    # with 2 slots both requests decode in parallel -> better JCT than serialized
    assert max(r.jct_ms for r in m_parallel) < max(r.jct_ms for r in m_serial)


def test_latency_breakdown_components():
    sim = _sim()
    r = sim.run([Request(0, 1.0, prompt_tokens=10, output_tokens=4)])[0]
    assert r.prefill_queuing_ms >= 0.0
    assert r.prefill_exec_ms > 0.0
    assert r.tpot_avg_ms > 0.0
    assert r.jct_ms == r.ttft_ms + r.tpot_avg_ms * 4


def test_kv_grows_per_request_token():
    sim = _sim(kv_per_token_mb=1.0)
    sim.run([Request(0, 0.0, prompt_tokens=10, output_tokens=6)])
    # prefill 10 + decode 6 tokens of KV
    assert sim._state.kv_gpu_mb == 16.0


def test_deterministic_across_runs():
    reqs = [Request(i, float(i), prompt_tokens=8, output_tokens=5) for i in range(3)]
    a = [(r.ttft_ms, r.jct_ms) for r in _sim().run(reqs)]
    b = [(r.ttft_ms, r.jct_ms) for r in _sim().run(reqs)]
    assert a == b
