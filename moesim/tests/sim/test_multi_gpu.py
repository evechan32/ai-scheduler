import numpy as np

from moesim.sim.resources import MultiGPUCluster


def test_multi_gpu_transfer_time_uses_bandwidth_matrix():
    cluster = MultiGPUCluster(
        gpu_capacities_mb=[100.0, 100.0],
        bandwidth_matrix=np.array([[100.0, 20.0], [20.0, 100.0]]),
    )
    # 10MB across 20 GB/s = 0.5ms
    assert cluster.transfer_time_ms(0, 1, 10.0) == 0.5
    # same-GPU transfer negligible
    assert cluster.transfer_time_ms(0, 0, 10.0) == 0.0


def test_state_per_gpu_residency():
    from moesim.scheduler.cost_model import ExpertProfile
    from moesim.scheduler.state import ScheduleState

    profiles = {f"e{i}": ExpertProfile(f"e{i}", 1.0, 0.1, 0.5) for i in range(3)}
    s = ScheduleState(profiles=profiles, resident=set(), gpu_capacity_mb=10.0,
                      gpu_resident={0: {"e0"}, 1: {"e1"}})
    assert s.gpu_resident[0] == {"e0"}
    assert s.gpu_resident[1] == {"e1"}
    assert s.resident == {"e0"}  # view of gpu 0
