"""2.0: DiskTierPolicy demotes cold experts to disk."""
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.disk_tier import DiskTierPolicy
from moesim.scheduler.state import ScheduleState
from moesim.sim.resources import BandwidthResource


def _profiles():
    return {
        "e0": ExpertProfile("e0", 10.0, 1.0, 5.0, activation_freq=0.9),
        "e1": ExpertProfile("e1", 10.0, 1.0, 5.0, activation_freq=0.5),
        "e2": ExpertProfile("e2", 10.0, 1.0, 5.0, activation_freq=0.05),
        "e3": ExpertProfile("e3", 10.0, 1.0, 5.0, activation_freq=0.01),
    }


def test_disk_tier_demotes_coldest_experts():
    pcie = BandwidthResource(bandwidth_gbps=10.0)
    s = ScheduleState(profiles=_profiles(), resident=set(), gpu_capacity_mb=100.0,
                      requested=("e0",))
    actions = DiskTierPolicy(pcie=pcie, prefetch_n=0, disk_budget_mb=20.0).decide(s, 0.0)
    demoted = {e for a in actions if a.kind == "demote_to_disk" for e in a.expert_ids}
    # 20MB budget = 2 experts: coldest are e3 (0.01), e2 (0.05)
    assert demoted == {"e3", "e2"}


def test_disk_tier_zero_budget_no_demotion():
    pcie = BandwidthResource(bandwidth_gbps=10.0)
    s = ScheduleState(profiles=_profiles(), resident=set(), gpu_capacity_mb=100.0,
                      requested=("e0",))
    actions = DiskTierPolicy(pcie=pcie, prefetch_n=0, disk_budget_mb=0.0).decide(s, 0.0)
    assert not any(a.kind == "demote_to_disk" for a in actions)
