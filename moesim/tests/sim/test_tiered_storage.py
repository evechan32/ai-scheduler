"""2.0: three-tier storage (VRAM / DRAM / disk) with tier-to-tier transfer cost."""
import pytest

from moesim.sim.resources import StorageTier, TieredStorage


def _three_tier():
    return TieredStorage([
        StorageTier("vram", capacity_mb=100.0, bandwidth_gbps=1000.0, latency_ms=0.0),
        StorageTier("dram", capacity_mb=1000.0, bandwidth_gbps=50.0, latency_ms=0.1),
        StorageTier("disk", capacity_mb=10000.0, bandwidth_gbps=2.0, latency_ms=5.0),
    ])


def test_tiered_insert_remove():
    ts = _three_tier()
    ts.insert(0, 50.0)
    ts.insert(2, 500.0)
    assert ts.tiers[0].used_mb == 50.0
    assert ts.tiers[2].used_mb == 500.0
    ts.remove(2, 100.0)
    assert ts.tiers[2].used_mb == 400.0


def test_tier_capacity_enforced():
    ts = _three_tier()
    with pytest.raises(ValueError):
        ts.insert(0, 101.0)


def test_transfer_time_disk_to_dram():
    ts = _three_tier()
    # disk->dram: 2GB/s + 5ms latency => 100MB = 50ms + 5ms = 55ms
    assert ts.transfer_time_ms(2, 1, 100.0) == pytest.approx(55.0)


def test_transfer_time_dram_to_vram():
    ts = _three_tier()
    # dram->vram: PCIe-ish 50GB/s + 0.1ms => 100MB = 2ms + 0.1ms = 2.1ms
    assert ts.transfer_time_ms(1, 0, 100.0) == pytest.approx(2.1)


def test_promote_demote_move_bytes():
    ts = _three_tier()
    ts.insert(2, 300.0)
    ts.promote(2, 1, 200.0)  # disk -> dram
    assert ts.tiers[2].used_mb == 100.0
    assert ts.tiers[1].used_mb == 200.0
    ts.demote(1, 2, 50.0)  # dram -> disk
    assert ts.tiers[1].used_mb == 150.0
    assert ts.tiers[2].used_mb == 150.0
