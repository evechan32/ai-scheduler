import json
import pytest
from moesim.sim.calibrate import calibrate_pcie, load_profiles


def test_calibrate_pcie():
    bw = calibrate_pcie(12.0)
    assert bw.bandwidth_gbps == 12.0


def test_load_profiles(tmp_path):
    data = [
        {"expert_id": "e0", "size_mb": 340.0, "gpu_exec_ms": 1.2, "cpu_exec_ms": 4.5},
        {"expert_id": "e1", "size_mb": 340.0, "gpu_exec_ms": 1.1, "cpu_exec_ms": 4.2},
    ]
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(data))
    profiles = load_profiles(str(p))
    assert len(profiles) == 2
    assert profiles[0]["expert_id"] == "e0"
    assert profiles[0]["cpu_exec_ms"] == 4.5


def test_load_profiles_missing_key(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([{"expert_id": "e0"}]))
    with pytest.raises(ValueError):
        load_profiles(str(p))
