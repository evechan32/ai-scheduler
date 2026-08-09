"""Micro-benchmark calibration helpers: turn measured numbers into resource params."""
from __future__ import annotations

import json
from pathlib import Path

from moesim.sim.resources import BandwidthResource

_REQUIRED_KEYS = {"expert_id", "size_mb", "gpu_exec_ms", "cpu_exec_ms"}


def calibrate_pcie(bandwidth_gbps: float, latency_ms: float = 0.0) -> BandwidthResource:
    return BandwidthResource(bandwidth_gbps=bandwidth_gbps, latency_ms=latency_ms)


def load_profiles(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("profiles file must be a JSON list")
    for item in data:
        missing = _REQUIRED_KEYS - set(item.keys())
        if missing:
            raise ValueError(f"profile entry missing keys: {sorted(missing)}")
    return data
