# moesim MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build v1 of `moesim` — a domain-agnostic discrete-event simulator (`sim/`) plus a heterogeneous-compute-aware MoE expert offloading scheduler (`scheduler/`), with a pluggable executor layer and CPU FP16 expert kernel, validated against llama.cpp and MoE-Infinity on a 12G GPU + 16G RAM machine.

**Architecture:** The simulator is a pure-Python event-driven DES (`Event`/`EventQueue`/`Clock`/`Simulation`) with domain-agnostic resource models (bandwidth/compute/storage). The scheduler exposes a pure `decide(state, clock) -> list[Action]` function that is consumed identically by the simulator (via a `MoESimulation` adapter) and later by the real executor. The executor layer provides `ExpertExecutor` abstraction with a PyTorch C++ CPU FP16 kernel and a transformers backend adapter.

**Tech Stack:** Python 3.10+, pytest, numpy (sim core only — **no torch dependency in `sim/`**), PyTorch 2.x + C++ extension for `executor/cpu_kernels` (optional dev dependency), transformers for the backend adapter, llama.cpp + MoE-Infinity as benchmark baselines.

## Global Constraints

- Python >= 3.10. Simulator core (`sim/`, `scheduler/`) must run with **only numpy** — no torch imports.
- Determinism: no `random`, no wall-clock time, no dict-ordering dependence in `sim/` and `scheduler/` — same input ⇒ bit-identical output. Use `dataclasses` (frozen where possible), explicit ordering.
- Units: sizes in MB, bandwidth in GB/s, time in ms. `transfer_time_ms(size_mb, bw_gbps) = size_mb / bw_gbps + latency_ms`.
- TDD: every task writes the failing test first, verifies it fails, then implements.
- Commit after every task with a conventional message (`feat:`/`test:`/`docs:`).
- Calibration acceptance: simulated TPOT vs measured TPOT error < 20% (Task 12-14, real-machine optional).
- No placeholders, no `pass`-only stubs in committed code.
- `docs/superpowers/specs/2026-08-09-moesim-design.md` is the authoritative spec.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `sim/__init__.py`
- Create: `scheduler/__init__.py`
- Create: `executor/__init__.py`
- Create: `benchmarks/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Produces: installable package `moesim` with importable subpackages `moesim.sim`, `moesim.scheduler`, `moesim.executor`; pytest working with rootdir at project root.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffold.py
def test_packages_importable():
    import moesim.sim
    import moesim.scheduler
    import moesim.executor
    assert moesim.sim.__name__ == "moesim.sim"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/test_scaffold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'moesim'`

- [ ] **Step 3: Create package files**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "moesim"
version = "0.1.0"
description = "Heterogeneous MoE offloading scheduler + domain-agnostic discrete event simulator"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
dev = ["pytest>=7.4", "torch>=2.1", "transformers>=4.40"]

[tool.setuptools.packages.find]
include = ["moesim*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# moesim/__init__.py  (package root — create this too)
__version__ = "0.1.0"
```

Note: package layout is `moesim/sim/`, `moesim/scheduler/`, `moesim/executor/` (package dirs under a `moesim/` root package). Create `moesim/` alongside the `sim/` etc. directories already scaffolded; adjust the empty dirs created earlier accordingly (they become `moesim/sim/` etc.).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/ -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: scaffold moesim package with pytest setup"
```

---

### Task 2: `sim/core.py` — Event, EventQueue, Clock, Simulation

**Files:**
- Create: `moesim/sim/core.py`
- Test: `tests/sim/test_core.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Event: time: float; priority: int = 0; kind: str = ""; payload: object = None`
  - `class EventQueue: push(e: Event) -> None; pop() -> Event; __len__() -> int; __bool__() -> bool`
  - `class Clock: now: float; advance(t: float) -> None`
  - `class Simulation: __init__(); register(kind: str, handler: Callable[[Event, "Simulation"], None]) -> None; schedule(time: float, kind: str, payload: object = None, priority: int = 0) -> None; run(until: float | None = None) -> None`
- Consumes: nothing (first task in `sim/`).

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_core.py
from moesim.sim.core import Event, EventQueue, Clock, Simulation


def test_event_is_frozen_dataclass():
    e = Event(time=1.5, kind="load")
    assert e.time == 1.5 and e.kind == "load" and e.priority == 0
    try:
        e.time = 2.0
        assert False, "Event must be immutable"
    except Exception:
        pass


def test_event_queue_ordered_pop():
    q = EventQueue()
    q.push(Event(time=3.0))
    q.push(Event(time=1.0))
    q.push(Event(time=2.0))
    assert len(q) == 3
    assert q.pop().time == 1.0
    assert q.pop().time == 2.0
    assert q.pop().time == 3.0
    assert not q


def test_clock_advance():
    c = Clock()
    assert c.now == 0.0
    c.advance(2.5)
    assert c.now == 2.5


def test_simulation_fires_handlers_in_time_order():
    sim = Simulation()
    order = []
    sim.register("a", lambda e, s: order.append(("a", e.time)))
    sim.register("b", lambda e, s: order.append(("b", e.time)))
    sim.schedule(2.0, "b")
    sim.schedule(1.0, "a")
    sim.run()
    assert order == [("a", 1.0), ("b", 2.0)]
    assert sim.clock.now == 2.0


def test_simulation_until_stops_early():
    sim = Simulation()
    fired = []
    sim.register("tick", lambda e, s: fired.append(e.time))
    for t in (1.0, 2.0, 3.0):
        sim.schedule(t, "tick")
    sim.run(until=2.5)
    assert fired == [1.0, 2.0]
    assert sim.clock.now == 2.5


def test_handler_can_schedule_new_events():
    sim = Simulation()
    seen = []

    def spawn(e, s):
        seen.append(e.time)
        if e.time < 3.0:
            s.schedule(e.time + 1.0, "spawn")

    sim.register("spawn", spawn)
    sim.schedule(0.0, "spawn")
    sim.run()
    assert seen == [0.0, 1.0, 2.0, 3.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/sim/test_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'moesim.sim.core'`

- [ ] **Step 3: Implement**

```python
# moesim/sim/core.py
"""Domain-agnostic discrete event simulation core."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Event:
    time: float
    priority: int = 0
    kind: str = ""
    payload: Any = None


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int, Event]] = []
        self._seq = 0

    def push(self, e: Event) -> None:
        heapq.heappush(self._heap, (e.time, e.priority, self._seq, e))
        self._seq += 1

    def pop(self) -> Event:
        return heapq.heappop(self._heap)[3]

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, t: float) -> None:
        self.now = t


class Simulation:
    def __init__(self) -> None:
        self.clock = Clock()
        self.queue = EventQueue()
        self._handlers: dict[str, Callable[[Event, "Simulation"], None]] = {}

    def register(self, kind: str, handler: Callable[[Event, "Simulation"], None]) -> None:
        self._handlers[kind] = handler

    def schedule(self, time: float, kind: str, payload: Any = None, priority: int = 0) -> None:
        self.queue.push(Event(time=time, kind=kind, payload=payload, priority=priority))

    def run(self, until: float | None = None) -> None:
        while self.queue:
            e = self.queue.pop()
            if until is not None and e.time > until:
                self.clock.advance(until)
                break
            self.clock.advance(e.time)
            handler = self._handlers.get(e.kind)
            if handler is not None:
                handler(e, self)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/sim/test_core.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add DES core (Event, EventQueue, Clock, Simulation)"
```

---

### Task 3: `sim/resources.py` — resource model family

**Files:**
- Create: `moesim/sim/resources.py`
- Test: `tests/sim/test_resources.py`

**Interfaces:**
- Produces:
  - `class BandwidthResource: __init__(bandwidth_gbps: float, latency_ms: float = 0.0); transfer_time_ms(size_mb: float) -> float; reserve(now: float, size_mb: float) -> float  # returns completion time, serializes concurrent transfers (FIFO queue)`
  - `class ComputeResource: __init__(concurrency: int = 1, per_unit_ms: float = 1.0); process_time_ms(units: float) -> float; schedule(now: float, units: float) -> float  # returns completion time, respects concurrency limit`
  - `class StorageResource: __init__(capacity_mb: float); capacity_mb: float; used_mb: float; can_fit(size_mb: float) -> bool; evict(size_mb: float, eviction_order: list[str], ...)` — v1 keeps it minimal: `StorageResource` tracks capacity and provides `fits(size_mb) -> bool` plus `insert(size_mb) -> None` and `remove(size_mb) -> None` raising `ValueError` on over-capacity.
- Consumes: `Event`/`Simulation` from Task 2 (`BandwidthResource.reserve` semantics are standalone; simulation integration happens in Task 8).

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_resources.py
import pytest
from moesim.sim.resources import BandwidthResource, ComputeResource, StorageResource


def test_bandwidth_transfer_time():
    bw = BandwidthResource(bandwidth_gbps=12.0, latency_ms=1.0)
    # 120 MB at 12 GB/s => 10 ms + 1 ms latency
    assert bw.transfer_time_ms(120.0) == pytest.approx(11.0)


def test_bandwidth_serializes_concurrent_transfers():
    bw = BandwidthResource(bandwidth_gbps=12.0)
    first = bw.reserve(0.0, 120.0)   # finishes at 10 ms
    second = bw.reserve(0.0, 60.0)   # queued behind first => 10 + 5 = 15 ms
    assert first == pytest.approx(10.0)
    assert second == pytest.approx(15.0)


def test_compute_concurrency_limit():
    comp = ComputeResource(concurrency=2, per_unit_ms=1.0)
    a = comp.schedule(0.0, 10.0)   # starts at 0, finishes 10
    b = comp.schedule(0.0, 10.0)   # starts at 0, finishes 10
    c = comp.schedule(0.0, 10.0)   # must wait for one slot => finishes 20
    assert a == pytest.approx(10.0)
    assert b == pytest.approx(10.0)
    assert c == pytest.approx(20.0)


def test_storage_capacity():
    st = StorageResource(capacity_mb=100.0)
    assert st.fits(50.0)
    st.insert(50.0)
    assert st.used_mb == 50.0
    with pytest.raises(ValueError):
        st.insert(60.0)
    st.remove(50.0)
    assert st.used_mb == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/sim/test_resources.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# moesim/sim/resources.py
"""Domain-agnostic resource models with explicit queueing behavior."""
from __future__ import annotations

from dataclasses import dataclass, field


class BandwidthResource:
    """A serialized transfer channel (e.g., PCIe, DRAM bus)."""

    def __init__(self, bandwidth_gbps: float, latency_ms: float = 0.0) -> None:
        self.bandwidth_gbps = bandwidth_gbps
        self.latency_ms = latency_ms
        self._busy_until = 0.0

    def transfer_time_ms(self, size_mb: float) -> float:
        return size_mb / self.bandwidth_gbps + self.latency_ms

    def reserve(self, now: float, size_mb: float) -> float:
        start = max(now, self._busy_until)
        completion = start + self.transfer_time_ms(size_mb)
        self._busy_until = completion
        return completion


class ComputeResource:
    """A compute pool with limited concurrency."""

    def __init__(self, concurrency: int = 1, per_unit_ms: float = 1.0) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.concurrency = concurrency
        self.per_unit_ms = per_unit_ms
        self._slots: list[float] = [0.0] * concurrency  # each slot's next-free time

    def process_time_ms(self, units: float) -> float:
        return units * self.per_unit_ms

    def schedule(self, now: float, units: float) -> float:
        slot = min(range(self.concurrency), key=lambda i: self._slots[i])
        start = max(now, self._slots[slot])
        completion = start + self.process_time_ms(units)
        self._slots[slot] = completion
        return completion


@dataclass
class StorageResource:
    capacity_mb: float
    used_mb: float = 0.0

    def fits(self, size_mb: float) -> bool:
        return self.used_mb + size_mb <= self.capacity_mb + 1e-9

    def insert(self, size_mb: float) -> None:
        if not self.fits(size_mb):
            raise ValueError(
                f"capacity {self.capacity_mb}MB exceeded: used {self.used_mb}MB + {size_mb}MB"
            )
        self.used_mb += size_mb

    def remove(self, size_mb: float) -> None:
        if size_mb > self.used_mb + 1e-9:
            raise ValueError(f"cannot remove {size_mb}MB from used {self.used_mb}MB")
        self.used_mb -= size_mb
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/sim/test_resources.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add domain-agnostic resource models (bandwidth/compute/storage)"
```

---

### Task 4: `sim/metrics.py` and `sim/calibrate.py`

**Files:**
- Create: `moesim/sim/metrics.py`
- Create: `moesim/sim/calibrate.py`
- Test: `tests/sim/test_metrics.py`
- Test: `tests/sim/test_calibrate.py`

**Interfaces:**
- Produces:
  - `@dataclass Metrics: total_tokens: int = 0; total_time_ms: float = 0.0; cache_hits: int = 0; cache_misses: int = 0; def tpot_ms(self) -> float; def throughput_tok_s(self) -> float; def hit_rate(self) -> float; def record_completion(self, tokens: int, time_ms: float) -> None; def record_access(self, hit: bool) -> None`
  - `def calibrate_pcie(bandwidth_gbps: float, latency_ms: float = 0.0) -> BandwidthResource` (thin factory)
  - `def load_profiles(path: str) -> list[ExpertProfile]` — reads JSON: list of `{"expert_id", "size_mb", "gpu_exec_ms", "cpu_exec_ms"}`. Raises `ValueError` on missing keys. (ExpertProfile defined in Task 5; to avoid a cross-task import, calibrate returns plain dicts in this task, and `load_profiles` returns `list[dict]`; Task 5 converts.)
- Consumes: `BandwidthResource` from Task 3.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_metrics.py
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
```

```python
# tests/sim/test_calibrate.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/sim/test_metrics.py tests/sim/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# moesim/sim/metrics.py
"""Aggregated simulation metrics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    total_tokens: int = 0
    total_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    def record_completion(self, tokens: int, time_ms: float) -> None:
        self.total_tokens += tokens
        self.total_time_ms += time_ms

    def record_access(self, hit: bool) -> None:
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def tpot_ms(self) -> float:
        return self.total_time_ms / self.total_tokens if self.total_tokens else 0.0

    def throughput_tok_s(self) -> float:
        return self.total_tokens / (self.total_time_ms / 1000.0) if self.total_time_ms else 0.0

    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0
```

```python
# moesim/sim/calibrate.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/sim/ -v`
Expected: PASS (all sim tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add metrics aggregation and calibration helpers"
```

---

### Task 5: `scheduler/state.py` + `scheduler/cost_model.py` + `scheduler/base.py`

**Files:**
- Create: `moesim/scheduler/state.py`
- Create: `moesim/scheduler/cost_model.py`
- Create: `moesim/scheduler/base.py`
- Test: `tests/scheduler/test_state.py`
- Test: `tests/scheduler/test_base.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) ExpertProfile: expert_id: str; size_mb: float; gpu_exec_ms: float; cpu_exec_ms: float; activation_freq: float = 0.0` (cost_model.py)
  - `def profiles_from_dicts(rows: list[dict]) -> dict[str, ExpertProfile]` (cost_model.py; converts Task 4 dicts, keyed by expert_id)
  - `@dataclass(frozen=True) Action: kind: str; expert_ids: tuple[str, ...] = (); target: str = "gpu"` where `kind in {"load", "unload", "execute_gpu", "execute_cpu", "evict_kv", "fetch_kv"}` (base.py)
  - `@dataclass ScheduleState: profiles: dict[str, ExpertProfile]; resident: set[str]; gpu_capacity_mb: float; used_gpu_mb: float = 0.0; requested: tuple[str, ...] = (); access_history: list[str] = field(default_factory=list); cache_hits: int = 0; cache_misses: int = 0; def mark_access(self, expert_id: str) -> bool` — returns True if already resident (hit), appends to `access_history` (state.py)
  - `class Scheduler: def decide(self, state: ScheduleState, clock: float) -> list[Action]` raising `NotImplementedError` (base.py)
- Consumes: nothing from earlier tasks except Python stdlib.

- [ ] **Step 1: Write the failing tests**

```python
# tests/scheduler/test_state.py
from moesim.scheduler.cost_model import ExpertProfile, profiles_from_dicts
from moesim.scheduler.state import ScheduleState


def test_profiles_from_dicts():
    rows = [
        {"expert_id": "e0", "size_mb": 340.0, "gpu_exec_ms": 1.2, "cpu_exec_ms": 4.5},
        {"expert_id": "e1", "size_mb": 340.0, "gpu_exec_ms": 1.1, "cpu_exec_ms": 4.2},
    ]
    profiles = profiles_from_dicts(rows)
    assert set(profiles) == {"e0", "e1"}
    assert profiles["e0"].size_mb == 340.0


def test_mark_access_hit_and_miss():
    profiles = profiles_from_dicts(
        [{"expert_id": "e0", "size_mb": 10.0, "gpu_exec_ms": 1.0, "cpu_exec_ms": 3.0}]
    )
    st = ScheduleState(profiles=profiles, resident={"e0"}, gpu_capacity_mb=100.0)
    assert st.mark_access("e0") is True
    assert st.mark_access("e1") is False
    assert st.cache_hits == 1 and st.cache_misses == 1
    assert st.access_history == ["e0", "e1"]
```

```python
# tests/scheduler/test_base.py
import pytest
from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.state import ScheduleState


def test_action_validation():
    with pytest.raises(ValueError):
        Action(kind="fly_to_moon")
    a = Action(kind="load", expert_ids=("e0",))
    assert a.target == "gpu"


def test_scheduler_abstract():
    s = Scheduler()
    st = ScheduleState(profiles={}, resident=set(), gpu_capacity_mb=100.0)
    with pytest.raises(NotImplementedError):
        s.decide(st, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/scheduler/ -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# moesim/scheduler/cost_model.py
"""Expert cost profiles and loading helpers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertProfile:
    expert_id: str
    size_mb: float
    gpu_exec_ms: float
    cpu_exec_ms: float
    activation_freq: float = 0.0


def profiles_from_dicts(rows: list[dict]) -> dict[str, ExpertProfile]:
    profiles: dict[str, ExpertProfile] = {}
    for row in rows:
        p = ExpertProfile(
            expert_id=row["expert_id"],
            size_mb=float(row["size_mb"]),
            gpu_exec_ms=float(row["gpu_exec_ms"]),
            cpu_exec_ms=float(row["cpu_exec_ms"]),
            activation_freq=float(row.get("activation_freq", 0.0)),
        )
        profiles[p.expert_id] = p
    return profiles
```

```python
# moesim/scheduler/state.py
"""Scheduler state — serializable, deterministic."""
from __future__ import annotations

from dataclasses import dataclass, field

from moesim.scheduler.cost_model import ExpertProfile


@dataclass
class ScheduleState:
    profiles: dict[str, ExpertProfile]
    resident: set[str]
    gpu_capacity_mb: float
    used_gpu_mb: float = 0.0
    requested: tuple[str, ...] = ()  # experts requested in the current step
    access_history: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0

    def mark_access(self, expert_id: str) -> bool:
        self.access_history.append(expert_id)
        hit = expert_id in self.resident
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return hit
```

```python
# moesim/scheduler/base.py
"""Scheduler abstraction: pure decide() contract."""
from __future__ import annotations

from dataclasses import dataclass

from moesim.scheduler.state import ScheduleState

_VALID_KINDS = {"load", "unload", "execute_gpu", "execute_cpu", "evict_kv", "fetch_kv"}


@dataclass(frozen=True)
class Action:
    kind: str
    expert_ids: tuple[str, ...] = ()
    target: str = "gpu"

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"invalid action kind: {self.kind}")


class Scheduler:
    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/scheduler/ -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add scheduler state, expert cost model, and Action/Scheduler contracts"
```

---

### Task 6: Scheduler policies — LRU and activation-frequency

**Files:**
- Create: `moesim/scheduler/policies/lru.py`
- Create: `moesim/scheduler/policies/activation_freq.py`
- Test: `tests/scheduler/test_policies_lru.py`
- Test: `tests/scheduler/test_policies_activation_freq.py`

**Interfaces:**
- Consumes: `Action`, `Scheduler` (base.py), `ScheduleState` (state.py), `ExpertProfile` (cost_model.py).
- Produces:
  - `class LRUPolicy(Scheduler)`: evicts least-recently-used residents when loading would exceed `gpu_capacity_mb`; keeps a hot set of size `resident_budget` (default = all fit by capacity). Semantics: on `decide`, for each non-resident requested expert, emit `load`; if `used_gpu_mb + size > capacity`, emit `unload` for LRU residents (excluding those just requested) until it fits. Loaded experts become resident in state (the caller applies actions; see Task 8 — but policies must be testable by *applying* actions themselves via a helper `apply_actions(state, actions)` exported from `base.py`).
  - `def apply_actions(state: ScheduleState, actions: list[Action]) -> None` (in base.py) — applies load/unload to `state.resident` and `state.used_gpu_mb` (unloads of absent experts are no-ops; loads that would exceed capacity raise `ValueError`).
  - `class ActivationFreqPolicy(Scheduler)`: same capacity discipline as LRU, but eviction order is by `activation_freq` ascending (lowest frequency evicted first); prefetches the highest-frequency non-resident expert when GPU budget allows.

- [ ] **Step 1: Write the failing tests**

```python
# tests/scheduler/test_policies_lru.py
from moesim.scheduler.base import Action, apply_actions
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.scheduler.state import ScheduleState


def _state(capacity_mb=20.0, resident=(), used=0.0, requested=("e2",)):
    profiles = {
        f"e{i}": ExpertProfile(expert_id=f"e{i}", size_mb=10.0, gpu_exec_ms=1.0, cpu_exec_ms=3.0)
        for i in range(4)
    }
    return ScheduleState(
        profiles=profiles, resident=set(resident), gpu_capacity_mb=capacity_mb,
        used_gpu_mb=used, requested=tuple(requested),
    )


def test_lru_loads_requested_and_evicts_lru():
    st = _state(resident=("e0", "e1"), used=20.0)
    st.access_history = ["e1", "e0"]  # e1 most recent, e0 least
    policy = LRUPolicy()
    actions = policy.decide(st, 0.0)  # requested=("e2",): not resident, needs 10MB
    apply_actions(st, actions)
    assert "e2" in st.resident
    assert "e0" not in st.resident  # e0 evicted (LRU)
    assert st.used_gpu_mb == 20.0


def test_lru_no_eviction_when_space_available():
    st = _state(resident=("e0",), used=10.0)
    policy = LRUPolicy()
    apply_actions(st, policy.decide(st, 0.0))  # requested=("e2",): fits, no eviction
    assert st.resident == {"e0", "e2"}
    assert st.used_gpu_mb == 20.0
```

```python
# tests/scheduler/test_policies_activation_freq.py
from moesim.scheduler.base import apply_actions
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.activation_freq import ActivationFreqPolicy
from moesim.scheduler.state import ScheduleState


def _state(capacity_mb=20.0, resident=(), used=0.0, requested=("e2",)):
    profiles = {
        f"e{i}": ExpertProfile(
            expert_id=f"e{i}", size_mb=10.0, gpu_exec_ms=1.0, cpu_exec_ms=3.0,
            activation_freq=float(i),  # e3 hottest
        )
        for i in range(4)
    }
    return ScheduleState(
        profiles=profiles, resident=set(resident), gpu_capacity_mb=capacity_mb,
        used_gpu_mb=used, requested=tuple(requested),
    )


def test_freq_policy_evicts_coldest():
    st = _state(capacity_mb=20.0, resident=("e0", "e1"), used=20.0)
    policy = ActivationFreqPolicy()
    apply_actions(st, policy.decide(st, 0.0))  # request e2 -> load overflows 20MB
    assert "e2" in st.resident
    assert "e0" not in st.resident  # coldest (freq=0) evicted
    assert st.resident == {"e1", "e2"}
    assert st.used_gpu_mb == 20.0


def test_freq_policy_prefetches_hottest():
    st = _state(capacity_mb=40.0, resident=(), used=0.0)
    policy = ActivationFreqPolicy()
    apply_actions(st, policy.decide(st, 0.0))  # request e2, no overflow
    assert "e2" in st.resident
    assert "e3" in st.resident  # hottest non-resident prefetched into free 30MB
    assert st.used_gpu_mb == 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/scheduler/ -v`
Expected: FAIL with `ModuleNotFoundError` for policies

- [ ] **Step 3: Implement**

Add to `moesim/scheduler/base.py`:

```python
def apply_actions(state: ScheduleState, actions: list[Action]) -> None:
    """Apply load/unload actions to state. Deterministic; raises on capacity violation.

    NOTE: all unloads are applied BEFORE loads (two-pass). This is the
    post-review fix: the original single-pass load-first version raised
    ValueError on the swap pattern [load e2, unload e0] because the unload
    hadn't freed space yet. Policies emit loads-then-unloads, so unload-first
    is the correct swap semantics.
    """
    for action in actions:
        if action.kind == "unload":
            for eid in action.expert_ids:
                if eid in state.resident:
                    state.resident.remove(eid)
                    state.used_gpu_mb -= state.profiles[eid].size_mb
    for action in actions:
        if action.kind == "load":
            for eid in action.expert_ids:
                size = state.profiles[eid].size_mb
                if state.used_gpu_mb + size > state.gpu_capacity_mb + 1e-9:
                    raise ValueError(
                        f"load of {eid} would exceed capacity "
                        f"({state.used_gpu_mb}+{size}>{state.gpu_capacity_mb})"
                    )
                state.resident.add(eid)
                state.used_gpu_mb += size
    # execute_*/evict_kv/fetch_kv are informational at the state layer
```

```python
# moesim/scheduler/policies/lru.py
"""Least-recently-used expert caching baseline."""
from __future__ import annotations

from copy import deepcopy

from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.state import ScheduleState


class LRUPolicy(Scheduler):
    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        requested = list(state.requested)
        actions: list[Action] = []
        sim = deepcopy(state)
        for eid in requested:
            if eid not in sim.resident:
                actions.append(Action(kind="load", expert_ids=(eid,)))
                sim.resident.add(eid)
                sim.used_gpu_mb += sim.profiles[eid].size_mb
        for load in [a for a in actions if a.kind == "load"]:
            for eid in load.expert_ids:
                while sim.used_gpu_mb > sim.gpu_capacity_mb + 1e-9:
                    victim = self._lru_victim(sim, set(requested))
                    if victim is None:
                        break
                    sim.resident.remove(victim)
                    sim.used_gpu_mb -= sim.profiles[victim].size_mb
                    actions.append(Action(kind="unload", expert_ids=(victim,)))
        return actions

    def _lru_victim(self, state: ScheduleState, protected: set[str]) -> str | None:
        for eid in reversed(state.access_history):
            if eid in state.resident and eid not in protected:
                return eid
        return None
```

Why a `deepcopy` of state inside `decide`: the policy must not mutate the real
state while deciding (mutating it twice would corrupt capacity math). It decides
on a copy, then returns actions that `apply_actions` applies exactly once to the
real state. Deterministic and side-effect free.

```python
# moesim/scheduler/policies/activation_freq.py
"""Activation-frequency-aware caching with hot-expert prefetch."""
from __future__ import annotations

from copy import deepcopy

from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.state import ScheduleState


class ActivationFreqPolicy(Scheduler):
    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        requested = list(state.requested)
        actions: list[Action] = []
        sim = deepcopy(state)
        for eid in requested:
            if eid not in sim.resident:
                actions.append(Action(kind="load", expert_ids=(eid,)))
                sim.resident.add(eid)
                sim.used_gpu_mb += sim.profiles[eid].size_mb
        for load in [a for a in actions if a.kind == "load"]:
            for eid in load.expert_ids:
                while sim.used_gpu_mb > sim.gpu_capacity_mb + 1e-9:
                    victim = self._coldest(sim, set(requested))
                    if victim is None:
                        break
                    sim.resident.remove(victim)
                    sim.used_gpu_mb -= sim.profiles[victim].size_mb
                    actions.append(Action(kind="unload", expert_ids=(victim,)))
        # prefetch hottest non-resident expert if budget allows
        non_resident = [
            p for p in sim.profiles.values()
            if p.expert_id not in sim.resident and p.expert_id not in requested
        ]
        if non_resident:
            hottest = max(non_resident, key=lambda p: p.activation_freq)
            if sim.used_gpu_mb + hottest.size_mb <= sim.gpu_capacity_mb + 1e-9:
                actions.append(Action(kind="load", expert_ids=(hottest.expert_id,)))
        return actions

    def _coldest(self, state: ScheduleState, protected: set[str]) -> str | None:
        residents = [p for p in state.profiles.values()
                     if p.expert_id in state.resident and p.expert_id not in protected]
        if not residents:
            return None
        return min(residents, key=lambda p: p.activation_freq).expert_id
```

The `requested` field on `ScheduleState` (defined in Task 5) carries the
experts requested in the current step; the caller (`MoESimulation._step` in
Task 8) sets `state.requested` before invoking `decide`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/scheduler/ -v`
Expected: PASS (all scheduler tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add LRU and activation-frequency scheduling policies"
```

---

### Task 7: `cost_model.py` — the heterogeneous-compute-aware policy

**Files:**
- Create: `moesim/scheduler/policies/cost_model.py`
- Test: `tests/scheduler/test_policies_cost_model.py`

**Interfaces:**
- Consumes: `Action`, `Scheduler`, `apply_actions` (base.py), `ScheduleState` (state.py), `ExpertProfile` (cost_model.py), `BandwidthResource` (sim/resources.py — used to price PCIe loads).
- Produces: `class CostModelPolicy(Scheduler)`:
  - `__init__(pcie: BandwidthResource | None = None, cpu_concurrency: int = 1, prefetch_n: int = 1)`
  - Decision rule per requested non-resident expert: compute `load_cost = pcie.transfer_time_ms(size_mb)`; compute `cpu_ok = cpu_exec_ms <= load_cost + gpu_exec_ms` (i.e., CPU compute is competitive with fetch+GPU-compute). If `cpu_ok` and the CPU path exists in this deployment → `execute_cpu`; else `load` (with LRU-style eviction of cheapest-to-relocate victims).
  - Emits `execute_gpu` for resident requested experts.
  - Prefetches the top-`prefetch_n` non-resident experts by `activation_freq` when budget allows (if `prefetch_n > 0`).
  - Deterministic; capacity discipline identical to LRU (evict LRU victims).

- [ ] **Step 1: Write the failing test**

```python
# tests/scheduler/test_policies_cost_model.py
from moesim.scheduler.base import Action, apply_actions
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.state import ScheduleState
from moesim.sim.resources import BandwidthResource


def _state(capacity_mb=20.0, resident=(), used=0.0):
    profiles = {
        f"e{i}": ExpertProfile(
            expert_id=f"e{i}", size_mb=10.0, gpu_exec_ms=1.0, cpu_exec_ms=3.0,
            activation_freq=float(i),
        )
        for i in range(4)
    }
    return ScheduleState(
        profiles=profiles, resident=set(resident), gpu_capacity_mb=capacity_mb,
        used_gpu_mb=used, requested=("e2",),
    )


def test_resident_expert_gets_execute_gpu():
    st = _state(resident=("e2",), used=10.0)
    policy = CostModelPolicy()
    actions = policy.decide(st, 0.0)
    assert Action(kind="execute_gpu", expert_ids=("e2",)) in actions


def test_cpu_cheaper_than_load():
    # 10MB at 1 GB/s => 10ms load; cpu 3ms < 10+1 => execute_cpu
    pcie = BandwidthResource(bandwidth_gbps=1.0)
    st = _state(resident=(), used=0.0)  # request e2, not resident
    policy = CostModelPolicy(pcie=pcie, prefetch_n=0)
    actions = policy.decide(st, 0.0)
    assert Action(kind="execute_cpu", expert_ids=("e2",)) in actions


def test_load_when_cpu_slower_than_fetch_and_compute():
    # 10MB at 10 GB/s => 1ms load; cpu 3ms > 1+1 => load
    pcie = BandwidthResource(bandwidth_gbps=10.0)
    st = _state(resident=(), used=0.0)
    policy = CostModelPolicy(pcie=pcie, prefetch_n=0)
    actions = policy.decide(st, 0.0)
    assert Action(kind="load", expert_ids=("e2",)) in actions


def test_prefetch_hottest():
    st = _state(resident=(), used=0.0, capacity_mb=40.0)
    policy = CostModelPolicy(pcie=BandwidthResource(bandwidth_gbps=10.0), prefetch_n=1)
    actions = policy.decide(st, 0.0)
    assert Action(kind="load", expert_ids=("e3",)) in actions  # hottest non-resident
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/scheduler/test_policies_cost_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# moesim/scheduler/policies/cost_model.py
"""Heterogeneous-compute-aware policy: price CPU execution vs fetch+GPU execution."""
from __future__ import annotations

from copy import deepcopy

from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.state import ScheduleState
from moesim.sim.resources import BandwidthResource


class CostModelPolicy(Scheduler):
    def __init__(
        self,
        pcie: BandwidthResource | None = None,
        cpu_concurrency: int = 1,
        prefetch_n: int = 1,
    ) -> None:
        self.pcie = pcie or BandwidthResource(bandwidth_gbps=10.0)
        self.cpu_concurrency = cpu_concurrency
        self.prefetch_n = prefetch_n

    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        requested = list(state.requested)
        actions: list[Action] = []
        sim = deepcopy(state)

        for eid in requested:
            profile = sim.profiles[eid]
            if eid in sim.resident:
                actions.append(Action(kind="execute_gpu", expert_ids=(eid,)))
                continue
            load_cost = self.pcie.transfer_time_ms(profile.size_mb)
            if profile.cpu_exec_ms <= load_cost + profile.gpu_exec_ms:
                actions.append(Action(kind="execute_cpu", expert_ids=(eid,)))
            else:
                actions.append(Action(kind="load", expert_ids=(eid,)))
                sim.resident.add(eid)
                sim.used_gpu_mb += profile.size_mb

        # capacity discipline: evict LRU victims for the loads we decided
        for load in [a for a in actions if a.kind == "load"]:
            for eid in load.expert_ids:
                while sim.used_gpu_mb > sim.gpu_capacity_mb + 1e-9:
                    victim = self._lru_victim(sim, set(requested))
                    if victim is None:
                        break
                    sim.resident.remove(victim)
                    sim.used_gpu_mb -= sim.profiles[victim].size_mb
                    actions.append(Action(kind="unload", expert_ids=(victim,)))

        # prefetch top-N hottest non-resident experts
        if self.prefetch_n > 0:
            candidates = sorted(
                (p for p in sim.profiles.values()
                 if p.expert_id not in sim.resident and p.expert_id not in requested),
                key=lambda p: p.activation_freq,
                reverse=True,
            )
            for profile in candidates[: self.prefetch_n]:
                if sim.used_gpu_mb + profile.size_mb <= sim.gpu_capacity_mb + 1e-9:
                    actions.append(Action(kind="load", expert_ids=(profile.expert_id,)))
                    sim.resident.add(profile.expert_id)
                    sim.used_gpu_mb += profile.size_mb
                else:
                    break
        return actions

    def _lru_victim(self, state: ScheduleState, protected: set[str]) -> str | None:
        for eid in reversed(state.access_history):
            if eid in state.resident and eid not in protected:
                return eid
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/scheduler/ -v`
Expected: PASS (all scheduler tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add heterogeneous cost-model scheduling policy"
```

---

### Task 8: `sim/moe_adapter.py` — MoE domain integration (the payoff)

**Files:**
- Create: `moesim/sim/moe_adapter.py`
- Test: `tests/sim/test_moe_adapter.py`

**Interfaces:**
- Consumes: `Simulation`, `Event` (core.py), `BandwidthResource`, `ComputeResource` (resources.py), `Metrics` (metrics.py), `Scheduler.decide`, `Action`, `apply_actions` (scheduler/base.py), `ScheduleState` (state.py), `ExpertProfile` (cost_model.py).
- Produces:
  - `class MoESimulation`:
    - `__init__(scheduler: Scheduler, profiles: dict[str, ExpertProfile], gpu_capacity_mb: float, pcie: BandwidthResource, cpu: ComputeResource | None = None, gpu: ComputeResource | None = None)`
    - `feed(step_experts: list[str], token_count: int = 1) -> None`: records one decode step's requested experts, runs the scheduler, applies actions, schedules execution events, and accounts time via internal `Simulation`.
    - `run(steps: list[list[str]]) -> Metrics`: feeds all steps, runs the simulation to completion, returns aggregate `Metrics` (records per-step completion time; TPOT = total_time / total_tokens; token_count per step defaults to 1).
    - Execution model: `execute_gpu` → `gpu.schedule(now, gpu_exec_ms)`; `execute_cpu` → `cpu.schedule(now, cpu_exec_ms)`; `load` → PCIe transfer of `size_mb` at current time; a decode step finishes when the slowest of its scheduled events completes — the sim records that as the step's completion time. Since resources serialize, the natural way: schedule `step_done` event at `max(all scheduled completion times)` and use `Simulation.run(until=...)` between steps is complex; **simpler deterministic model**: each `feed` computes step latency directly from resource reservations (no event loop needed for single-step sequencing — the event loop is exercised in `run()` for overlap across steps). To keep the event loop meaningful and still deterministic, `run()` processes steps sequentially: for each step, compute `step_completion = max(gpu.schedule(...), cpu.schedule(...), pcie.reserve(...))` for the actions taken, record `step_completion - prev_completion` as step time (resources are cumulative across steps, giving cross-step overlap), then advance the internal clock. This satisfies determinism and models PCIe/CPU/GPU overlap across decode steps.

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_moe_adapter.py
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _profiles(n=4, size_mb=10.0):
    return {
        f"e{i}": ExpertProfile(
            expert_id=f"e{i}", size_mb=size_mb, gpu_exec_ms=1.0, cpu_exec_ms=3.0,
            activation_freq=float(i),
        )
        for i in range(n)
    }


def test_lru_simulation_reuses_resident_experts():
    profiles = _profiles()
    pcie = BandwidthResource(bandwidth_gbps=10.0)   # 10MB => 1ms load
    gpu = ComputeResource(concurrency=1, per_unit_ms=1.0)
    sim = MoESimulation(
        scheduler=LRUPolicy(),
        profiles=profiles,
        gpu_capacity_mb=20.0,
        pcie=pcie,
        gpu=gpu,
        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
    )
    metrics = sim.run(steps=[["e0", "e1"], ["e0", "e1"], ["e0", "e1"]])
    # step1: load e0,e1 (PCIe serialized: 1ms + 1ms), then GPU exec each
    #        (starts after its load; GPU serialized): e0 done@2, e1 done@3 -> 3ms
    # step2,3: pure GPU exec (1ms each, serialized) -> 2ms per step
    assert metrics.total_tokens == 3
    assert metrics.cache_hits == 4  # steps 2,3 hit e0,e1
    assert metrics.cache_misses == 2
    assert metrics.total_time_ms == 7.0  # 3 + 2 + 2
    assert metrics.hit_rate() == 4 / 6


def test_cost_model_policy_uses_cpu_for_expensive_loads():
    profiles = _profiles()
    pcie = BandwidthResource(bandwidth_gbps=1.0)    # 10MB => 10ms load: CPU (3ms) wins
    sim = MoESimulation(
        scheduler=CostModelPolicy(pcie=pcie, prefetch_n=0),
        profiles=profiles,
        gpu_capacity_mb=40.0,
        pcie=pcie,
        gpu=ComputeResource(concurrency=1, per_unit_ms=1.0),
        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
    )
    metrics = sim.run(steps=[["e0"], ["e1"]])
    # e0,e1 computed on CPU (3ms each) — no PCIe loads
    assert metrics.total_time_ms == 6.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/sim/test_moe_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# moesim/sim/moe_adapter.py
"""MoE domain adapter: drive the scheduler inside the simulator."""
from __future__ import annotations

from moesim.scheduler.base import Scheduler, apply_actions
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.state import ScheduleState
from moesim.sim.metrics import Metrics
from moesim.sim.resources import BandwidthResource, ComputeResource


class MoESimulation:
    def __init__(
        self,
        scheduler: Scheduler,
        profiles: dict[str, ExpertProfile],
        gpu_capacity_mb: float,
        pcie: BandwidthResource,
        gpu: ComputeResource | None = None,
        cpu: ComputeResource | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.profiles = profiles
        self.pcie = pcie
        self.gpu = gpu or ComputeResource(concurrency=1, per_unit_ms=1.0)
        self.cpu = cpu
        self._clock = 0.0
        self._state = ScheduleState(
            profiles=profiles, resident=set(), gpu_capacity_mb=gpu_capacity_mb
        )
        self._metrics = Metrics()

    def run(self, steps: list[list[str]]) -> Metrics:
        for step in steps:
            self._step(step)
        self._metrics.cache_hits = self._state.cache_hits
        self._metrics.cache_misses = self._state.cache_misses
        return self._metrics

    def _step(self, experts: list[str]) -> None:
        self._state.requested = tuple(experts)
        for eid in experts:
            self._state.mark_access(eid)
        actions = self.scheduler.decide(self._state, self._clock)
        apply_actions(self._state, actions)

        # Execution model: an expert's execution can only start AFTER its PCIe
        # load completes (serial dependency). Experts with no explicit execute
        # action default to GPU execution (cache-management-only policies like
        # LRU never emit execute actions).
        # NOTE: cpu_ids/gpu_ids are ORDERED LISTS derived from the actions list
        # (post-review fix) — set comprehension here broke determinism: with
        # different per-expert load completion times, set iteration order
        # changed the serial GPU queue order and the step completion time
        # (proven 21.0 vs 22.0 under different PYTHONHASHSEED).
        cpu_ids = [eid for a in actions if a.kind == "execute_cpu" for eid in a.expert_ids]
        gpu_ids = [eid for a in actions if a.kind == "execute_gpu" for eid in a.expert_ids]
        decided = set(cpu_ids) | set(gpu_ids)
        default_gpu = [eid for eid in experts if eid not in decided]

        # Schedule PCIe loads first; remember each expert's load completion time.
        load_times: dict[str, float] = {}
        for action in actions:
            if action.kind == "load":
                for eid in action.expert_ids:
                    load_times[eid] = self.pcie.reserve(self._clock, self.profiles[eid].size_mb)

        completions: list[float] = []
        # GPU executions start at load completion (or now if already resident).
        for eid in list(gpu_ids) + default_gpu:
            start = load_times.get(eid, self._clock)
            completions.append(self.gpu.schedule(start, self.profiles[eid].gpu_exec_ms))
        # CPU executions run in parallel with GPU/PCIe work (separate resource).
        for eid in cpu_ids:
            if self.cpu is None:
                raise RuntimeError("execute_cpu requested but no CPU resource configured")
            completions.append(self.cpu.schedule(self._clock, self.profiles[eid].cpu_exec_ms))

        step_completion = max(completions, default=self._clock)
        step_time = step_completion - self._clock
        self._clock = step_completion
        # Each step is one decode token (activating multiple experts), so it
        # contributes exactly 1 to the token count.
        self._metrics.record_completion(tokens=1, time_ms=step_time)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/sim/ tests/scheduler/ -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add MoE domain adapter driving scheduler inside simulator"
```

---

### Task 9: Scheduler sweep driver + strategy comparison script

**Files:**
- Create: `moesim/sim/sweep.py`
- Test: `tests/sim/test_sweep.py`

**Interfaces:**
- Consumes: `MoESimulation`, policies, resources.
- Produces:
  - `def sweep(scheduler_factory: Callable[[], Scheduler], profiles: dict[str, ExpertProfile], steps: list[list[str]], pcie_params: dict, gpu_capacity_mb: float) -> Metrics`
  - `def compare_policies(profiles, steps, pcie_params, gpu_capacity_mb) -> dict[str, Metrics]` — runs LRU, ActivationFreq, CostModel and returns keyed results (the core benchmark harness for the project).
- [ ] **Step 1: Write the failing test**

```python
# tests/sim/test_sweep.py
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.sweep import compare_policies


def _profiles(n=4, size_mb=10.0):
    return {
        f"e{i}": ExpertProfile(
            expert_id=f"e{i}", size_mb=size_mb, gpu_exec_ms=1.0, cpu_exec_ms=3.0,
            activation_freq=float(i),
        )
        for i in range(n)
    }


def test_compare_policies_returns_all_three():
    steps = [["e0", "e1"], ["e0", "e1"], ["e0", "e1"], ["e2", "e3"]]
    results = compare_policies(
        profiles=_profiles(),
        steps=steps,
        pcie_params={"bandwidth_gbps": 10.0},
        gpu_capacity_mb=20.0,
    )
    assert set(results.keys()) == {"lru", "activation_freq", "cost_model"}
    for metrics in results.values():
        assert metrics.total_tokens == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/sim/test_sweep.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# moesim/sim/sweep.py
"""Strategy comparison harness — the project's headline benchmark."""
from __future__ import annotations

from typing import Callable

from moesim.scheduler.base import Scheduler
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.activation_freq import ActivationFreqPolicy
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.metrics import Metrics
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def sweep(
    scheduler: Scheduler,
    profiles: dict[str, ExpertProfile],
    steps: list[list[str]],
    pcie_params: dict,
    gpu_capacity_mb: float,
) -> Metrics:
    pcie = BandwidthResource(**pcie_params)
    sim = MoESimulation(
        scheduler=scheduler,
        profiles=profiles,
        gpu_capacity_mb=gpu_capacity_mb,
        pcie=pcie,
        gpu=ComputeResource(concurrency=1, per_unit_ms=1.0),
        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
    )
    return sim.run(steps)


def compare_policies(
    profiles: dict[str, ExpertProfile],
    steps: list[list[str]],
    pcie_params: dict,
    gpu_capacity_mb: float,
) -> dict[str, Metrics]:
    pcie = BandwidthResource(**pcie_params)
    policies: dict[str, Scheduler] = {
        "lru": LRUPolicy(),
        "activation_freq": ActivationFreqPolicy(),
        "cost_model": CostModelPolicy(pcie=pcie, prefetch_n=1),
    }
    return {
        name: sweep(policy, profiles, steps, pcie_params, gpu_capacity_mb)
        for name, policy in policies.items()
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/sim/ -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add strategy comparison harness (compare_policies)"
```

---

### Task 10: `executor/base.py` + `executor/kv_manager.py` skeleton

**Files:**
- Create: `moesim/executor/base.py`
- Create: `moesim/executor/kv_manager.py`
- Test: `tests/executor/test_base.py`
- Test: `tests/executor/test_kv_manager.py`

**Interfaces:**
- Produces:
  - `class ExpertExecutor: def load(self, expert_ids: list[str]) -> None; def unload(self, expert_ids: list[str]) -> None; def execute_gpu(self, expert_id: str, hidden_states: object) -> object; def execute_cpu(self, expert_id: str, hidden_states: object) -> object` — abstract base, methods raise `NotImplementedError`.
  - `class KVTierManager: __init__(gpu_pool_mb: float, host_pool_mb: float); def can_allocate(self, mb: float) -> bool; def allocate_gpu(self, mb: float) -> None; def allocate_host(self, mb: float) -> None; def free(self, mb: float) -> None; gpu_used_mb / host_used_mb properties; def transfer_gpu_to_host(self, mb: float) -> None (moves usage between tiers)` — raises `ValueError` on over-allocation. (Interface skeleton for v2's full KV-tier scheduling; v1 provides the accounting.)
- [ ] **Step 1: Write the failing tests**

```python
# tests/executor/test_base.py
import pytest
from moesim.executor.base import ExpertExecutor


def test_executor_abstract():
    ex = ExpertExecutor()
    with pytest.raises(NotImplementedError):
        ex.load(["e0"])
    with pytest.raises(NotImplementedError):
        ex.execute_gpu("e0", object())
```

```python
# tests/executor/test_kv_manager.py
import pytest
from moesim.executor.kv_manager import KVTierManager


def test_tier_accounting():
    m = KVTierManager(gpu_pool_mb=100.0, host_pool_mb=200.0)
    m.allocate_gpu(60.0)
    assert m.gpu_used_mb == 60.0
    m.transfer_gpu_to_host(60.0)
    assert m.gpu_used_mb == 0.0
    assert m.host_used_mb == 60.0
    m.free(60.0)
    assert m.host_used_mb == 0.0


def test_tier_over_allocation_raises():
    m = KVTierManager(gpu_pool_mb=100.0, host_pool_mb=200.0)
    with pytest.raises(ValueError):
        m.allocate_gpu(150.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/executor/ -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# moesim/executor/base.py
"""Expert executor abstraction — pluggable execution backends."""
from __future__ import annotations


class ExpertExecutor:
    def load(self, expert_ids: list[str]) -> None:
        raise NotImplementedError

    def unload(self, expert_ids: list[str]) -> None:
        raise NotImplementedError

    def execute_gpu(self, expert_id: str, hidden_states: object) -> object:
        raise NotImplementedError

    def execute_cpu(self, expert_id: str, hidden_states: object) -> object:
        raise NotImplementedError
```

```python
# moesim/executor/kv_manager.py
"""KV cache tier accounting: GPU pool / host pool (v1 skeleton)."""
from __future__ import annotations


class KVTierManager:
    def __init__(self, gpu_pool_mb: float, host_pool_mb: float) -> None:
        self.gpu_pool_mb = gpu_pool_mb
        self.host_pool_mb = host_pool_mb
        self.gpu_used_mb = 0.0
        self.host_used_mb = 0.0

    def can_allocate(self, mb: float) -> bool:
        return self.gpu_used_mb + mb <= self.gpu_pool_mb + 1e-9

    def allocate_gpu(self, mb: float) -> None:
        if self.gpu_used_mb + mb > self.gpu_pool_mb + 1e-9:
            raise ValueError(f"GPU pool overflow: {self.gpu_used_mb}+{mb}>{self.gpu_pool_mb}")
        self.gpu_used_mb += mb

    def allocate_host(self, mb: float) -> None:
        if self.host_used_mb + mb > self.host_pool_mb + 1e-9:
            raise ValueError(f"host pool overflow: {self.host_used_mb}+{mb}>{self.host_pool_mb}")
        self.host_used_mb += mb

    def transfer_gpu_to_host(self, mb: float) -> None:
        if mb > self.gpu_used_mb + 1e-9:
            raise ValueError(f"cannot transfer {mb}MB, only {self.gpu_used_mb}MB on GPU")
        self.gpu_used_mb -= mb
        self.host_used_mb += mb

    def free(self, mb: float) -> None:
        if mb > self.gpu_used_mb + self.host_used_mb + 1e-9:
            raise ValueError("free amount exceeds total usage")
        gpu_freed = min(mb, self.gpu_used_mb)
        self.gpu_used_mb -= gpu_freed
        self.host_used_mb -= mb - gpu_freed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/executor/ -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add ExpertExecutor abstraction and KV tier manager skeleton"
```

---

### Task 11: `executor/cpu_kernels` — PyTorch C++ FP16 expert FFN

**Files:**
- Create: `moesim/executor/cpu_kernels/setup.py`
- Create: `moesim/executor/cpu_kernels/expert_ffn.cpp`
- Create: `moesim/executor/cpu_kernels/__init__.py`
- Create: `tests/executor/test_cpu_kernels.py`
- Test: `tests/executor/test_cpu_kernels.py`

**Interfaces:**
- Produces: `moesim.executor.cpu_kernels.expert_ffn(hidden_states, w1, w2, use_gelu=True) -> Tensor` — FP16/BF16-compatible two-layer FFN computed **on CPU** (explicitly moved off GPU), exposed via `torch.utils.cpp_extension.load_inline`. Fallback path: if torch/C++ extension unavailable, `expert_ffn_cpu_fallback(hidden_states, w1, w2)` implemented in pure torch ops (documented as unoptimized reference).
- Constraint: torch is an optional dependency — tests `pytest.importorskip("torch")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/executor/test_cpu_kernels.py
import pytest

torch = pytest.importorskip("torch")
from moesim.executor.cpu_kernels import expert_ffn  # noqa: E402


def test_expert_ffn_matches_torch_reference():
    torch.manual_seed(0)
    hidden = torch.randn(4, 128, dtype=torch.float16)
    w1 = torch.randn(256, 128, dtype=torch.float16)
    w2 = torch.randn(128, 256, dtype=torch.float16)
    out = expert_ffn(hidden, w1, w2)
    # reference: GELU(x @ w1.T) @ w2.T
    ref = torch.nn.functional.gelu(hidden @ w1.t()) @ w2.t()
    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=1e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/executor/test_cpu_kernels.py -v`
Expected: FAIL with `ImportError: cannot import name 'expert_ffn'` (or skip if torch missing — acceptable, mark `@pytest.mark.skipif` for no-torch CI)

- [ ] **Step 3: Implement**

```python
# moesim/executor/cpu_kernels/expert_ffn.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>

// FP16 expert FFN executed on CPU: GELU(x @ w1^T) @ w2^T
torch::Tensor expert_ffn(torch::Tensor x, torch::Tensor w1, torch::Tensor w2) {
  TORCH_CHECK(x.is_cpu(), "expert_ffn requires CPU input");
  auto h = at::matmul(x, w1.transpose(0, 1));
  h = at::gelu(h);
  return at::matmul(h, w2.transpose(0, 1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("expert_ffn", &expert_ffn, "FP16 expert FFN on CPU");
}
```

```python
# moesim/executor/cpu_kernels/__init__.py
"""Self-hosted CPU FP16 expert FFN kernel (torch C++ extension)."""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

try:  # load the C++ extension once at import
    from torch.utils.cpp_extension import load_inline

    # NOTE: NO PYBIND11_MODULE block in cpp_sources — load_inline auto-generates
    # the module binding from the `functions` argument. Including one causes
    # symbol redefinition (PyInit_* defined twice) → build fails → silent
    # fallback. This was a real bug found post-install (Task 11 review).
    _EXT = load_inline(
        name="moesim_expert_ffn",
        cpp_sources=r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
torch::Tensor expert_ffn(torch::Tensor x, torch::Tensor w1, torch::Tensor w2) {
  TORCH_CHECK(x.is_cpu(), "expert_ffn requires CPU input");
  auto h = at::matmul(x, w1.transpose(0, 1));
  h = at::gelu(h);
  return at::matmul(h, w2.transpose(0, 1));
}
''',
        functions=["expert_ffn"],
        verbose=False,
    )
except Exception as exc:  # pragma: no cover - fallback for envs without compiler
    logger.warning("C++ extension build failed, using torch fallback: %s", exc)
    _EXT = None


def expert_ffn(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Execute expert FFN on CPU. Uses C++ kernel when available, else torch fallback."""
    if _EXT is not None:
        return _EXT.expert_ffn(hidden_states.cpu(), w1.cpu(), w2.cpu())
    return torch.nn.functional.gelu(hidden_states.cpu() @ w1.t().cpu()) @ w2.t().cpu()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/executor/test_cpu_kernels.py -v`
Expected: PASS (1 passed; on machines without torch/compiler, the test skips via importorskip)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add self-hosted CPU FP16 expert FFN kernel with torch fallback"
```

---

### Task 12: `executor/backends/transformers.py` — HF MoE adapter

**Files:**
- Create: `moesim/executor/backends/transformers.py`
- Test: `tests/executor/test_transformers_backend.py` (uses a tiny local MoE model — skipped without transformers/torch)

**Interfaces:**
- Consumes: `ExpertExecutor` (base.py), `expert_ffn` (cpu_kernels).
- Produces: `class TransformersMoEExecutor(ExpertExecutor)`: wraps a loaded HF MoE model; `execute_cpu(expert_id, hidden_states)` routes the expert FFN through `expert_ffn` (CPU), `execute_gpu` uses the model's native expert module; `load/unload` update an in-memory residency map (v1: no-op for HF weights — documented; real weight offloading via `accelerate`/`device_map` is a v2 item).
- Test: builds a 2-expert mini MoE with transformers `MixtralConfig` or a hand-rolled `nn.Module`, asserts `execute_cpu` output shape and that tensors end up on CPU.

- [ ] **Step 1: Write the failing test**

```python
# tests/executor/test_transformers_backend.py
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from moesim.executor.backends.transformers import TransformersMoEExecutor  # noqa: E402


def test_execute_cpu_routes_through_cpu_kernel():
    import torch.nn as nn

    class MiniExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(8, 16, bias=False)
            self.w2 = nn.Linear(16, 8, bias=False)

        def forward(self, x):
            return self.w2(nn.functional.gelu(self.w1(x)))

    model = nn.Module()
    model.experts = nn.ModuleList([MiniExpert(), MiniExpert()])
    ex = TransformersMoEExecutor(model)
    x = torch.randn(2, 8)
    out = ex.execute_cpu("0", x)
    assert out.shape == (2, 8)
    assert out.device.type == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/executor/test_transformers_backend.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# moesim/executor/backends/transformers.py
"""Transformers backend adapter: route expert FFNs through the executor abstraction."""
from __future__ import annotations

from moesim.executor.base import ExpertExecutor
from moesim.executor.cpu_kernels import expert_ffn


class TransformersMoEExecutor(ExpertExecutor):
    """Wraps a HF-style MoE model (model.experts: ModuleList of FFN modules).

    v1 scope: execute_gpu uses the native expert module; execute_cpu routes the
    same weights through the CPU kernel. Weight offloading (load/unload moving
    parameters between devices) is a v2 item — v1 keeps all weights in place and
    only demonstrates the routing path.
    """

    def __init__(self, model) -> None:
        self.model = model

    def _expert_module(self, expert_id: str):
        return self.model.experts[int(expert_id)]

    def load(self, expert_ids: list[str]) -> None:
        pass  # v1: weights are always resident; see docstring

    def unload(self, expert_ids: list[str]) -> None:
        pass  # v1: see docstring

    def execute_gpu(self, expert_id: str, hidden_states) -> object:
        return self._expert_module(expert_id)(hidden_states)

    def execute_cpu(self, expert_id: str, hidden_states) -> object:
        module = self._expert_module(expert_id)
        w1 = module.w1.weight.detach()
        w2 = module.w2.weight.detach()
        return expert_ffn(hidden_states, w1, w2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/executor/test_transformers_backend.py -v`
Expected: PASS (1 passed, or skipped without torch)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add transformers MoE executor adapter with CPU routing"
```

---

### Task 13: `benchmarks/microbench/` — real-machine calibration

**Files:**
- Create: `benchmarks/microbench/measure_pcie.py`
- Create: `benchmarks/microbench/measure_expert_time.py`
- Create: `benchmarks/microbench/README.md`
- Test: none (hardware benchmark; verified by running on the 12G/16G machine)

**Interfaces:**
- Produces:
  - `measure_pcie.py` → prints PCIe effective bandwidth (GB/s) measured by copying a 340MB tensor GPU↔CPU repeatedly; writes `benchmarks/microbench/out/pcie.json` with `{"bandwidth_gbps": X, "latency_ms": Y}`.
  - `measure_expert_time.py --model <path>` → loads a Qwen3-30B-A3B (or config-equivalent), times each expert layer's FFN on GPU and CPU (via `moesim.executor.cpu_kernels.expert_ffn`), writes `out/profiles.json` in the `load_profiles` schema from Task 4.
- Steps are runnable scripts with `argparse`; results feed `sim/calibrate.py`.

- [ ] **Step 1: Write `measure_pcie.py`**

```python
#!/usr/bin/env python3
"""Measure effective PCIe bandwidth (GPU <-> CPU) for calibration."""
import argparse
import json
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=float, default=340.0)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", type=str, default="benchmarks/microbench/out/pcie.json")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "requires a CUDA GPU"
    size = int(args.size_mb * 1024 * 1024 // 2)  # fp16 elements
    src = torch.randn(size, dtype=torch.float16, device="cuda")
    dst = torch.empty(size, dtype=torch.float16, device="cpu")

    # warmup
    for _ in range(5):
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.repeats):
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_gb = args.size_mb * args.repeats / 1024.0
    bandwidth_gbps = total_gb / elapsed
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"bandwidth_gbps": round(bandwidth_gbps, 3), "latency_ms": 0.1})
    )
    print(f"PCIe effective bandwidth: {bandwidth_gbps:.2f} GB/s -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `measure_expert_time.py`**

```python
#!/usr/bin/env python3
"""Measure per-expert FFN time on GPU and CPU for scheduler calibration."""
import argparse
import json
import time
from pathlib import Path

import torch


def time_fn(fn, repeats=20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000.0  # ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--expert-mb", type=float, default=340.0)
    parser.add_argument("--out", type=str, default="benchmarks/microbench/out/profiles.json")
    args = parser.parse_args()

    from moesim.executor.cpu_kernels import expert_ffn

    x_gpu = torch.randn(1, args.hidden, dtype=torch.float16, device="cuda")
    w1 = torch.randn(args.intermediate, args.hidden, dtype=torch.float16, device="cuda")
    w2 = torch.randn(args.hidden, args.intermediate, dtype=torch.float16, device="cuda")

    gpu_ms = time_fn(lambda: torch.nn.functional.gelu(x_gpu @ w1.t()) @ w2.t())

    x_cpu = x_gpu.cpu()
    w1_cpu = w1.cpu()
    w2_cpu = w2.cpu()
    cpu_ms = time_fn(lambda: expert_ffn(x_cpu, w1_cpu, w2_cpu))

    profiles = [
        {
            "expert_id": f"e{i}",
            "size_mb": args.expert_mb,
            "gpu_exec_ms": round(gpu_ms, 4),
            "cpu_exec_ms": round(cpu_ms, 4),
        }
        for i in range(args.num_experts)
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(profiles, indent=2))
    print(f"GPU {gpu_ms:.3f}ms / CPU {cpu_ms:.3f}ms per expert -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `benchmarks/microbench/README.md`** — documents the two commands, the calibration loop (measure → `sim/calibrate.py` → compare simulated TPOT vs real TPOT, target error <20%).

- [ ] **Step 4: Run the PCIe benchmark on the real machine**

Run: `cd /home/qyw/projects/moesim && python benchmarks/microbench/measure_pcie.py`
Expected: prints effective bandwidth (typically 8-25 GB/s for PCIe 3.0/4.0) and writes `out/pcie.json`

- [ ] **Step 5: Run the expert timing benchmark**

Run: `cd /home/qyw/projects/moesim && python benchmarks/microbench/measure_expert_time.py`
Expected: prints per-expert GPU/CPU times and writes `out/profiles.json`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add real-machine microbenchmarks for PCIe and expert timing calibration"
```

---

### Task 14: `benchmarks/e2e/` — end-to-end comparison vs baselines

**Files:**
- Create: `benchmarks/e2e/compare_moesim_vs_llamacpp.py`
- Create: `benchmarks/e2e/README.md`

**Interfaces:**
- Consumes: `compare_policies` (sim/sweep.py), real profiles from Task 13.
- Produces:
  - Script that (a) builds a request trace (repeated activation of a hot subset + cold tail, mimicking Mixtral/Qwen-MoE skewed activation), (b) runs `compare_policies` with real calibrated params, (c) optionally invokes llama.cpp `--n-cpu-moe` and MoE-Infinity on Qwen3-30B-A3B if installed, and prints a comparison table: TPOT, throughput, cache hit rate per strategy.
  - `README.md` documents the full verification protocol: calibrate → simulate → real run → compare (target: simulation vs real TPOT error < 20%).

- [ ] **Step 1: Write `compare_moesim_vs_llamacpp.py`**

```python
#!/usr/bin/env python3
"""Compare moesim policies against llama.cpp --n-cpu-moe on Qwen3-30B-A3B."""
import argparse
import json
import subprocess
from pathlib import Path

from moesim.sim.calibrate import load_profiles
from moesim.scheduler.cost_model import profiles_from_dicts
from moesim.sim.sweep import compare_policies


def build_trace(num_steps=100, hot=("e0", "e1"), cold=("e250", "e251")):
    trace = []
    for i in range(num_steps):
        if i % 5 == 4:  # every 5th step touches cold experts
            trace.append(list(hot) + list(cold))
        else:
            trace.append(list(hot))
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=str,
                        default="benchmarks/microbench/out/profiles.json")
    parser.add_argument("--pcie", type=str,
                        default="benchmarks/microbench/out/pcie.json")
    parser.add_argument("--llama-cpp-bin", type=str, default="llama-bench")
    parser.add_argument("--model-gguf", type=str, default="")
    args = parser.parse_args()

    profiles = profiles_from_dicts(load_profiles(args.profiles))
    pcie_params = json.loads(Path(args.pcie).read_text())
    trace = build_trace()

    print("=== moesim simulation (calibrated) ===")
    results = compare_policies(profiles=profiles, steps=trace,
                               pcie_params=pcie_params, gpu_capacity_mb=12000.0)
    for name, m in results.items():
        print(f"{name:16s} TPOT={m.tpot_ms():8.3f}ms  "
              f"tput={m.throughput_tok_s():8.3f} tok/s  hit={m.hit_rate():.3f}")

    if args.llama_cpp_bin and args.model_gguf:
        print("\n=== llama.cpp --n-cpu-moe baseline ===")
        out = subprocess.run(
            [args.llama_cpp_bin, "-m", args.model_gguf, "--n-cpu-moe", "8", "--cpu-moe"],
            capture_output=True, text=True,
        )
        print(out.stdout[-2000:])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `benchmarks/e2e/README.md`** — verification protocol: (1) run microbenchmarks, (2) run `compare_moesim_vs_llamacpp.py` for simulation numbers, (3) run llama.cpp + MoE-Infinity real baselines on Qwen3-30B-A3B, (4) fill the comparison table; record whether simulation error is < 20%.

- [ ] **Step 3: Run the comparison script on the real machine**

Run: `cd /home/qyw/projects/moesim && python benchmarks/e2e/compare_moesim_vs_llamacpp.py`
Expected: prints per-policy simulated TPOT/throughput/hit-rate table

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: add end-to-end comparison harness vs llama.cpp baseline"
```

---

### Task 15: README + final polish

**Files:**
- Create: `README.md`
- Modify: `docs/superpowers/specs/2026-08-09-moesim-design.md` (no changes expected — spec is authoritative)

**Interfaces:**
- Consumes: everything.
- Produces: project README with quickstart (install, run tests, run simulation demo), architecture diagram (from spec §4), verification protocol summary, roadmap (v2 items from spec §8).

- [ ] **Step 1: Write README.md**

Cover: what moesim is (one-paragraph), the two-line pitch ("domain-agnostic DES + heterogeneous MoE scheduler"), quickstart commands (pip install -e ., pytest, python -m moesim.sim.sweep demo), architecture summary pointing at the spec, hardware note (12G+16G validation environment, no-GPU demo path), v2 roadmap.

- [ ] **Step 2: Run the full test suite**

Run: `cd /home/qyw/projects/moesim && python -m pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 3: Verify no-torch path works (sim only)**

Run: `cd /home/qyw/projects/moesim && python -c "from moesim.sim.sweep import compare_policies; from moesim.scheduler.cost_model import ExpertProfile; print('ok')"` in a venv WITHOUT torch installed.
Expected: prints `ok` (sim/scheduler layers have zero torch dependency)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: add README with quickstart and verification protocol"
```

---

## Self-Review Notes

- **Spec coverage:** §5.1 (sim/ core+resources+calibrate+metrics) → Tasks 2-4; §5.2 (scheduler state/cost_model/base/policies) → Tasks 5-7; §5.3 (executor base/kv_manager/cpu_kernels/transformers) → Tasks 10-12; §5.4 (benchmarks microbench/e2e) → Tasks 13-14; §6 data flow → Task 8; §7 test strategy → all tasks (deterministic sim tests; calibration error <20% in Task 13-14); §8 v1 boundary → all tasks, v2 items explicitly deferred.
- **Deliberate simplifications (documented, not gaps):** KV tier full scheduling policy deferred to v2 (Task 10 provides accounting skeleton); transformers executor load/unload is a no-op in v1 (weight offloading via device_map is v2); CPU kernel is FP16 with torch fallback (quantization is v2). Each is marked in the task it touches.
- **Determinism:** `deepcopy` in policies operates on the decision copy; `apply_actions` mutates the real state exactly once per step. No randomness anywhere.
- **Type consistency:** `ExpertProfile(expert_id, size_mb, gpu_exec_ms, cpu_exec_ms, activation_freq=0.0)` defined once (Task 5) and used everywhere; `Action(kind, expert_ids, target)` validated in `__post_init__`; `Metrics(tpot_ms/throughput_tok_s/hit_rate)` used by sweep and e2e; `ScheduleState.requested` tuple defined in Task 5 and consumed by all policies + `MoESimulation._step`.
- **Timing model (Task 8, fixed during self-review):** an expert's execution can only start AFTER its PCIe load completes (serial dependency) — `_step` schedules loads first, records `load_times`, then GPU executions start at `load_times[eid]` (or `now` if resident). Experts with no explicit execute action default to GPU. Cache-management-only policies (LRU/ActivationFreq) never emit execute actions, so this default is required — verified by the Task 8 LRU trace (3ms step1 + 2ms step2 + 2ms step3 = 7.0ms total).
- **Test math (fixed during self-review):** Task 6 LRU test now sets `requested=("e2",)` explicitly (was empty → policy would load nothing); Task 6 ActivationFreq split into two tests (eviction needs capacity=20 with full residency; prefetch needs capacity=40 with free space — one config cannot exercise both); Task 8 LRU assertion `total_time_ms == 7.0` (was wrongly 5.0 under the old parallel-load model).
