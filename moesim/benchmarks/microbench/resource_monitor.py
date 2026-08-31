#!/usr/bin/env python3
"""Real-machine resource utilization monitor — zero-dependency.

Collects during a test run:
- gpu_util (%): nvidia-smi utilization.gpu (SM-busy approximation)
- gpu_mem_util (%): nvidia-smi utilization.memory
- gpu_mem_used / gpu_mem_total (MiB)
- sm_clocks / mem_clocks (MHz)
- cpu_util (%): /proc/stat delta (per-core + aggregate)
- sys_mem_util (%): /proc/meminfo (used / total)

Optionally (NCU): DRAM bandwidth utilization and SM active throughput via
NSight Compute (`dram__throughput.avg.pct_of_peak_sustained_elapsed`,
`sm__throughput.avg.pct_of_peak_sustained_elapsed`), which are the precise
GPU bandwidth-utilization / sm_active metrics nvidia-smi cannot provide.

Optionally (dmon): `nvidia-smi dmon -s u` gives driver-level SM utilization
(`sm` column, ~sm_active) and memory-controller utilization (`mem` column,
~DRAM bandwidth utilization) at 1s granularity. Available in containers where
NCU/CUPTI are blocked.

Metric semantics: `gpu_util` (utilization.gpu) is the driver's SM-busy
approximation; treat `dmon_sm` as the better sm_active proxy.

No pynvml / psutil dependency: nvidia-smi CLI + /proc parsing only.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field


class _ProcCpu:
    def __init__(self) -> None:
        self._last = self._read()

    @staticmethod
    def _read() -> tuple[int, int]:
        with open("/proc/stat", encoding="utf-8") as f:
            line = f.readline()
        parts = [int(x) for x in line.split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        return sum(parts), idle

    def utilization(self) -> float:
        cur = self._read()
        total_delta = cur[0] - self._last[0]
        idle_delta = cur[1] - self._last[1]
        self._last = cur
        if total_delta <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


class _ProcMem:
    @staticmethod
    def utilization() -> float:
        with open("/proc/meminfo", encoding="utf-8") as f:
            fields = {}
            for line in f:
                key, rest = line.split(":", 1)
                fields[key] = int(rest.strip().split()[0])
        total = fields["MemTotal"]
        available = fields["MemAvailable"]
        if total <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * (total - available) / total))


def _nvidia_smi_query(fields: str) -> dict[str, float] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        values = out.stdout.strip().split(",")
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    if len(values) != len(fields.split(",")):
        return None
    result = {}
    for name, value in zip(fields.split(","), values):
        try:
            result[name] = float(value.strip())
        except ValueError:
            result[name] = -1.0
    return result


@dataclass
class DmonSampler:
    samples: list[dict[str, float]] = field(default_factory=list)
    _proc: subprocess.Popen | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    self.samples.append({
                        "dmon_sm": float(parts[1]),
                        "dmon_mem": float(parts[2]),
                    })
                except ValueError:
                    pass

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except ProcessLookupError:
                pass
        self._thread.join(timeout=5.0)
        self._thread = None
        self._proc = None

    def summary(self) -> dict[str, dict[str, float]]:
        result = {}
        for key in ("dmon_sm", "dmon_mem"):
            values = [s[key] for s in self.samples]
            result[key] = ResourceMonitor._agg(values) if values else {"mean": 0.0, "max": 0.0, "p95": 0.0}
        return result


@dataclass
class ResourceMonitor:
    interval_s: float = 0.2
    samples: list[dict[str, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        cpu = _ProcCpu()
        while not self._stop.is_set():
            gpu = _nvidia_smi_query(
                "utilization.gpu,utilization.memory,memory.used,memory.total,"
                "clocks.sm,clocks.mem"
            ) or {}
            gpu.setdefault("utilization.gpu", -1.0)
            gpu.setdefault("utilization.memory", -1.0)
            gpu.setdefault("memory.used", -1.0)
            gpu.setdefault("memory.total", -1.0)
            gpu.setdefault("clocks.sm", -1.0)
            gpu.setdefault("clocks.mem", -1.0)
            self.samples.append({
                "ts": time.time(),
                "gpu_util": gpu["utilization.gpu"],
                "gpu_mem_util": gpu["utilization.memory"],
                "gpu_mem_used_mib": gpu["memory.used"],
                "gpu_mem_total_mib": gpu["memory.total"],
                "sm_clock_mhz": gpu["clocks.sm"],
                "mem_clock_mhz": gpu["clocks.mem"],
                "cpu_util": cpu.utilization(),
                "sys_mem_util": _ProcMem.utilization(),
            })
            time.sleep(self.interval_s)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    @staticmethod
    def _agg(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "max": 0.0, "p95": 0.0}
        sorted_v = sorted(values)
        p95 = sorted_v[min(len(sorted_v) - 1, int(0.95 * len(sorted_v)))]
        return {"mean": statistics.fmean(values), "max": max(values), "p95": p95}

    def summary(self) -> dict[str, dict[str, float]]:
        def series(key: str) -> list[float]:
            return [s[key] for s in self.samples if s[key] >= 0.0]
        return {
            "gpu_util": self._agg(series("gpu_util")),
            "gpu_mem_util": self._agg(series("gpu_mem_util")),
            "gpu_mem_used_mib": self._agg(series("gpu_mem_used_mib")),
            "cpu_util": self._agg(series("cpu_util")),
            "sys_mem_util": self._agg(series("sys_mem_util")),
            "sm_clock_mhz": self._agg(series("sm_clock_mhz")),
            "mem_clock_mhz": self._agg(series("mem_clock_mhz")),
        }

    def to_json(self) -> str:
        return json.dumps({"summary": self.summary(), "samples": self.samples}, indent=2)


NCU_METRICS = (
    "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"
)


def run_ncu(cmd: list[str], timeout_s: int = 600) -> dict[str, dict[str, float]]:
    """Profile `cmd` with NSight Compute; returns per-metric aggregate (%)."""
    try:
        out = subprocess.run(
            ["ncu", "--metrics", NCU_METRICS, "--target-processes", "all"]
            + cmd,
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}
    header = None
    rows = []
    for line in out.stdout.splitlines():
        stripped = line.strip()
        if "dram__throughput" in stripped and "sm__throughput" in stripped:
            header = [c.strip() for c in stripped.split()]
            continue
        if header is not None and stripped and stripped[0].isdigit():
            rows.append([float(c) for c in stripped.split()])
    if not rows:
        return {}
    metrics = [c for c in header if "throughput" in c or "cycles" in c]
    result = {}
    for i, name in enumerate(metrics):
        values = [row[i] for row in rows if row[i] >= 0.0]
        result[name] = ResourceMonitor._agg(values)
    return result


def print_summary(summary: dict[str, dict[str, float]], ncu: dict[str, dict[str, float]] | None = None) -> None:
    print(f"{'metric':28s} {'mean':>8s} {'max':>8s} {'p95':>8s}")
    labels = {
        "gpu_util": "GPU util (%)",
        "gpu_mem_util": "GPU mem util (%)",
        "gpu_mem_used_mib": "GPU mem used (MiB)",
        "cpu_util": "CPU util (%)",
        "sys_mem_util": "Sys mem util (%)",
        "sm_clock_mhz": "SM clock (MHz)",
        "mem_clock_mhz": "Mem clock (MHz)",
    }
    for key, agg in summary.items():
        label = labels.get(key, key)
        print(f"{label:28s} {agg['mean']:8.1f} {agg['max']:8.1f} {agg['p95']:8.1f}")
    if ncu:
        print("\nNSight Compute (per-kernel aggregates, % of peak):")
        for metric, agg in ncu.items():
            print(f"{metric:28s} {agg['mean']:8.1f} {agg['max']:8.1f} {agg['p95']:8.1f}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Sample GPU/CPU/memory utilization for a command.")
    parser.add_argument("cmd", nargs="+", help="command to run while sampling")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--with-ncu", action="store_true", help="also run NCU for bandwidth/sm_active")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    monitor = ResourceMonitor(interval_s=args.interval)
    monitor.start()
    start = time.perf_counter()
    proc = subprocess.run(args.cmd)
    elapsed = time.perf_counter() - start
    monitor.stop()

    ncu = run_ncu(args.cmd) if args.with_ncu else None
    print(f"command: {' '.join(args.cmd)}  (wall {elapsed:.2f}s, exit {proc.returncode})")
    print_summary(monitor.summary(), ncu)
    if args.out:
        payload = {"command": args.cmd, "wall_s": elapsed, "ncu": ncu or {},
                   **monitor.to_json()}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"recorded: {args.out}")
    sys.exit(proc.returncode)
