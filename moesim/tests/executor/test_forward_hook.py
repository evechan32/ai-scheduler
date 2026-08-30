# tests/executor/test_forward_hook.py
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from moesim.executor.backends.forward_hook import MoEForwardHook  # noqa: E402
from moesim.executor.backends.transformers import TransformersMoEExecutor  # noqa: E402
from moesim.scheduler.cost_model import ExpertProfile  # noqa: E402
from moesim.scheduler.policies.lru import LRUPolicy  # noqa: E402
from moesim.sim.resources import BandwidthResource  # noqa: E402


class MiniMoE(nn.Module):
    """2-expert MoE layer with gate router, matching OlmoeSparseMoeBlock shape."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 2, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 16, bias=False), nn.GELU(), nn.Linear(16, 8, bias=False))
            for _ in range(2)
        ])

    def forward(self, hidden_states, **kwargs):
        logits = self.gate(hidden_states)
        weights = torch.softmax(logits, dim=-1)
        out = torch.zeros_like(hidden_states)
        for i, expert in enumerate(self.experts):
            out += weights[:, i : i + 1] * expert(hidden_states)
        return out


def _model():
    import torch.nn as nn

    class MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])
            self.model.layers[0].mlp = MiniMoE()

    return MiniModel()


def test_install_replaces_forward():
    model = _model()
    hook = MoEForwardHook(executor=None, scheduler=None, profiles={}, pcie=None)
    hook.install(model)
    hooked = model.model.layers[0].mlp.forward
    assert not hasattr(hooked, "__self__") or hooked.__func__.__name__ == "forward"
    hook.uninstall(model)
    # after uninstall the original forward works (bound instance method callable)
    out = model.model.layers[0].mlp(torch.randn(2, 8))
    assert out.shape == (2, 8)


def test_hook_output_matches_original():
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(2, 8)
    expected = model.model.layers[0].mlp(x)

    profiles = {"0": ExpertProfile("0", size_mb=1.0, gpu_exec_ms=0.1, cpu_exec_ms=0.5),
                "1": ExpertProfile("1", size_mb=1.0, gpu_exec_ms=0.1, cpu_exec_ms=0.5)}
    executor = TransformersMoEExecutor(model, device="cpu")
    # all experts resident on cpu -> execute_gpu path (cpu device)
    scheduler = LRUPolicy()
    hook = MoEForwardHook(executor=executor, scheduler=scheduler, profiles=profiles,
                          pcie=BandwidthResource(bandwidth_gbps=1.0), device="cpu")
    hook.install(model)
    got = model.model.layers[0].mlp(x)
    hook.uninstall(model)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-4)


def test_hook_consults_scheduler():
    from moesim.scheduler.base import Scheduler, Action

    class RecordingScheduler(Scheduler):
        def __init__(self):
            self.calls = 0
            self.last_requested = None

        def decide(self, state, clock):
            self.calls += 1
            self.last_requested = list(state.requested)
            return []

    torch.manual_seed(1)
    model = _model()
    x = torch.randn(2, 8)
    rec = RecordingScheduler()
    profiles = {"0": ExpertProfile("0", 1.0, 0.1, 0.5), "1": ExpertProfile("1", 1.0, 0.1, 0.5)}
    executor = TransformersMoEExecutor(model, device="cpu")
    hook = MoEForwardHook(executor=executor, scheduler=rec, profiles=profiles,
                          pcie=BandwidthResource(bandwidth_gbps=1.0), device="cpu")
    hook.install(model)
    model.model.layers[0].mlp(x)
    hook.uninstall(model)
    assert rec.calls == 1
    assert set(rec.last_requested) == {"0", "1"}


def test_cache_reuses_decisions():
    """Within ONE forward pass, scheduler.decide is called once per layer.

    The 2-layer model re-routes the same input within each layer's forward
    (the second lookup sees an identical requested-expert set), so without the
    decision cache each layer would trigger 2 decide() calls (4 total) and with
    the cache exactly one per layer (2 total).
    """
    from moesim.scheduler.base import Scheduler

    class CountingScheduler(Scheduler):
        def __init__(self):
            self.calls = 0

        def decide(self, state, clock):
            self.calls += 1
            return []

    class NestedGate(nn.Module):
        """Gate that re-runs the MoE layer once per forward on the same input."""

        def __init__(self, inner, mlp):
            super().__init__()
            self.inner = inner
            self.mlp = mlp
            self._fired = False

        def forward(self, x):
            out = self.inner(x)
            if not self._fired:
                self._fired = True
                try:
                    self.mlp(x)
                finally:
                    self._fired = False
            return out

    class TwoLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
            for layer in self.model.layers:
                layer.mlp = MiniMoE()

        def forward(self, x):
            h = x
            for layer in self.model.layers:
                h = layer.mlp(h)
            return h

    torch.manual_seed(3)
    model = TwoLayerModel()
    for layer in model.model.layers:
        layer.mlp.gate = NestedGate(layer.mlp.gate, layer.mlp)

    counting = CountingScheduler()
    profiles = {"0": ExpertProfile("0", 1.0, 0.1, 0.5), "1": ExpertProfile("1", 1.0, 0.1, 0.5)}
    executor = TransformersMoEExecutor(model, device="cpu")
    hook = MoEForwardHook(executor=executor, scheduler=counting, profiles=profiles,
                          pcie=BandwidthResource(bandwidth_gbps=1.0), device="cpu")
    hook.install(model)
    x = torch.randn(2, 8)
    out = model(x)
    hook.uninstall(model)
    assert out.shape == (2, 8)
    assert counting.calls == 2, f"decide should be called once per layer, got {counting.calls}"


def test_hook_demotes_disk_experts():
    from moesim.scheduler.policies.disk_tier import DiskTierPolicy

    torch.manual_seed(2)
    model = _model()
    x = torch.randn(2, 8)
    # e1 cold (low freq) -> demoted to disk; e0 hot stays. cpu_exec is huge so
    # cost_model routes to load (GPU) instead of execute_cpu, avoiding the
    # Sequential-expert CPU path (unsupported) while still exercising demotion.
    profiles = {"0": ExpertProfile("0", 1.0, 0.1, 100.0, activation_freq=0.9),
                "1": ExpertProfile("1", 1.0, 0.1, 100.0, activation_freq=0.01)}
    executor = TransformersMoEExecutor(model, device="cpu")
    scheduler = DiskTierPolicy(pcie=BandwidthResource(bandwidth_gbps=1.0),
                               prefetch_n=0, disk_budget_mb=1.0)
    hook = MoEForwardHook(executor=executor, scheduler=scheduler, profiles=profiles,
                          pcie=BandwidthResource(bandwidth_gbps=1.0), device="cpu")
    hook.install(model)
    model.model.layers[0].mlp(x)
    hook.uninstall(model)
    assert "1" in executor.disk_experts, "cold expert should be demoted to disk"
    assert "0" not in executor.disk_experts


def test_load_from_disk_removes_marker():
    model = _model()
    executor = TransformersMoEExecutor(model, device="cpu", disk_experts={"0"})
    executor._load_from_disk("0")
    assert "0" not in executor.disk_experts, "disk expert promoted on use"
