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
    original = MiniMoE.forward
    assert model.model.layers[0].mlp.forward is not original
    hook.uninstall(model)
    assert model.model.layers[0].mlp.forward is original


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
