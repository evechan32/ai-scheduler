# tests/executor/test_transformers_offload.py
import pytest

torch = pytest.importorskip("torch")

from moesim.executor.backends.transformers import TransformersMoEExecutor  # noqa: E402


def _make_model():
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
    return model


def test_load_moves_weights_to_gpu():
    torch.cuda.is_available() or pytest.skip("no CUDA")
    ex = TransformersMoEExecutor(_make_model())
    ex.load(["0"])
    assert ex.model.experts[0].w1.weight.device.type == "cuda"
    assert ex.residency["0"] == "cuda"


def test_unload_moves_weights_back_to_cpu():
    torch.cuda.is_available() or pytest.skip("no CUDA")
    ex = TransformersMoEExecutor(_make_model())
    ex.load(["0"])
    ex.unload(["0"])
    assert ex.model.experts[0].w1.weight.device.type == "cpu"
    assert ex.residency["0"] == "cpu"


def test_execute_gpu_auto_loads_and_output_on_gpu():
    torch.cuda.is_available() or pytest.skip("no CUDA")
    ex = TransformersMoEExecutor(_make_model())
    x = torch.randn(2, 8)
    out = ex.execute_gpu("1", x)
    assert out.device.type == "cuda"
    assert ex.residency["1"] == "cuda"


def test_execute_cpu_runs_on_cpu_and_output_on_cpu():
    ex = TransformersMoEExecutor(_make_model())
    x = torch.randn(2, 8)
    out = ex.execute_cpu("0", x)
    assert out.device.type == "cpu"
    assert ex.residency["0"] == "cpu"
