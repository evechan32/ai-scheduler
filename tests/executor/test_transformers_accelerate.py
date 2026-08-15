# tests/executor/test_transformers_accelerate.py
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


def test_accelerate_flag_requires_package():
    pytest.importorskip("accelerate")
    ex = TransformersMoEExecutor(_make_model(), use_accelerate=True)
    assert ex.uses_accelerate is True


def test_accelerate_flag_raises_when_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "accelerate", None)
    with pytest.raises(ImportError, match="pip install accelerate"):
        TransformersMoEExecutor(_make_model(), use_accelerate=True)


def test_accelerate_load_unload_moves_weights():
    pytest.importorskip("accelerate")
    torch.cuda.is_available() or pytest.skip("no CUDA")
    ex = TransformersMoEExecutor(_make_model(), use_accelerate=True)
    ex.load(["0"])
    assert ex.model.experts[0].w1.weight.device.type == "cuda"
    assert ex.residency["0"] == "cuda"
    ex.unload(["0"])
    assert ex.model.experts[0].w1.weight.device.type == "cpu"
    assert ex.residency["0"] == "cpu"


def test_default_path_unchanged():
    ex = TransformersMoEExecutor(_make_model(), use_accelerate=False)
    assert ex.uses_accelerate is False
    torch.cuda.is_available() or pytest.skip("no CUDA")
    ex.load(["0"])
    assert ex.model.experts[0].w1.weight.device.type == "cuda"
    assert ex.residency["0"] == "cuda"
    ex.unload(["0"])
    assert ex.model.experts[0].w1.weight.device.type == "cpu"
    assert ex.residency["0"] == "cpu"
