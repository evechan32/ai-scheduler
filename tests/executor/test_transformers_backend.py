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
