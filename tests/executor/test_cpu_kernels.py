import pytest

torch = pytest.importorskip("torch")
from moesim.executor.cpu_kernels import _EXT, expert_ffn  # noqa: E402


def test_cpp_extension_loaded():
    assert _EXT is not None


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
