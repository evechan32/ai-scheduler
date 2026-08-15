# tests/executor/test_quantized_kernel.py
import pytest

torch = pytest.importorskip("torch")

from moesim.executor.cpu_kernels.quantized import expert_ffn_int8  # noqa: E402


def test_int8_output_shape_and_close_to_fp16():
    torch.manual_seed(0)
    x = torch.randn(2, 16, dtype=torch.float16)
    w1 = torch.randn(32, 16, dtype=torch.float16) - 0.5
    w2 = torch.randn(16, 32, dtype=torch.float16) - 0.5

    ref = torch.nn.functional.gelu(x @ w1.t()) @ w2.t()
    out = expert_ffn_int8(x, w1, w2)

    assert out.shape == ref.shape
    assert out.dtype == torch.float16
    # 相对误差 < 5%（int8 量化误差）
    rel_err = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel_err < 0.05, f"rel err {rel_err:.4f}"


def test_int8_deterministic():
    torch.manual_seed(1)
    x = torch.randn(2, 16, dtype=torch.float16)
    w1 = torch.randn(32, 16, dtype=torch.float16) - 0.5
    w2 = torch.randn(16, 32, dtype=torch.float16) - 0.5
    a = expert_ffn_int8(x, w1, w2)
    b = expert_ffn_int8(x, w1, w2)
    assert torch.equal(a, b)
