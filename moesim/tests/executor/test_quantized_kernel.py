# tests/executor/test_quantized_kernel.py
import pytest

torch = pytest.importorskip("torch")

from moesim.executor.cpu_kernels.quantized import (  # noqa: E402
    expert_ffn_int4,
    expert_ffn_int4_gemm,
    expert_ffn_int8,
    _quantize_int4,
)


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


def test_int4_output_shape_and_close_to_fp16():
    torch.manual_seed(3)
    x = torch.randn(2, 16, dtype=torch.float16)
    w1 = torch.randn(32, 16, dtype=torch.float16) - 0.5
    w2 = torch.randn(16, 32, dtype=torch.float16) - 0.5

    ref = torch.nn.functional.gelu(x @ w1.t()) @ w2.t()
    out = expert_ffn_int4(x, w1, w2)

    assert out.shape == ref.shape
    assert out.dtype == torch.float16
    # 相对误差 < 10%（int4 量化误差）
    rel_err = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel_err < 0.10, f"rel err {rel_err:.4f}"


def test_int4_packed_halves_memory():
    torch.manual_seed(4)
    t = torch.randn(32, 16, dtype=torch.float16)
    packed, scale = _quantize_int4(t)

    assert packed.dtype == torch.uint8
    assert packed.numel() == t.numel() // 2

    # 解包（低 nibble 在前）并与原始值比较（相对全量程 amax 的 12.5% 误差内）
    lo = (packed.to(torch.int16) & 0xF) - 8
    hi = ((packed.to(torch.int16) >> 4) & 0xF) - 8
    q = torch.stack([lo, hi], dim=-1).reshape(-1)
    deq = q.float() * scale

    orig = t.reshape(-1).float()
    amax = orig.abs().max()
    rel = (deq - orig).abs() / amax
    assert rel.max() <= 0.125, f"max rel err (vs amax) {rel.max():.4f}"


def test_int4_deterministic():
    torch.manual_seed(5)
    x = torch.randn(2, 16, dtype=torch.float16)
    w1 = torch.randn(32, 16, dtype=torch.float16) - 0.5
    w2 = torch.randn(16, 32, dtype=torch.float16) - 0.5
    a = expert_ffn_int4(x, w1, w2)
    b = expert_ffn_int4(x, w1, w2)
    assert torch.equal(a, b)


def test_int4_gemm_close_to_fp16():
    torch.manual_seed(6)
    x = torch.randn(2, 16, dtype=torch.float16)
    w1 = torch.randn(32, 16, dtype=torch.float16) - 0.5
    w2 = torch.randn(16, 32, dtype=torch.float16) - 0.5
    ref = torch.nn.functional.gelu(x @ w1.t()) @ w2.t()
    out = expert_ffn_int4_gemm(x, w1, w2)
    assert out.shape == ref.shape
    rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel < 0.10, f"rel err {rel:.4f}"
