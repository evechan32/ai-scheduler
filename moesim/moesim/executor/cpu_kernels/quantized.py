# moesim/executor/cpu_kernels/quantized.py
"""INT8 quantized expert FFN for CPU (v2 quantization, INT4 later)."""
from __future__ import annotations

import torch


def _quantize_symmetric(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric INT8 quantization. Returns (int8_tensor, scale)."""
    amax = t.float().abs().max()
    if amax == 0:
        return torch.zeros_like(t, dtype=torch.int8), torch.tensor(1.0)
    scale = amax / 127.0
    q = torch.clamp(torch.round(t.float() / scale), -127, 127).to(torch.int8)
    return q, scale


def expert_ffn_int8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """INT8-weight expert FFN on CPU: x -> gelu(x @ w1_t) @ w2_t with int8 weights.

    Weights are quantized per-tensor to int8; activations stay fp16; matmuls use
    fp16 activations against int8 weights cast back to fp16 (accuracy-first v2
    quantization; real int8 gemm kernels are a follow-up).
    """
    q1, s1 = _quantize_symmetric(w1)
    q2, s2 = _quantize_symmetric(w2)
    w1_r = q1.float().to(hidden_states.dtype) * s1
    w2_r = q2.float().to(hidden_states.dtype) * s2
    h = torch.nn.functional.gelu(hidden_states @ w1_r.t())
    return h @ w2_r.t()


def _quantize_int4(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric INT4 quantization, packed 2-per-byte (low nibble first).

    Signed 4-bit range is [-8, 7]. Each quantized value `v` is stored as an
    unsigned nibble `v + 8`; two nibbles share a byte as
    ``byte = (v0 & 0xF) | ((v1 & 0xF) << 4)``. Unpacking reverses this with
    ``val = ((byte >> 4*n) & 0xF) - 8``. The packed tensor holds
    ``t.numel() // 2`` elements (last element zero-padded when odd).

    Returns (packed_uint8_tensor, scale).
    """
    amax = t.float().abs().max()
    if amax == 0:
        q = torch.zeros(t.numel(), dtype=torch.int8)
        scale = torch.tensor(1.0)
    else:
        scale = amax / 7.0
        q = torch.clamp(torch.round(t.float() / scale), -8, 7).to(torch.int8)
    flat = q.reshape(-1)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8)])
    u = (flat + 8).to(torch.uint8)
    packed = (u[0::2] & 0xF) | ((u[1::2] & 0xF) << 4)
    return packed, scale


def expert_ffn_int4(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """INT4-weight expert FFN on CPU: x -> gelu(x @ w1_t) @ w2_t with packed int4 weights.

    Weights are quantized per-tensor to int4 and packed 2-per-byte; they are
    unpacked and dequantized back to fp16 before matmul (accuracy-first; a real
    int4 gemm kernel is a follow-up).
    """
    p1, s1 = _quantize_int4(w1)
    p2, s2 = _quantize_int4(w2)

    lo1 = (p1.to(torch.int16) & 0xF) - 8
    hi1 = ((p1.to(torch.int16) >> 4) & 0xF) - 8
    q1 = torch.stack([lo1, hi1], dim=-1).reshape(-1)[: w1.numel()].reshape(w1.shape)
    lo2 = (p2.to(torch.int16) & 0xF) - 8
    hi2 = ((p2.to(torch.int16) >> 4) & 0xF) - 8
    q2 = torch.stack([lo2, hi2], dim=-1).reshape(-1)[: w2.numel()].reshape(w2.shape)

    w1_r = q1.float().to(hidden_states.dtype) * s1
    w2_r = q2.float().to(hidden_states.dtype) * s2
    h = torch.nn.functional.gelu(hidden_states @ w1_r.t())
    return h @ w2_r.t()


def _unpack_int4_to_int8(packed: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Unpack 2-per-byte packed int4 into an int8 tensor of ``shape`` (values -8..7)."""
    lo = (packed.to(torch.int16) & 0xF) - 8
    hi = ((packed.to(torch.int16) >> 4) & 0xF) - 8
    q = torch.stack([lo, hi], dim=-1).reshape(-1)[: int(shape.numel())].reshape(shape)
    return q.to(torch.int8)


def _quantize_int4_rowwise(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row asymmetric INT4 quantization, packed 2-per-byte (low nibble first).

    Each row is mapped onto the full signed nibble range [-8, 7] via its own
    (min, max) so shifted weight distributions keep full resolution. Returns
    ``(packed_uint8, scale[row], bias[row])`` with dequantization
    ``w ~ scale * q + bias`` (q in -8..7, unpacked with ``_unpack_int4_to_int8``).
    """
    a = t.float()
    tmin = a.min(dim=-1, keepdim=True).values
    tmax = a.max(dim=-1, keepdim=True).values
    scale = (tmax - tmin).clamp(min=1e-6) / 15.0
    q = torch.clamp(torch.round((a - tmin) / scale) - 8, -8, 7).to(torch.int8)
    bias = (tmin - (-8.0) * scale).reshape(-1)

    flat = q.reshape(-1)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8)])
    u = (flat + 8).to(torch.uint8)
    packed = (u[0::2] & 0xF) | ((u[1::2] & 0xF) << 4)
    return packed, scale.reshape(-1), bias


def _int8_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """int8 x int8 -> int32 matmul, zero-padding the contracting dim to 16.

    Uses ``torch._int_mm`` when available (CPU and CUDA), otherwise falls back
    to an int16 cast + integer matmul. Padding with zeros is exact: the extra
    rows/columns contribute nothing to the dot product.
    """
    k = a.shape[-1]
    k_pad = (k + 15) // 16 * 16
    if k_pad > k:
        a = torch.nn.functional.pad(a, (0, k_pad - k))
        b = torch.nn.functional.pad(b, (0, 0, 0, k_pad - k))
    if hasattr(torch, "_int_mm"):
        return torch._int_mm(a, b)
    return a.to(torch.int16) @ b.to(torch.int16)


def _quantize_act_int8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric INT8 quantization of activations. Returns (int8, scale)."""
    amax = t.float().abs().max().clamp(min=1e-6)
    scale = amax / 127.0
    q = torch.clamp(torch.round(t.float() / scale), -127, 127).to(torch.int8)
    return q, scale


def expert_ffn_int4_gemm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    scale_act: float = 1.0,
    use_int_gemm: bool = True,
) -> torch.Tensor:
    """INT4-weight expert FFN with a real int8 gemm path: x -> gelu(x @ w1_t) @ w2_t.

    Weights are quantized per-row to int4 (asymmetric, packed 2-per-byte) and
    unpacked to int8 (values -8..7); activations are quantized per-tensor to
    int8. Both matmuls run as int8 x int8 -> int32 via ``torch._int_mm`` and are
    dequantized with their scales plus the per-row bias correction term.

    Set ``use_int_gemm=False`` to fall back to the dequant-to-fp16 path of
    ``expert_ffn_int4`` (accuracy-first comparison baseline).
    """
    p1, s1, b1 = _quantize_int4_rowwise(w1)
    p2, s2, b2 = _quantize_int4_rowwise(w2)
    w1_i8 = _unpack_int4_to_int8(p1, w1.shape)
    w2_i8 = _unpack_int4_to_int8(p2, w2.shape)

    if not use_int_gemm:
        w1_r = (w1_i8.float() * s1[:, None] + b1[:, None]).to(hidden_states.dtype)
        w2_r = (w2_i8.float() * s2[:, None] + b2[:, None]).to(hidden_states.dtype)
        h = torch.nn.functional.gelu(hidden_states @ w1_r.t())
        return (h @ w2_r.t()).to(hidden_states.dtype)

    x_i8, x_scale = _quantize_act_int8(hidden_states * scale_act)
    x = hidden_states.float() * scale_act
    h1 = _int8_gemm(x_i8, w1_i8.t()).float() * (x_scale * s1[None, :])
    h1 = h1 + b1[None, :] * x.sum(dim=-1, keepdim=True)
    h = torch.nn.functional.gelu(h1)
    h_i8, h_scale = _quantize_act_int8(h)
    out = _int8_gemm(h_i8, w2_i8.t()).float() * (h_scale * s2[None, :])
    out = out + b2[None, :] * h.sum(dim=-1, keepdim=True)
    return out.to(hidden_states.dtype)
