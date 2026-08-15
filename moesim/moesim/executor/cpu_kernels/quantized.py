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
