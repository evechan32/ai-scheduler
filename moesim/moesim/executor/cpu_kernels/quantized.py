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
