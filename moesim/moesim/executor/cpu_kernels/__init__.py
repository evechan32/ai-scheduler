"""Self-hosted CPU FP16 expert FFN kernel (torch C++ extension)."""
from __future__ import annotations

import logging

import torch

try:  # load the C++ extension once at import
    from torch.utils.cpp_extension import load_inline

    _EXT = load_inline(
        name="moesim_expert_ffn",
        cpp_sources=r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
torch::Tensor expert_ffn(torch::Tensor x, torch::Tensor w1, torch::Tensor w2) {
  TORCH_CHECK(x.is_cpu(), "expert_ffn requires CPU input");
  auto h = at::matmul(x, w1.transpose(0, 1));
  h = at::gelu(h);
  return at::matmul(h, w2.transpose(0, 1));
}
''',
        functions=["expert_ffn"],
        verbose=False,
    )
except Exception as exc:  # pragma: no cover - fallback for envs without compiler
    _EXT = None
    logging.getLogger(__name__).warning(
        "C++ extension build failed, using torch fallback: %s", exc
    )


def expert_ffn(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Execute expert FFN on CPU. Uses C++ kernel when available, else torch fallback."""
    if _EXT is not None:
        return _EXT.expert_ffn(hidden_states.cpu(), w1.cpu(), w2.cpu())
    return torch.nn.functional.gelu(hidden_states.cpu() @ w1.t().cpu()) @ w2.t().cpu()
