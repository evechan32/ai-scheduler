"""Transformers backend adapter: route expert FFNs through the executor abstraction."""
from __future__ import annotations

from moesim.executor.base import ExpertExecutor
from moesim.executor.cpu_kernels import expert_ffn


class TransformersMoEExecutor(ExpertExecutor):
    """Wraps a HF-style MoE model (model.experts: ModuleList of FFN modules).

    v1 scope: execute_gpu uses the native expert module; execute_cpu routes the
    same weights through the CPU kernel. Weight offloading (load/unload moving
    parameters between devices) is a v2 item — v1 keeps all weights in place and
    only demonstrates the routing path.
    """

    def __init__(self, model) -> None:
        self.model = model

    def _expert_module(self, expert_id: str):
        return self.model.experts[int(expert_id)]

    def load(self, expert_ids: list[str]) -> None:
        pass  # v1: weights are always resident; see docstring

    def unload(self, expert_ids: list[str]) -> None:
        pass  # v1: see docstring

    def execute_gpu(self, expert_id: str, hidden_states) -> object:
        return self._expert_module(expert_id)(hidden_states)

    def execute_cpu(self, expert_id: str, hidden_states) -> object:
        module = self._expert_module(expert_id)
        w1 = module.w1.weight.detach()
        w2 = module.w2.weight.detach()
        return expert_ffn(hidden_states, w1, w2)
