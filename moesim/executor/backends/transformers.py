"""Transformers backend adapter: route expert FFNs through the executor abstraction."""
from __future__ import annotations

from moesim.executor.base import ExpertExecutor
from moesim.executor.cpu_kernels import expert_ffn


class TransformersMoEExecutor(ExpertExecutor):
    """Wraps a HF-style MoE model (model.experts: ModuleList of FFN modules).

    load/unload move expert parameters between GPU and CPU for real
    (v2: weight offloading). execute_gpu auto-loads; execute_cpu runs the
    weights on CPU via the moesim CPU kernel.
    """

    def __init__(self, model, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        self.residency: dict[str, str] = {}

    def _expert_module(self, expert_id: str):
        return self.model.experts[int(expert_id)]

    def _to(self, module, target: str) -> None:
        module.to(target)

    def load(self, expert_ids: list[str]) -> None:
        for eid in expert_ids:
            self._to(self._expert_module(eid), self.device)
            self.residency[eid] = self.device

    def unload(self, expert_ids: list[str]) -> None:
        for eid in expert_ids:
            self._to(self._expert_module(eid), "cpu")
            self.residency[eid] = "cpu"

    def execute_gpu(self, expert_id: str, hidden_states) -> object:
        if self.residency.get(expert_id) != self.device:
            self.load([expert_id])
        return self._expert_module(expert_id)(hidden_states.to(self.device))

    def execute_cpu(self, expert_id: str, hidden_states) -> object:
        module = self._expert_module(expert_id)
        if self.residency.get(expert_id) != "cpu":
            self.unload([expert_id])
        w1 = module.w1.weight.detach()
        w2 = module.w2.weight.detach()
        return expert_ffn(hidden_states, w1, w2)
