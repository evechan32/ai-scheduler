"""MoE forward hook: replace HF MoE layer forward with scheduler-driven execution."""
from __future__ import annotations

import torch

from moesim.scheduler.state import ScheduleState


class MoEForwardHook:
    """Wrap OlmoeSparseMoeBlock-style MoE layers so expert routing goes through
    the moesim scheduler, which decides per-expert CPU/GPU placement."""

    def __init__(self, executor, scheduler, profiles, pcie, device: str = "cuda") -> None:
        self.executor = executor
        self.scheduler = scheduler
        self.profiles = profiles
        self.pcie = pcie
        self.device = device
        self._originals: dict[int, object] = {}

    def install(self, model) -> None:
        for idx, layer in enumerate(model.model.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "gate"):
                self._originals[idx] = type(mlp).forward
                mlp.forward = self._make_forward(idx, mlp)

    def uninstall(self, model) -> None:
        for idx, layer in enumerate(model.model.layers):
            mlp = getattr(layer, "mlp", None)
            if idx in self._originals:
                mlp.forward = self._originals.pop(idx)

    def _make_forward(self, layer_idx: int, mlp):
        def forward(hidden_states, **kwargs):
            if self.executor is not None:
                # Point the executor at this layer's expert container so its
                # _expert_module() resolves experts for the active layer.
                self.executor.model = mlp
            hidden_states = hidden_states.to(self.device)
            logits = mlp.gate(hidden_states)
            weights = torch.softmax(logits, dim=-1)
            topk = torch.topk(weights, k=min(len(mlp.experts), weights.shape[-1]), dim=-1)
            requested = [str(i) for i in topk.indices.flatten().tolist()]

            resident = set(self.executor.residency.keys()) if self.executor else set()
            state = ScheduleState(
                profiles=self.profiles, resident=resident,
                gpu_capacity_mb=1e9, requested=tuple(requested),
            )
            actions = self.scheduler.decide(state, 0.0) if self.scheduler else []

            out = torch.zeros_like(hidden_states)
            for eid in sorted(set(requested)):
                i = int(eid)
                weight = weights[:, i : i + 1]
                use_cpu = any(a.kind == "execute_cpu" and eid in a.expert_ids for a in actions)
                if use_cpu and self.executor:
                    expert_out = self.executor.execute_cpu(eid, hidden_states)
                else:
                    if self.executor and self.executor.residency.get(eid) != self.device:
                        self.executor.load([eid])
                    if self.executor:
                        expert_out = self.executor.execute_gpu(eid, hidden_states)
                    else:
                        expert_out = mlp.experts[i](hidden_states)
                out = out + weight * expert_out.to(hidden_states.dtype)
            return out

        return forward
