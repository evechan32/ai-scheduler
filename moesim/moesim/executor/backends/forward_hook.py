"""MoE forward hook: replace HF MoE layer forward with scheduler-driven execution."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch

from moesim.scheduler.state import ScheduleState


class MoEForwardHook:
    """Wrap OlmoeSparseMoeBlock-style MoE layers so expert routing goes through
    the moesim scheduler, which decides per-expert CPU/GPU placement.

    Supports two call shapes:
    - 3D input [batch, seq, hidden] (real OlmoeSparseMoeBlock): flattens tokens,
      routes per-token top-k, returns the HF contract tuple (hidden, router_logits).
    - 2D input [tokens, hidden] (mini-MoE test models): keeps the legacy
      all-expert sum and returns a single tensor, matching the simple test model.

    Experts run CONCURRENTLY (CPU experts and GPU experts in parallel): GPU
    kernels launch asynchronously on the current CUDA stream; CPU experts run
    in a thread pool via the CPU kernel. The final merge waits for all experts.
    """

    def __init__(self, executor, scheduler, profiles, pcie, device: str = "cuda",
                 parallel: bool = True) -> None:
        self.executor = executor
        self.scheduler = scheduler
        self.profiles = profiles
        self.pcie = pcie
        self.device = device
        self.parallel = parallel
        self._pool = ThreadPoolExecutor(max_workers=8) if parallel else None
        self._originals: dict[int, object] = {}
        self._in_forward = False
        self.decision_cache: dict[tuple, frozenset] = {}

    def install(self, model) -> None:
        for idx, layer in enumerate(model.model.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "gate"):
                self._originals[idx] = mlp.forward
                mlp.forward = self._make_forward(idx, mlp)

    def uninstall(self, model) -> None:
        for idx, layer in enumerate(model.model.layers):
            mlp = getattr(layer, "mlp", None)
            if idx in self._originals:
                mlp.forward = self._originals.pop(idx)

    def _forward_3d(self, layer_idx: int, mlp, hidden_states: torch.Tensor):
        """OLMoE path: [batch, seq, hidden] -> flatten -> per-token top-k route.

        Mirrors OlmoeSparseMoeBlock.forward's math so the hooked output matches
        the original within float tolerance.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)  # [N, hidden]
        router_logits = mlp.gate(flat)  # [N, n_experts]
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float)
        top_k = min(getattr(mlp, "top_k", len(mlp.experts)), len(mlp.experts))
        top_weights, top_idx = torch.topk(routing_weights, k=top_k, dim=-1)
        top_weights = top_weights.to(flat.dtype)  # [N, top_k]

        requested = sorted({str(i) for i in top_idx.flatten().tolist()})
        cache_key = (layer_idx, tuple(sorted(requested)))
        if self._in_forward and cache_key in self.decision_cache:
            cpu_experts = self.decision_cache[cache_key]
        else:
            resident = set(self.executor.residency.keys()) if self.executor else set()
            state = ScheduleState(
                profiles=self.profiles, resident=resident,
                gpu_capacity_mb=1e9, requested=tuple(requested),
            )
            actions = self.scheduler.decide(state, 0.0) if self.scheduler else []
            cpu_experts = frozenset(
                eid for a in actions if a.kind == "execute_cpu" for eid in a.expert_ids
            )
            if self._in_forward:
                self.decision_cache[cache_key] = cpu_experts

        final = torch.zeros_like(flat)

        def run_expert(i_str: str):
            """Execute one expert (CPU or GPU) for its routed tokens; returns
            (token_mask_indices, weighted_out) to merge into final."""
            i = int(i_str)
            use_cpu = i_str in cpu_experts
            token_mask = (top_idx == i).any(dim=-1)  # [N]
            if not token_mask.any():
                return None
            tokens = flat[token_mask]
            if use_cpu and self.executor:
                expert_out = self.executor.execute_cpu(i_str, tokens)
            else:
                if self.executor and self.executor.residency.get(i_str) != self.device:
                    self.executor.load([i_str])
                if self.executor:
                    expert_out = self.executor.execute_gpu(i_str, tokens)
                else:
                    expert_out = mlp.experts[i](tokens)
            expert_out = expert_out.to(self.device)  # CPU experts return CPU tensors
            pos = (top_idx == i).float()  # [N, top_k]
            per_token_w = (top_weights * pos).sum(dim=-1)[token_mask]  # [n_routed]
            return token_mask, per_token_w.unsqueeze(-1) * expert_out.to(flat.dtype)

        if self.parallel and len(requested) > 1:
            futures = [self._pool.submit(run_expert, i_str) for i_str in requested]
            for fut in futures:
                res = fut.result()
                if res is not None:
                    token_mask, contrib = res
                    final[token_mask] += contrib
        else:
            for i_str in requested:
                res = run_expert(i_str)
                if res is not None:
                    token_mask, contrib = res
                    final[token_mask] += contrib

        final = final.view(batch_size, seq_len, hidden_dim)
        return final, router_logits

    def _forward_2d(self, layer_idx: int, mlp, hidden_states: torch.Tensor):
        """Legacy path for simple test models: all-expert sum, single tensor out."""
        logits = mlp.gate(hidden_states)
        weights = torch.softmax(logits, dim=-1)
        topk = torch.topk(weights, k=min(len(mlp.experts), weights.shape[-1]), dim=-1)
        requested = [str(i) for i in topk.indices.flatten().tolist()]

        cache_key = (layer_idx, tuple(sorted(requested)))
        if self._in_forward and cache_key in self.decision_cache:
            cpu_experts = self.decision_cache[cache_key]
        else:
            resident = set(self.executor.residency.keys()) if self.executor else set()
            state = ScheduleState(
                profiles=self.profiles, resident=resident,
                gpu_capacity_mb=1e9, requested=tuple(requested),
            )
            actions = self.scheduler.decide(state, 0.0) if self.scheduler else []
            cpu_experts = frozenset(
                eid for a in actions if a.kind == "execute_cpu" for eid in a.expert_ids
            )
            if self._in_forward:
                self.decision_cache[cache_key] = cpu_experts

        out = torch.zeros_like(hidden_states)

        def run_expert(eid: str):
            i = int(eid)
            weight = weights[:, i : i + 1]
            use_cpu = eid in cpu_experts
            if use_cpu and self.executor:
                expert_out = self.executor.execute_cpu(eid, hidden_states)
            else:
                if self.executor and self.executor.residency.get(eid) != self.device:
                    self.executor.load([eid])
                if self.executor:
                    expert_out = self.executor.execute_gpu(eid, hidden_states)
                else:
                    expert_out = mlp.experts[i](hidden_states)
            expert_out = expert_out.to(self.device)
            return weight * expert_out.to(hidden_states.dtype)

        if self.parallel and len(set(requested)) > 1:
            futures = [self._pool.submit(run_expert, eid) for eid in sorted(set(requested))]
            for fut in futures:
                out = out + fut.result()
        else:
            for eid in sorted(set(requested)):
                out = out + run_expert(eid)
        return out

    def _make_forward(self, layer_idx: int, mlp):
        def forward(hidden_states, **kwargs):
            outer = not self._in_forward
            if outer:
                self._in_forward = True
                self.decision_cache = {}
            try:
                if self.executor is not None:
                    self.executor.model = mlp
                    # Sync residency to THIS layer's actual expert placement.
                    # residency is shared across layers, so without this, decide()
                    # may see layer-0 placement and wrongly treat layer-1 experts
                    # as resident (cross-layer stale-state bug in mixed execution).
                    if self.executor.residency:
                        for eid in list(self.executor.residency):
                            i = int(eid)
                            if i < len(mlp.experts):
                                params = list(mlp.experts[i].parameters())
                                if params:
                                    actual = params[0].device.type
                                    if actual != self.executor.residency[eid]:
                                        self.executor.residency[eid] = actual
                hidden_states = hidden_states.to(self.device)
                if hidden_states.dim() == 3:
                    return self._forward_3d(layer_idx, mlp, hidden_states)
                return self._forward_2d(layer_idx, mlp, hidden_states)
            finally:
                if outer:
                    self._in_forward = False
                    self.decision_cache = {}

        return forward
