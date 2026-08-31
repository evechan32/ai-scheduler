#!/usr/bin/env python3
"""Step 2: patch the model so experts are per-expert modules (not fused).

The whole point: vLLM's CPU offload matches parameter-name segments
(`f".{param}." in f".{name}."`), but its FusedMoE stacks all experts into
`experts.w13_weight` / `experts.w2_weight`. This script replaces the MoE layer
with a per-expert `nn.ModuleList`, producing parameter names like
`experts.0.w1` / `experts.1.w2`, so `cpu_offload_params` can offload individual
cold experts (from Step 1's offload_plan.json).

This is a SKELETON: the exact vLLM custom-layer API (registering the model,
weight_loader, SupportsLoRA/PP) must follow vllm/model_executor/models/olmoe.py.
The critical piece is the per-expert parameter naming below.
"""
from __future__ import annotations

import torch
from torch import nn


class PerExpertMoE(nn.Module):
    """Non-fused MoE: one Linear per expert, so parameter names are per-expert.

    Parameter naming contract (must match cpu_offload_params segment matching):
        experts.{j}.w1  ->  gate+up projection of expert j
        experts.{j}.w2  ->  down projection of expert j
    """

    def __init__(self, num_experts: int, hidden: int, intermediate: int,
                 top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                "w1": nn.Linear(hidden, 2 * intermediate, bias=False),
                "w2": nn.Linear(intermediate, hidden, bias=False),
            })
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        # Router: select top_k experts per token.
        logits = self.gate(hidden_states)
        topk_weights, topk_ids = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_weights, dim=-1)

        out = torch.zeros_like(hidden_states)
        for k in range(self.top_k):
            expert_ids = topk_ids[..., k]
            weight = topk_weights[..., k].unsqueeze(-1)
            for j in range(self.num_experts):
                mask = (expert_ids == j)
                if not mask.any():
                    continue
                tok = hidden_states[mask]
                x = self.experts[j]["w1"](tok)
                gate, up = x.chunk(2, dim=-1)
                y = torch.nn.functional.silu(gate) * up
                out[mask] += weight[mask] * self.experts[j]["w2"](y)
        return out

    def load_hf_expert_weights(self, layer_prefix: str, state_dict: dict):
        """Map HF per-expert tensors (experts.{j}.gate_proj/up_proj/down_proj)
        into our w1/w2 layout. gate_proj+up_proj stack into w1; down_proj is w2."""
        for j in range(self.num_experts):
            gp = state_dict[f"{layer_prefix}.experts.{j}.gate_proj.weight"]
            up = state_dict[f"{layer_prefix}.experts.{j}.up_proj.weight"]
            dp = state_dict[f"{layer_prefix}.experts.{j}.down_proj.weight"]
            self.experts[j]["w1"].weight.data = torch.cat([gp, up], dim=0)
            self.experts[j]["w2"].weight.data = dp
        g = state_dict[f"{layer_prefix}.gate.weight"]
        self.gate.weight.data = g


def patch_olmoe_config(cfg_path: str) -> dict:
    """Placeholder: load config.json, mark the model for per-expert MoE."""
    import json
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_moesim_per_expert_moe"] = True  # consumed by the vLLM custom loader
    return cfg


if __name__ == "__main__":
    # Smoke test of the per-expert naming contract that cpu_offload_params relies on.
    moe = PerExpertMoE(num_experts=4, hidden=8, intermediate=16, top_k=2)
    names = [n for n, _ in moe.named_parameters()]
    sample = names[:4] + ["..."] + names[-2:]
    print("per-expert parameter names:", sample)
    assert "experts.0.w1.weight" in names
    assert "experts.3.w2.weight" in names
    print("OK: parameter naming supports per-expert offload via segment match")
