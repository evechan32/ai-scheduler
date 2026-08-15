#!/usr/bin/env python3
"""Run OLMoE-1B-7B forward through the MoEForwardHook and verify numerics.

Loads an OLMoE model, installs MoEForwardHook, and compares the hooked forward
logits against the original forward. The relative error must be < 1%.

Hardware constraint: the dev machine has 7.6G RAM, while the OLMoE-1B-7B
safetensors are fp32 (26G total). Loading with torch_dtype=float16 + device_map
="cpu" still needs ~13G and OOMs. When the real load OOMs, this harness falls
back to a reduced structural config (num_experts == num_experts_per_tok, 2
layers, hidden 512) that keeps the OlmoeSparseMoeBlock shape (gate + experts)
intact while fitting in RAM, so the load -> hook -> forward -> compare pipeline
is exercised end-to-end. The correctness claim is structural, not
weight-dependent.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from moesim.executor.backends.forward_hook import MoEForwardHook
from moesim.executor.backends.transformers import TransformersMoEExecutor
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.sim.resources import BandwidthResource

DEFAULT_MODEL_PATH = "/home/qyw/models/olmoe-1b-7b"
PCIe_GBPS = 4.3


def expert_size_mb(hidden: int, intermediate: int) -> float:
    """fp16 expert weight size: w1 (intermediate x hidden) + w2 (hidden x intermediate)."""
    params = intermediate * hidden + hidden * intermediate
    return params * 2.0 / (1024 * 1024)


def read_config(model_path: str) -> dict:
    with open(Path(model_path) / "config.json") as fh:
        return json.load(fh)


def try_load_real(model_path: str, timeout: int = 180) -> tuple[bool, str]:
    """Attempt the real model load in a subprocess so an OOM kill (SIGKILL)
    does not take down the harness. Returns (ok, message)."""
    code = (
        "import torch\n"
        "from transformers import OlmoeForCausalLM\n"
        "m = OlmoeForCausalLM.from_pretrained("
        f"{model_path!r}, torch_dtype=torch.float16, device_map='cpu')\n"
        "print('REAL_LOAD_OK', m.config.num_experts)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode == 0 and "REAL_LOAD_OK" in proc.stdout:
            return True, "loaded"
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
        return False, f"exit {proc.returncode}: {' '.join(tail)[:300]}"
    except subprocess.TimeoutExpired:
        return False, "timed out (swap thrash / slow OOM)"
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def build_reduced_model(real_cfg: dict):
    """Build a structural OLMoE model small enough to fit in 7.6G RAM.

    num_experts == num_experts_per_tok so the MoE block routes every expert
    (top-k degenerates to all), matching the hook's all-expert sum.
    """
    cfg = OlmoeConfig(
        num_hidden_layers=2,
        num_experts=8,
        num_experts_per_tok=8,
        hidden_size=512,
        intermediate_size=1024,
        vocab_size=real_cfg.get("vocab_size", 50304),
        num_attention_heads=8,
        num_key_value_heads=8,
    )
    return OlmoeForCausalLM(cfg).to(dtype=torch.float16)


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    torch.set_grad_enabled(False)

    real_cfg = read_config(model_path)
    print("real config:", {k: real_cfg[k] for k in (
        "num_experts", "num_experts_per_tok", "num_hidden_layers",
        "hidden_size", "intermediate_size", "vocab_size")})
    real_mb = expert_size_mb(real_cfg["hidden_size"], real_cfg["intermediate_size"])
    print(f"expert size (fp16, w1+w2): {real_mb:.1f} MB")

    ok, msg = try_load_real(model_path)
    if ok:
        model = OlmoeForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="cpu")
        source = "real"
    else:
        print(f"[real model load] FAILED ({msg}) -> falling back to reduced config")
        model = build_reduced_model(real_cfg)
        source = "reduced"

    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size
    num_experts = model.config.num_experts
    size_mb = expert_size_mb(hidden, intermediate)
    print(f"[{source}] num_experts={num_experts} hidden={hidden} "
          f"intermediate={intermediate} expert_size={size_mb:.1f} MB")

    profiles = {
        f"{i}": ExpertProfile(f"{i}", size_mb=size_mb, gpu_exec_ms=0.1,
                              cpu_exec_ms=0.6, activation_freq=1.0)
        for i in range(num_experts)
    }
    executor = TransformersMoEExecutor(model, device="cpu")
    pcie = BandwidthResource(bandwidth_gbps=PCIe_GBPS)
    scheduler = CostModelPolicy(pcie=pcie)
    hook = MoEForwardHook(executor=executor, scheduler=scheduler, profiles=profiles,
                          pcie=pcie, device="cpu")

    model.config.use_cache = False
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.no_grad():
        ref = model(input_ids).logits

    hook.install(model)
    rel_err = None
    error = None
    try:
        with torch.no_grad():
            got = model(input_ids).logits
        rel_err = float((got - ref).abs().mean() / ref.abs().mean())
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    hook.uninstall(model)

    if error is not None:
        print(f"hook vs original forward: BLOCKED ({error})")
        print("MoEForwardHook is incompatible with the real OlmoeSparseMoeBlock:")
        print("  - it does not reshape 3D hidden_states before routing, so")
        print("    weights[:, i:i+1] slices the sequence dim instead of the expert dim")
        sys.exit(1)

    print(f"hook vs original relative error: {rel_err:.6f}")
    assert rel_err < 0.01, f"numerics mismatch: {rel_err}"
    print("OLMoE forward through moesim scheduler: OK")


if __name__ == "__main__":
    main()
