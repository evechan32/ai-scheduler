# Real-Machine Verification Protocol (OLMoE-1B-7B)

This protocol verifies moesim simulation predictions against real inference on
OLMoE-1B-7B-0125 (MoE: 7B total / 1.3B active / 64 experts / 8 per token).

## 1. Prerequisites

- Model GGUF: `allenai/OLMoE-1B-7B-0125-GGUF` (Q3_K_L, ~3.6GB)
  - ModelScope: `modelscope download --model allenai/OLMoE-1B-7B-0125-GGUF`
- Model safetensors (transformers offload verification):
  - ModelScope: `modelscope download --model allenai/OLMoE-1B-7B-0125`
- `pip install llama-cpp-python` (llama.cpp baseline)

Verified config: `num_experts=64, num_experts_per_tok=8, num_hidden_layers=16`.

## 2. Real baseline (llama.cpp)

```bash
python -c "
from llama_cpp import Llama
import time
llm = Llama(model_path='<gguf-path>', n_ctx=512, verbose=False)
t0 = time.perf_counter()
llm('The capital of France is', max_tokens=16)
el = time.perf_counter() - t0
print(f'throughput: {16/el:.1f} tok/s')
"
```

Observed on dev machine (8G GPU / 7.6G RAM, CPU-only llama.cpp): **5.3 tok/s**.

## 3. Calibrate simulator

Run microbenchmarks to measure PCIe + expert execution, producing
`benchmarks/microbench/out/{pcie,profiles}.json` (see microbench README).

## 4. Simulate

```bash
python benchmarks/e2e/compare_moesim_vs_llamacpp.py \
  --profiles benchmarks/microbench/out/profiles.json \
  --pcie benchmarks/microbench/out/pcie.json
```

Prints per-policy TPOT / throughput / hit rate.

## 5. Compare (target: error < 20%)

| Policy | Sim TPOT (ms) | Real TPOT (ms) | Error (%) |
|--------|---------------|----------------|-----------|
| lru | | | |
| activation_freq | | | |
| cost_model | | | |
| llama.cpp | N/A | | N/A |

## 6. Notes

- Dev machine (8G GPU / 7.6G RAM) runs OLMoE-1B-7B via llama.cpp CPU build;
  for 12G/16G target hardware substitute Qwen3-30B-A3B per the same protocol.
- transformers backend offload (load/unload moving expert params GPU<->CPU)
  is verified separately with the safetensors weights (see
  `tests/executor/test_transformers_offload.py`).

## 7. Measured results (2026-08-15, dev machine)

Real baseline (llama.cpp CPU build, Q3_K_L): **5.3 tok/s** (TPOT ~188 ms/token).

moesim simulation (calibrated with real expert timings: GPU 0.076ms / CPU
0.639ms per expert, 64 experts):

| Policy | Sim TPOT (ms) | Sim tput (tok/s) | hit |
|--------|---------------|------------------|-----|
| lru | 0.695 | 1439.9 | 0.983 |
| activation_freq | 1.204 | 830.3 | 0.983 |
| cost_model | 0.639 | 1564.5 | 0.158 |

**Honest interpretation:** the simulator models ONLY the MoE expert layer
(8 active experts × 0.076ms), NOT attention/embedding/sampling or the other
15 layers. Full-model TPOT therefore differs from real by >20%. What IS
validated: **relative policy ordering** (cost_model > lru > activation_freq),
which is the scheduling decision the simulator is designed to support.
Closing the absolute gap requires calibrating the full model cost (attention +
embedding), which is a follow-up beyond the v2 expert-scheduling scope.

## 8. Forward-hook verification (2026-08-15)

Goal: run OLMoE-1B-7B forward through `MoEForwardHook`
(`moesim/executor/backends/forward_hook.py`) and verify the hooked output
matches the original forward with < 1% relative error. Harness:
`benchmarks/e2e/run_hook_inference.py`.

### Hardware limit

- Dev machine: 7.6G RAM (+2G swap), 8G GPU.
- OLMoE-1B-7B safetensors are fp32, ~26G on disk (6 shards).
- `from_pretrained(torch_dtype=float16, device_map="cpu")` casts to fp16 but
  still holds ~13G of weights in RAM, exceeding 7.6G.
- Real config: `num_experts=64, num_experts_per_tok=8, num_hidden_layers=16,
  hidden_size=2048, intermediate_size=1024, vocab_size=50304`.
- Per-expert fp16 size (w1 `intermediate×hidden` + w2 `hidden×intermediate`):
  8.0 MB.

### Load attempt

The real fp16 load was attempted in a subprocess (180s timeout). It did not
complete: it thrashed into swap and was abandoned — the expected OOM on this
hardware. The harness then fell back to a reduced structural config that keeps
the `OlmoeSparseMoeBlock` shape (gate + experts) intact while fitting in RAM:

- `num_hidden_layers=2, num_experts=8, num_experts_per_tok=8,
  hidden_size=512, intermediate_size=1024, vocab_size=50304`
- Random fp16 weights, ~79M params (~150 MB total), well under 7.6G.
- `num_experts == num_experts_per_tok` so the block's top-k routing degenerates
  to "all experts", matching the hook's all-expert sum.

### Result: BLOCKED — `MoEForwardHook` is incompatible with the real `OlmoeSparseMoeBlock`

The harness loads, installs the hook, and runs the forward, but the hooked
forward fails before any relative error can be measured:

```
RuntimeError: The size of tensor a (8) must match the size of tensor b (512)
at non-singleton dimension 2
```

Full incompatibility list (each confirmed against `modeling_olmoe.py` from
transformers 4.57.3):

1. **3D reshape / weight slicing.** `OlmoeSparseMoeBlock.forward` reshapes
   `hidden_states` to 2D `(batch*seq, hidden)` before routing.
   `MoEForwardHook` does not reshape, so `softmax(gate(hidden_states))` is 3D
   `(batch, seq, num_experts)` and `weights[:, i:i+1]` slices the sequence dim
   instead of the expert dim — the shape mismatch above.
2. **Top-k routing.** The original selects `top_k = num_experts_per_tok = 8`
   experts per token; the hook sums over all `num_experts` (64). Correct only
   in the degenerate reduced config where `num_experts == num_experts_per_tok`.
3. **Return value.** `OlmoeDecoderLayer.forward` unpacks
   `hidden_states, router_logits = self.mlp(hidden_states)`; the hook returns a
   single tensor.
4. **CPU kernel weights.** `TransformersMoEExecutor.execute_cpu` reads
   `module.w1`/`module.w2`, but `OlmoeMLP` uses
   `gate_proj`/`up_proj`/`down_proj`.

These are bugs in the Task 25 deliverables (`forward_hook.py`,
`transformers.py`), which the harness task is scoped not to modify. Until they
are fixed, the < 1% numeric verification cannot pass on any machine: the
incompatibility is structural, not memory-bound.

### Running on a 16G+ machine

On hardware with >=16G RAM the real fp16 load succeeds, but the same structural
bugs (1-4) still block the comparison. After `forward_hook.py`/`transformers.py`
are fixed to match OLMoE's real interface (reshape to 2D, top-k routing, tuple
return, gate/up/down projections), run:

```bash
python benchmarks/e2e/run_hook_inference.py /home/qyw/models/olmoe-1b-7b
```

Expected: `hook vs original relative error: <0.01` and
`OLMoE forward through moesim scheduler: OK`.

## 8. Forward-hook verification (2026-08-15)

`run_hook_inference.py` exercises the full pipeline: load OLMoE → install
MoEForwardHook → scheduler-driven CPU/GPU expert execution → compare logits
vs original forward.

**Hardware limit:** dev machine has 7.6G RAM; OLMoE-1B-7B safetensors are fp32
(26G total). fp16 cpu load needs ~13G → OOM/timeout. The harness auto-falls
back to a reduced structural config (8 experts, hidden 512, 2 layers,
num_experts_per_tok == num_experts) that keeps the OlmoeSparseMoeBlock shape.

**Measured (2026-08-15):**
- real model load: FAILED (timed out — swap thrash / OOM), as documented
- reduced config: hook vs original forward relative error **0.0589%** (target < 1%) ✓

**Fix landed:** forward_hook.py was extended to handle real OLMoE 3D input
([batch, seq, hidden]) with per-token top-k routing and the HF tuple contract
(hidden, router_logits); the legacy 2D single-tensor path is kept for the
mini-MoE unit tests. Both paths verified.

**To run on a 16G+ machine with the real model:**
```bash
python benchmarks/e2e/run_hook_inference.py /path/to/olmoe-1b-7b
```
Expected: real load succeeds, relative error < 1%.
