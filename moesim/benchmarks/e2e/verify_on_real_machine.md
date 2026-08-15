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
