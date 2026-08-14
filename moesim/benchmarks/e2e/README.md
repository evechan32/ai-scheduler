# End-to-End Verification Protocol

This harness verifies that moesim simulation predictions match real serving baselines
within 20% error on Qwen3-30B-A3B (12GB GPU + 16GB CPU machine).

## Protocol

### 1. Run microbenchmarks

```bash
cd /home/qyw/projects/moesim
python benchmarks/microbench/run_microbench.py
```

Produces `benchmarks/microbench/out/profiles.json` and `benchmarks/microbench/out/pcie.json`.

### 2. Simulate with calibrated profiles

```bash
python benchmarks/e2e/compare_moesim_vs_llamacpp.py
```

Prints a per-policy table of simulated TPOT, throughput, and cache hit rate for the
default skewed-request trace (hot experts activated 4 out of 5 steps).

### 3. Run real baselines

#### llama.cpp (CPU offloading)

```bash
llama-bench -m <path-to-Qwen3-30B-A3B-GGUF> --n-cpu-moe 8 --cpu-moe
```

#### MoE-Infinity

```bash
moe_launch <path-to-Qwen3-30B-A3B> --offload-per-layer 1
```

Both require a Qwen3-30B-A3B model in GGUF / HuggingFace format on the target machine.

### 4. Fill the comparison table

| Policy     | Sim TPOT (ms) | Real TPOT (ms) | Error (%) | Sim Tput (tok/s) | Real Tput (tok/s) |
|------------|---------------|----------------|-----------|------------------|-------------------|
| lru        |               |                |           |                  |                   |
| activation_freq |         |                |           |                  |                   |
| cost_model |               |                |           |                  |                   |
| llama.cpp  | N/A           |                | N/A       | N/A              |                   |
| MoE-Infinity | N/A         |                | N/A       | N/A              |                   |

Target: simulation vs real TPOT error < 20%.
