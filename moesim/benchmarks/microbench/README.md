# Microbenchmarks — Real-Machine Calibration

These scripts measure the two numbers the cost model needs: PCIe transfer
bandwidth (GPU <-> CPU) and per-expert FFN execution time on GPU and CPU.

## Commands

```bash
# 1. PCIe effective bandwidth (GPU -> CPU copy of ~340MB fp16 tensor, 50 repeats)
python benchmarks/microbench/measure_pcie.py
# -> prints bandwidth, writes benchmarks/microbench/out/pcie.json

# 2. Per-expert FFN time on GPU (native matmul) and CPU (moesim.expert_ffn)
python benchmarks/microbench/measure_expert_time.py
# -> prints GPU/CPU ms per expert, writes benchmarks/microbench/out/profiles.json
```

Both require a CUDA GPU and torch built with CUDA support.

## Calibration Loop

1. Run both scripts on the target machine (12G GPU + 16G RAM).
2. Feed the outputs into the simulator:
   `compare_policies` (see `benchmarks/e2e/`) reads `out/pcie.json` and
   `out/profiles.json` directly.
3. Compare simulated TPOT against real TPOT (llama.cpp `--n-cpu-moe` or
   MoE-Infinity on Qwen3-30B-A3B). Target: simulation error < 20%.

If the error exceeds 20%, re-measure with more repeats (larger `--repeats` /
`--num-experts`), or tune the latency estimate in the resource model.
