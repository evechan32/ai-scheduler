# 框架性能基准 + 硬件遥测对比

> 记录时间：2026-08-29 | 环境：WSL + RTX 5070 Laptop 8GiB（SM 12.0）
> 脚本：`benchmarks/e2e/benchmark_vllm.py`、`benchmark_llamacpp.py`、
> `compare_request_concurrency.py`（硬件遥测见 `benchmarks/microbench/resource_monitor.py`）

## 1. 实测数据

| 框架 | 模型 | 计算硬件 | 加载 | TPOT | 吞吐 | GPU util | CPU util | 内存占用 |
|---|---|---|---|---|---|---|---|---|
| **vLLM 0.27.2rc1** | Qwen3.5-2B（dense 2B 多模态） | GPU (SM12) | 76.1s | 35.7ms | **28.0 tok/s** | 29.6%（峰值 100%） | 10.2% | GPU 4.4–7.4 GiB，sys 77% |
| **llama.cpp 0.3.34** | OLMoE-1B-7B（MoE 64 专家，1.3B active） | CPU（多线程） | 3.7s | 76.2ms | **13.1 tok/s** | 0%（CPU 版） | 49.2%（峰值 100%） | sys 42% |
| **moesim（模拟）** | OLMoE-1B-7B 专家层（64 专家，8/token） | —（模拟） | — | 0.638ms | 1568 tok/s | — | — | 模拟值 |

## 2. 关键结论

1. **GPU vs CPU 异构**：vLLM（GPU）跑 dense 2B = 28 tok/s；llama.cpp（CPU）跑 MoE
   （1.3B active）= 13.1 tok/s。模型不同、硬件不同，**绝对吞吐不可直接比较**，但有
   参考意义的资源画像：GPU decode 是 memory-bound（GPU util 仅 29.6%），CPU 推理
   多线程吃满（CPU util 峰值 100%）。
2. **moesim 模拟的是专家层 only**（8 专家 × 0.076ms ≈ 0.64ms），不含 attention /
   embedding / 其余 15 层，**绝对 TPOT 与完整模型不可比**（与
   `verify_on_real_machine.md` 的诚实标注一致）。moesim 价值在于**相对策略排序**与
   异构调度决策的确定性验证，而非绝对时延预测。
3. **vLLM decode 慢的真相**：Qwen3.5-2B 的 linear-attention（`fused_recurrent_gated_delta_rule`）
   等 kernel 在 SM12 上 JIT 编译 spike + 未优化，首次推理 95s（含 JIT），warmup 后 28 tok/s。

## 3. 硬件遥测（resource_monitor 采集）

- GPU 利用率：`nvidia-smi utilization.gpu`（SM 活跃近似）；`nvidia-smi dmon -s u` 的
  `sm` 列（sm_active 代理）、`mem` 列（DRAM 带宽利用率代理）。
- DRAM 实测带宽：**315–323 GB/s（r+w）= 理论 448 GB/s 的 ~70–72%**（D2D copy 实测）。
- CPU / 系统内存：`/proc/stat` / `/proc/meminfo` 零依赖解析。
- NCU（精确 sm__throughput / dram__throughput）在本 WSL 容器不可用（`LibraryNotLoaded`），
  用 dmon 驱动指标代理，已标注。

## 4. vLLM 运行环境修复

6 个问题的完整根因与解决见 `docs/vllm-runtime-environment-fixes.md`（表格）。

## 5. 复现命令

```bash
# vLLM（vllm-build 环境，需先按 vllm-runtime-environment-fixes.md 设环境变量）
python benchmarks/e2e/benchmark_vllm.py --model /home/qyw/models/Qwen3.5-2B --max-tokens 64

# llama.cpp（py311 环境）
python benchmarks/e2e/benchmark_llamacpp.py --max-tokens 64

# 请求级并发对比（v9）
python benchmarks/e2e/compare_request_concurrency.py
```
