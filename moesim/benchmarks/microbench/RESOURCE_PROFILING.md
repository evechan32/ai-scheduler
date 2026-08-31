# Resource Usage Profiling — Real-Machine Metrics

> 2026-08-20 | 目的：测试/基准运行时记录 GPU 带宽利用率、sm_active、GPU/CPU/内存利用率。
> 脚本：`benchmarks/microbench/resource_monitor.py`（采样器）+ `profile_resource_usage.py`（演示负载）。

## 指标与采集方式

| 指标 | 含义 | 采集方式 | 备注 |
|---|---|---|---|
| `gpu_util` | GPU 利用率（%） | `nvidia-smi utilization.gpu`，0.2s 轮询 | 驱动的 SM-busy 近似 |
| `dmon_sm` | **sm_active 代理**（%） | `nvidia-smi dmon -s u` 的 `sm` 列，1s 粒度 | 驱动级 SM 利用率；NCU 不可用时最接近 sm_active |
| `dmon_mem` | **DRAM 带宽利用率代理**（%） | `nvidia-smi dmon -s u` 的 `mem` 列（显存控制器利用率） | 近似值 |
| `measured_bw_gbs` | DRAM 实测带宽（GB/s，r+w） | torch D2D copy（1GiB × 5 轮取最优） | 预热避免页面错误干扰 |
| `sm_clock_mhz` / `mem_clock_mhz` | SM/显存时钟 | `nvidia-smi clocks.sm/clocks.mem` | |
| `cpu_util` | CPU 利用率（%） | `/proc/stat` 差值 | 零依赖 |
| `sys_mem_util` | 系统内存利用率（%） | `/proc/meminfo` | 零依赖 |
| NCU 指标 | 精确 `dram__throughput` / `sm__throughput`（% of peak） | NSight Compute `ncu` | 本环境**不可用**（`LibraryNotLoaded`，容器驱动限制）；真机可用时启用 |

所有指标零依赖实现（nvidia-smi CLI + /proc），无需 pynvml/psutil。

## 使用

```bash
# 采样模式：边跑命令边采样
python benchmarks/microbench/resource_monitor.py "python my_test.py" --interval 0.2 --out out/usage.json

# 演示负载模式：compute-bound（matmul）+ bandwidth-bound（copy）实测 + 记录
python benchmarks/microbench/profile_resource_usage.py
```

## 实测记录（2026-08-20，RTX 5070 Laptop 8GiB，torch 2.11.0+cu128，vllm-build 环境）

负载：4096³ matmul ×60（compute-bound）+ 1GiB D2D copy ×5（bandwidth-bound）。

| 指标 | mean | max | p95 |
|---|---|---|---|
| GPU util (%) | 62.5 | 100.0 | 100.0 |
| sm_active 代理（dmon sm, %） | 33.3 | 100.0 | 100.0 |
| DRAM bw-util 代理（dmon mem, %） | 3.0 | 9.0 | 9.0 |
| GPU mem used (MiB) | ~300 | ~352 | ~352 |
| CPU util (%) | 11.2 | 20.5 | 20.5 |
| Sys mem util (%) | 37.5 | 38.4 | 38.4 |
| SM clock (MHz) | ~1170 | 1545 | 1545 |
| Mem clock (MHz) | ~11140 | 12001 | 12001 |

**DRAM 实测带宽：315–323 GB/s（r+w）= 理论峰值（448 GB/s，128-bit GDDR7）的 ~70–72%。**

读取要点：
- compute-bound 阶段 GPU util 打到 100%、sm_active 峰值 100%，但 DRAM 带宽利用率低
  （~3–9%）——计算密集特征与 matmul 负载一致。
- `dmon` 采样为 1s 粒度，短负载下样本少（2–8 个），均值参考意义有限，max/p95 更有代表性。
- NCU（精确 sm__throughput / dram__throughput）在本容器不可用（`LibraryNotLoaded`），
  真机（宿主机/特权容器）上 `--with-ncu` 可启用。

## 环境限制

- `ncu` / `torch.profiler(CUPTI)` 在容器 GPU 直通下不可用（`LibraryNotLoaded` /
  `CUPTI_ERROR_INVALID_DEVICE`）→ 以 dmon 驱动指标 + 实测带宽作为代理，已标注。
