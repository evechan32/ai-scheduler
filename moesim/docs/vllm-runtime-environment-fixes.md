# vLLM 运行时环境修复记录（vllm-build，RTX 5070 / SM 12.0 / WSL）

> 记录时间：2026-08-29 | 目标：让 vllm-build 环境跑通 vLLM 0.27.2rc1（cu128 编译）真实推理
> 环境：WSL + RTX 5070 Laptop 8GiB（SM 12.0）+ conda `vllm-build`（py3.11, torch 2.13.0+cu130）

vLLM 编译完成后，`import vllm` 到真实推理之间遇到 6 个运行时问题，逐一排查并解决。
全部通过**环境变量 / 符号补丁 / symlink** 修复，未重新编译安装 vLLM。

## 问题修复总表

| # | 现象 | 根因 | 解决方案 | 涉及变量/文件 |
|---|---|---|---|---|
| 1 | `ImportError: _C_stable_libtorch.abi3.so: undefined symbol: cooperative_topk` | 编译时用 `VLLM_SKIP_COOPERATIVE_TOPK` 跳过了 `cooperative_topk.cu`，但 `CMakeLists.txt` 仍定义 `VLLM_ENABLE_COOPERATIVE_TOPK=1`，`torch_bindings.cpp` 编进了对 `cooperative_topk` 的引用 | 编译 C stub 提供该符号 + `LD_PRELOAD` 注入（该算子仅 DeepSeek-V3 sparse-attention 用，OLMoE/Qwen 不走） | `LD_PRELOAD=/tmp/opencode/libcoop_stub.so` |
| 2 | `libstdc++.so.6: version CXXABI_1.3.15 not found`（由 libicui18n.so.78 触发） | conda 的 libicu 需要新版 libstdc++（GCC13），但系统 `/lib` 的旧 libstdc++ 被优先加载（`LD_LIBRARY_PATH` 只指向 cuda） | 把 conda `env/lib`（含 libstdc++ 6.0.35）放到 `LD_LIBRARY_PATH` 最前 | `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/local/cuda/lib64` |
| 3 | `FlashInfer: SM 12.x requires CUDA >= 12.9` | SM 12.0（RTX 5070）需 `compute_120f` 编译，要求 CUDA ≥ 12.9；但 PATH 里 nvcc 是 12.8 | flashinfer JIT 改用 cu13 的 nvcc 13.3；`CUDA_HOME` 指向 cu13 | `FLASHINFER_NVCC=$CU13/bin/nvcc`、`CUDA_HOME=$CU13` |
| 4 | `error: CUDA compiler and CUDA toolkit headers are incompatible`（cccl `cuda_toolkit.h`） | flashinfer 自带 cccl 头文件对 nvcc 13.3 与 CUDART_VERSION 的版本一致性检查失败 | 用 cccl 提供的豁免宏禁用该检查 | `FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"` |
| 5 | `ld: cannot find -lcudart / -lcuda` | cu13 pip 包只有 `libcudart.so.13`（无 `libcudart.so` symlink）；`libcuda.so` 是 WSL 驱动库（在 `/usr/lib/wsl/lib`）不在链接搜索路径 | 建 `libcudart.so→libcudart.so.13`、`libcuda.so→/usr/lib/wsl/lib/libcuda.so` symlink + 显式 `-L` | `FLASHINFER_EXTRA_LDFLAGS="-L$CU13/lib -L/usr/lib/wsl/lib"` |
| 6 | `torch: undefined symbol ncclCommResume` | torch 2.13 依赖 `nvidia-nccl-cu13`，但 `site-packages/nvidia/nccl/lib/libnccl.so.2` 被 cu12 覆盖 | 强制重装 cu13 NCCL | `pip install --force-reinstall --no-deps nvidia-nccl-cu13==2.29.7` |

## 最终可用环境（每次运行 vllm 前）

```bash
conda activate vllm-build
CU13=/home/qyw/miniconda3/envs/vllm-build/lib/python3.11/site-packages/nvidia/cu13
export CUDA_HOME=$CU13
export FLASHINFER_NVCC=$CU13/bin/nvcc
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export FLASHINFER_EXTRA_LDFLAGS="-L$CU13/lib -L/usr/lib/wsl/lib"
export PATH=$CU13/bin:$PATH
export LD_LIBRARY_PATH=$CU13/lib:$CONDA_PREFIX/lib:/usr/local/cuda/lib64
export LD_PRELOAD=/tmp/opencode/libcoop_stub.so
```

## 关键结论

- 问题 1 是 vllm 编译期跳过的残留（**本质 bug**：跳过 `.cu` 未同步关闭宏），stub 是绕过非正式修复；
  正式修复两条路：①恢复 `cooperative_topk.cu` 编译 ②在 `CMakeLists.txt` 同时注释掉两处
  `VLLM_ENABLE_COOPERATIVE_TOPK=1`（与 `.cu` 一起关闭）。
- 问题 3/4/5 是 **CUDA 12.8 工具链 vs SM 12.0 + flashinfer JIT** 的三连：SM120 需 `compute_120f`
  （CUDA ≥ 12.9），而系统 nvcc 12.8 不够，切 cu13 nvcc 后又触发 cccl 检查与缺库，逐层修复。
- 全程未重新编译 vLLM（编译缓存已清理，完整重编需数小时）。

## 真机验证（首次跑通）

vLLM 0.27.2rc1 成功加载并推理 `Qwen3.5-2B`（dense 多模态，4.5GB safetensors，加载 73s，
显存 4.25GiB）。首次推理含大量 Triton JIT kernel 编译 spike（`fused_recurrent_gated_delta_rule`
等 linear-attention kernel，SM12 未优化，decode 慢至 0.67 tok/s）。硬件遥测见
`benchmarks/e2e/out/vllm_benchmark.json` 与 `docs/FRAMEWORK_BENCHMARK.md`。
