# vLLM 源码编译安装指南（RTX 5070 / SM 12.0 / 受限网络环境）

> 适用环境：NVIDIA RTX 5070（SM 12.0 消费级 Blackwell）、CUDA 13.x、GitHub 网络受限、内存 8G 以下
> 记录时间：2026-08-17 | 基于实际编译过程经验

---

## 1. 为什么需要源码编译

vLLM 预编译 wheel **不含 SM 12.0（RTX 50 系列消费级 Blackwell）的 CUDA kernel**，
启动时报错：`No supported CUDA architectures found for major versions [12]`。
官方文档（vLLM PR #38412）确认：消费级 RTX 50 系列需从源码编译（CUDA 12.8+），
预编译 wheel 只支持数据中心 Blackwell（B200/GB200）。

## 2. 前置条件检查

```bash
# 检查 GPU 架构（确认是 SM 12.0）
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# 检查编译工具
which nvcc cmake gcc g++
nvcc --version    # 需要 >= 12.8（SM 12.0 要求）

# 检查内存/磁盘（编译 vLLM 需较多内存，建议 >= 16G；磁盘 >= 50G）
free -h && df -h /tmp
```

**注意**：系统 CUDA 版本 < 12.8 时，可用 Python 包自带的 nvcc（torch cu130 wheel 内含 nvcc 13.3）。

## 3. 创建干净环境（conda 推荐）

```bash
# conda 方式（推荐：依赖管理更可靠）
conda create -n vllm-build python=3.11 -y
conda activate vllm-build

# 或 venv 方式（需持久路径，勿放 /tmp）
python3 -m venv /home/qyw/vllm-venv
source /home/qyw/vllm-venv/bin/activate
```

## 4. 安装依赖

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install setuptools_rust setuptools_scm cmake ninja psutil
pip install bitsandbytes accelerate
```

## 5. 获取 vLLM 源码

```bash
# GitHub 网络受限时用 gh API 下载 tarball
gh api repos/vllm-project/vllm/tarball/v0.27.1 -H "Accept: application/vnd.github+json" -o vllm.tar.gz
tar xzf vllm.tar.gz -C /home/qyw/vllm-src --strip-components=1
```

## 6. 准备 8 个外部依赖（GitHub 网络受限的核心解决）

vLLM 构建需从 GitHub 拉取 8 个外部依赖，网络受限时全部用 **SRC_DIR 环境变量**指向本地克隆：

```bash
mkdir -p /home/qyw/vllm-deps
# 每个依赖（您的网络能访问 GitHub 时执行）
git clone --depth 1 https://github.com/nvidia/cutlass.git /home/qyw/vllm-deps/cutlass
git clone --depth 1 https://github.com/triton-lang/triton.git /home/qyw/vllm-deps/triton
git clone --depth 1 https://github.com/vllm-project/flash-attention.git /home/qyw/vllm-deps/flash-attention
git clone --depth 1 https://github.com/deepseek-ai/DeepGEMM.git /home/qyw/vllm-deps/deepgemm
git clone --depth 1 https://github.com/IST-DASLab/qutlass.git /home/qyw/vllm-deps/qutlass
git clone --depth 1 https://github.com/vllm-project/FlashMLA.git /home/qyw/vllm-deps/FlashMLA
git clone --depth 1 https://github.com/vllm-project/MSA.git /home/qyw/vllm-deps/MSA
git clone --depth 1 https://github.com/vllm-project/FlashKDA.git /home/qyw/vllm-deps/FlashKDA
git clone --depth 1 https://github.com/vllm-project/tml-fa4.git /home/qyw/vllm-deps/tml-fa4
```

**关键坑 1 — Triton**：vLLM 的 `TRITON_KERNELS_SRC_DIR` 需指向 triton 的
`python/triton_kernels/triton_kernels` 子目录，且该目录需有 CMakeLists.txt
（vLLM 只拷贝 .py 文件，但 FetchContent 需要 CMake 项目存在）。若无，手工创建：
```bash
cat > /home/qyw/vllm-deps/triton/python/triton_kernels/triton_kernels/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(triton_kernels NONE)
EOF
```

**关键坑 2 — 版本 pin**：vLLM 对不同依赖 pin 特定 tag/commit
（如 triton v3.5.1、qutlass e74319e），克隆后需 checkout 匹配版本。

## 7. 设置 CUDA_HOME（用 torch 自带的 nvcc）

```bash
# 找到 torch 自带的 nvcc（cu130 wheel 内含）
# 通常在: <venv>/lib/python3.x/site-packages/nvidia/cu13/
export CUDA_HOME=$(python -c "import nvidia.cu13, os; print(os.path.dirname(nvidia.cu13.__file__))")
```

**关键坑 3 — CUDA 版本错位**：`nvidia-cuda-nvcc`（13.3.73）与
`nvidia-cuda-runtime`（可能被装成 13.0.96）版本不匹配会导致
`cuda_toolkit.h:41 error: CUDA compiler and CUDA toolkit headers are incompatible`。
修复：升级 runtime 匹配 nvcc：
```bash
pip install --upgrade "nvidia-cuda-runtime==13.3.29"
```

## 8. 编译安装

```bash
cd /home/qyw/vllm-src
env \
  CUDA_HOME="$CUDA_HOME" \
  TORCH_CUDA_ARCH_LIST="12.0" \          # 关键：指定 SM 12.0
  MAX_JOBS="2" \                         # 低内存机器限制并发
  VLLM_MAIN_CUDA_VERSION="13.0" \
  VLLM_CUTLASS_SRC_DIR="/home/qyw/vllm-deps/cutlass" \
  TRITON_KERNELS_SRC_DIR="/home/qyw/vllm-deps/triton/python/triton_kernels/triton_kernels" \
  VLLM_FLASH_ATTN_SRC_DIR="/home/qyw/vllm-deps/flash-attention" \
  DEEPGEMM_SRC_DIR="/home/qyw/vllm-deps/deepgemm" \
  QUTLASS_SRC_DIR="/home/qyw/vllm-deps/qutlass" \
  TML_FA4_SRC_DIR="/home/qyw/vllm-deps/tml-fa4" \
  FLASH_MLA_SRC_DIR="/home/qyw/vllm-deps/FlashMLA" \
  FMHA_SM100_SRC_DIR="/home/qyw/vllm-deps/MSA" \
  FLASH_KDA_SRC_DIR="/home/qyw/vllm-deps/FlashKDA" \
  python -m pip install -e . --no-build-isolation
```

编译耗时：低内存机器（MAX_JOBS=2）约 30-90 分钟。

## 9. 验证

```bash
# 导入测试
python -c "import vllm; print('vLLM:', vllm.__version__)"

# 推理测试（显存检查确认 GPU 真正使用）
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='<model-path>', gpu_memory_utilization=0.8, enforce_eager=True)
out = llm.generate(['The capital of France is'], SamplingParams(max_tokens=32))
print(out[0].outputs[0].text)
"
nvidia-smi  # 确认显存被占用
```

## 10. 常见坑汇总

| 问题 | 原因 | 解决 |
|---|---|---|
| No supported CUDA architectures [12] | wheel 无 SM12 | 源码编译 + TORCH_CUDA_ARCH_LIST=12.0 |
| cuda_toolkit.h incompatible | nvcc/runtime 版本错位 | 升级 nvidia-cuda-runtime 匹配 nvcc |
| git clone timeout (443) | GitHub 网络受限 | SRC_DIR 本地源 + 手动克隆 |
| triton CMakeLists MLIR 错误 | SRC_DIR 指向 triton 根 | 指向 triton_kernels 子目录 + 手工 CMakeLists |
| QUTLASS_SRC_DIR not directory | 路径错误 | 确认克隆路径与变量一致 |
| /tmp venv 丢失 | /tmp 清理 | venv 放持久路径（/home/qyw/） |
| 内存不足 OOM | 8G 编译大型项目 | MAX_JOBS=2 限制并发 |

## 11. 相关链接

- vLLM 源码：https://github.com/vllm-project/vllm
- Blackwell 消费级构建指南：https://github.com/vllm-project/vllm/pull/38412
- SM12 兼容修复：https://github.com/vllm-project/vllm/pull/48956

## 12. moesim × vLLM 可插入性验证（2026-08-20）

> 目标：测试 moesim 当前是否能插入 vLLM。结论分四层——**环境兼容 ✓、API 契约 ✓、
> 真实引擎端到端 ✗（被 vLLM 编译未完成阻塞）、kernel 级深度插入 ✗（架构限制）。**

### 12.1 环境状态检查

- `vllm-build` conda 环境存在：Python 3.11.15 / torch 2.11.0+cu128 / transformers 5.15.1 / numpy 2.3.5。
- **vLLM 未完成编译安装**：`pip show vllm` 无记录；`import vllm` 命中源码目录的命名空间空壳
  （`NamespaceLoader` → `/home/qyw/projects/vllm`，`vllm.__file__` 为 None）；`import vllm.engine`
  报 ModuleNotFoundError。与 §10 记录的 CUDA 头文件不兼容阻塞一致。
- 因此**真实 LLM 引擎的端到端集成测试当前无法执行**，需先完成编译（nvcc/cccl 版本统一）。

### 12.2 moesim 在 vllm-build 环境兼容性（实测）

- `pip install -e ".[core]"` 安装成功（numpy 2.3.5 满足）。
- 全量测试：**109 passed, 2 skipped**（23.6s）——包括 INT4 kernel 测试。
  **注**：主环境（torch 2.9）下 INT4 测试失败（rel err 0.175 > 0.10），在 vllm-build
  （torch 2.11）下**通过**——确认该失败是 torch 版本相关的环境性问题，非代码缺陷。
- moesim 与 vLLM 配套依赖栈（torch 2.11 / transformers 5.15）完全兼容。

### 12.3 API 契约检查（VLLMExecutor ↔ vLLM LLM.generate）

moesim 后端 `moesim/executor/backends/vllm.py::VLLMExecutor.execute_gpu` 调用
`self.engine.generate(hidden_states)`（单位置参数转发）。

vLLM 源码（`/home/qyw/projects/vllm/vllm/entrypoints/llm.py:418`）签名：

```python
def generate(self, prompts, sampling_params=None, *, use_tqdm=True,
             lora_request=None, priority=None, tokenization_kwargs=None,
             mm_processor_kwargs=None) -> list[RequestOutput]
```

**结论：位置参数契约兼容**（prompts 接受 hidden_states 透传）。vLLM 编译完成后
`VLLMExecutor(LLM(...))` 可直接对接，无需改 moesim 代码。

### 12.4 深度插入边界（架构限制）

vLLM 的 MoE 层是 **fused kernel 黑盒**：`vllm/model_executor/layers/fused_moe/`
（Triton/CUDA fused MoE + deep_gemm 等），**没有逐专家 Python 分派接口**——
无法像 transformers 后端（`MoEForwardHook`）那样插入逐专家 CPU/GPU 调度。

moesim 在 vLLM 上可用的插入形态：
1. **记账式 backend**（现状）：`VLLMExecutor` 转发整个请求给 vLLM，驻留记录在 moesim 侧。
2. **kernel 级集成**（路线图 #5，未做）：改造 `FusedMoE` 层暴露逐专家钩子——工作量较大，
   属于远期。

### 12.5 结论

| 层面 | 状态 |
|---|---|
| moesim 安装/测试于 vllm-build 环境 | ✅ 109 passed |
| VLLMExecutor ↔ LLM.generate API 契约 | ✅ 兼容 |
| 真实引擎端到端（需 vLLM 编译完成） | ⏸ 阻塞于 vLLM 编译 |
| MoE kernel 级深度插入 | ✗ 架构限制（fused 黑盒） |
