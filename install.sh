#!/usr/bin/env bash
set -euo pipefail

# moesim 一键安装脚本
# 用法:
#   ./install.sh            # 核心 (numpy only, 模拟器+调度器)
#   ./install.sh --full     # + torch/transformers/accelerate (执行层)
#   ./install.sh --gpu-llama# + CUDA 版 llama-cpp-python (真机推理)
#   ./install.sh --dev      # + pytest (开发)

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

# 1. 检测 Python 版本
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "错误: 未找到 python3" >&2; exit 1
fi
PY_VER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[1/5] Python: $PY_VER"
if [[ "$(echo "$PY_VER" | cut -d. -f1)" -lt 3 || ("$(echo "$PY_VER" | cut -d. -f1)" -eq 3 && "$(echo "$PY_VER" | cut -d. -f2)" -lt 10) ]]; then
  echo "错误: 需要 Python >= 3.10 (当前 $PY_VER)" >&2; exit 1
fi

# 2. 创建 venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[2/5] 创建 venv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "[2/5] venv 就绪: $("$PYTHON" -c 'import sys; print(sys.executable)')"

# 3. 安装核心依赖
echo "[3/5] 安装核心依赖 (numpy)"
pip install -q --upgrade pip
pip install -q -e ".[core]"

# 4. 可选组件
for arg in "$@"; do
  case "$arg" in
    --full)
      echo "[4/5] 安装执行层 (torch/transformers/accelerate)"
      pip install -q -e ".[executor]"
      ;;
    --gpu-llama)
      echo "[4/5] 编译 CUDA 版 llama-cpp-python (需 nvidia-smi)"
      if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "警告: 未检测到 CUDA, 跳过 GPU llama.cpp" >&2
      else
        CMAKE_ARGS="-DGGML_CUDA=on" pip install -q --force-reinstall --no-cache-dir llama-cpp-python
      fi
      ;;
    --dev)
      echo "[4/5] 安装开发依赖 (pytest)"
      pip install -q -e ".[dev]"
      ;;
  esac
done

# 5. 验证
echo "[5/5] 运行测试验证"
if python -c "import pytest" >/dev/null 2>&1; then
  python -m pytest tests/ -q
else
  python -c "from moesim.sim.sweep import compare_policies; print('核心模块 OK (numpy-only)')"
fi
echo "✅ 安装完成"
