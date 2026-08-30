# moesim 调度集成 vLLM 解决方案

> **更新（2026-08-30）**：经进一步调研，**首选路径改为"不改模型"的方法 1+2**
> （逐层 MoE offload + KV offload，脚本 `scripts/moesim_vllm_config.py` 已跑通）。
> 本文档原"改模型拆专家"方案降级为进阶方法（方法 4，逐专家粒度，见 §2-4）。

## 0. 首选：不改模型的 vLLM 原生适配（方法 1+2）

- **方法 1 — 逐层 MoE offload**：`cpu_offload_params = {"layers.{i}.experts"}`。
  vLLM 的段匹配（`.param.` in `.name.`）支持逐层 MoE offload（等价 llama.cpp
  `--n-cpu-moe`）。moesim 按**层重要性**（早期层更敏感，QuantMoE-Bench "+FirstL"）
  选冷层 offload，优于 vLLM 默认"非选择性 offload 直到 gb 满"。**不改模型。**
- **方法 2 — KV offload**：moesim v8 的 KV 分层调度（压力阈值/驱逐）映射 vLLM
  `SimpleCPUOffloadScheduler`（lazy/eager 模式 + 池大小）。**vLLM 现成。**
- 脚本：`scripts/moesim_vllm_config.py`（生成 cpu_offload_params + KV 参数）。

---

## 1. 进阶：核心洞察（为什么逐专家需要改模型）

## 1. 核心洞察（为什么需要改模型）

vLLM 的 CPU offload（UVA）是**逐参数**的——`vllm/model_executor/offloader/uva.py` 遍历
`module.named_parameters()`，用 `cpu_offload_params` 的段匹配决定每个参数是否 offload：

```python
should_offload = any(f".{param}." in f".{name}." for param in self.cpu_offload_params)
```

但 vLLM 的 `FusedMoE` 把**所有专家权重 fused 堆叠**成两个 tensor（`experts.w13_weight`
= 64 专家堆叠、`experts.w2_weight`），参数名里没有专家索引。因此段匹配只能"整层 MoE
offload"，**无法逐专家**。

**结论**：要让 moesim 的逐专家调度生效，必须**脚本改造模型**，把专家权重拆成逐专家
module（`nn.ModuleList`），使参数名变成 `experts.0.w1`, `experts.1.w2` …，这样
`cpu_offload_params` 才能精确匹配冷专家。

## 2. 三步脚本方案

### 脚本 1：生成 offload 计划（moesim 模拟器 → 冷专家集合）

`scripts/moesim_vllm_offload_plan.py`：
- 输入：模型结构（16 层 × 64 专家）+ 显存预算 + 激活频率
- 用 moesim 的 ExpertProfile + activation_freq 排序，输出冷专家集合
- 输出：`offload_plan.json` = `{"cold_experts": ["layers.0.experts.7", ...]}`

### 脚本 2：改造模型（把 MoE 层拆成逐专家 module）

`scripts/moesim_vllm_patch_model.py`：
- 读 HF OLMoE safetensors（权重本就是逐专家的 `experts.{j}.gate_proj.weight`）
- 定义自定义 MoE 层，专家用 `nn.ModuleList([Linear, ...])` 逐专家注册
- 关键：**不 fused**，让 vLLM 的 `named_parameters()` 产生 `experts.{j}.w1/w2`
- 生成改造后的模型目录（config + 权重重排，或直接在加载时注册）

### 脚本 3：vLLM 加载 + offload + 对比

`scripts/moesim_vllm_run.py`：
- `cpu_offload_params = {f"layers.{i}.experts.{j}" for 冷专家}`
- `LLM(model=..., cpu_offload_gb=..., cpu_offload_params=...)` 加载
- 对比：vLLM 默认（非选择性 offload）vs vLLM + moesim 放置

## 3. 诚实限制（必须写清楚）

1. **性能代价**：逐专家拆分会失去 vLLM fused MoE kernel 的加速——offload 专家走慢
   路径是预期内的，但**留 GPU 的热专家也从 fused 变 unfused**，会拖慢。这是"可逐专家
   offload"换来的代价。
2. **静态 vs 动态**：这是**加载时静态放置**（moesim 算好计划，vLLM 一次配好），不是
   运行时每步动态调度。真正动态需改 vLLM 执行循环（阶段 3）。
3. **本机硬件**：7.6G 内存跑不了 14G OLMoE offload（前面 OOM 已验证），此方案需
   ≥32G 内存机器验证；本机只能验证到"计划生成 + cpu_offload_params 正确性"。
4. **逐专家参数名的确切格式**取决于自定义层的命名，需在实现时与 vLLM 的
   `cpu_offload_params` 段匹配规则（`.param.` in `.name.`）对齐验证。

## 4. 落地顺序

1. 先在本机做**脚本 1 + 脚本 2 的单元验证**（生成计划 → 确认参数名段匹配逻辑正确），
   不需要跑完整 vLLM 推理。
2. 脚本 3 在 ≥32G 内存机器上跑真实对比（vLLM 默认 vs vLLM + moesim 放置）。

## 5. 与"运行时动态调度"的关系

本方案解决"静态放置集成"（阶段 1）。若要做到 moesim `decide()` 每步驱动的**运行时
动态调度**，需在脚本 2 的自定义 MoE 层里加**每步 offload 钩子**（forward 时按
`decide()` 结果动态 `.to(device)` 专家）——这是阶段 3，在自定义层上做比改 FusedMoE
的 C++ 更容易。
