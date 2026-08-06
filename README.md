# 基于可替换算子的 Transformer 性能评估模型

这是一个面向部署容量、性能数量级和瓶颈趋势的解析模型。它不根据模型名称选择公式，而是把文本主干写成受约束的层序和有序独立算子列表：

```text
Embedding
  → prefix → repeating pattern → suffix
  → Final Norm → LM Head → Sampling

layer operators: Norm | Attention | MoE | Residual | AttnRes | mHC
```

每个算子独立报告参数、持久状态、临时空间、逻辑/执行 Ops、HBM 流量、kernel 数量和通信请求；公共 Roofline 引擎再根据芯片吞吐、效率、带宽和启动延迟换算时间。模型目标是可解释的趋势评估，不是周期精确仿真。

## 功能范围

- Schema v3 的 `layer_prefix → layer_pattern → layer_suffix` 线性层序；每个层段由可编辑的有序 `operations` 列表组成。
- Attention：标准 MHA/GQA/MQA、独立纯滑窗 Attention、KDA、Gated MLA；标准/滑窗 Attention 支持低秩 Q/O 投影。
- 压缩 Attention：CSA（细粒度压缩 + Lightning Indexer top-k + 滑窗）与 HCA（重度压缩 + 等距采样）。
- FFN：Dense、Gated/SwiGLU、普通 MoE、LatentMoE 近似；MoE 支持 learned/hash 路由、多个共享专家和按组件权重精度。
- Norm：RMSNorm、LayerNorm。
- Residual、AttnRes 与 mHC 均为独立算子；按其在完整展开层序中的实际出现次数估算状态、临时激活与通信近似。
- Embedding、LM Head、Sampling 和显式 `unmodeled` 算子。
- TP、连续层 PP、MoE 简化 EP，以及 Ring、Bus、Crossbar、2D Mesh 通信公式。
- `rmsnorm_linear`、`rope_kv_write` 融合开关影响流量与 kernel 计量；`kv_paged` 按页粒度取整 KV 容量。
- 容量不足时仍返回 TTFT、TPOT 和吞吐，并标记为理论性能。
- 可视化建模器是唯一网页入口，可直接编辑模型、芯片、请求、经验效率、并行参数和层内有序算子。
- Qwen、Llama、MoE、Kimi K3与DeepSeek-V4官方文本主干对齐预设；预设只生成算子组合，近似性能公式另行标注置信度。

暂不包含 CP、DP 模型并行、CPU/NVMe Offload、热点专家、网络拥塞、离散事件调度、服务排队和尾延迟。

## 启动网页

Python 3.10 及以上，无第三方依赖：

```powershell
git clone https://github.com/Sereinthur/Transformer-modeling.git
cd Transformer-modeling
python -m transformer_modeling.visual_app
```

也可以双击 `启动可视化.bat`。默认地址为 `http://127.0.0.1:8001`；不自动打开浏览器时：

```powershell
python -m transformer_modeling.visual_app --no-browser
```

模型页包含 Prefix、循环 Pattern 与 Suffix 的有序算子编辑器。每个层段可设置重复次数；每个算子均可插入、删除、移动、替换并独立编辑参数。KDA、MLA、AttnRes、mHC、MoE 与 MXFP 参数不依赖具体模型预设。停止服务可双击 `停止可视化.bat`，脚本只会终止监听 8001 且命令行属于 `transformer_modeling.visual_app` 的 Python 进程。

示例中的效率系数是依据公开评测资料约束校准的保守经验初值。公开资料提供的是端到端吞吐、TTFT、TPOT和容量分解，不能唯一反解单个算子的效率，因此这些数值不是实测真值：Dense示例使用`0.65/0.20`的Prefill/Decode GEMM效率、`0.50/0.15`的Attention效率、`0.15`的向量效率和`0.75`的HBM效率；MoE与KDA/MLA示例使用更保守的初值。用户应按硬件、框架、精度和Shape继续调整。

GEMM 效率也可以改用形状感知口径：在 `execution.efficiencies.gemm_by_rows` 提供 `[[rows, efficiency], ...]` 校准点，模型会按每个 GEMM 的 M 维在 log2 域插值（两端取边界值），并覆盖 Prefill/Decode 常数效率；未配置时回退常数。MoE 专家 GEMM 的效率与 tile 对齐按平均每专家行数计量。

## 命令行与 Python API

```powershell
python -m transformer_modeling examples/single_chip_gqa.json -o result.json
python -m transformer_modeling examples/tp4_gqa.json -o tp4_result.json
python -m transformer_modeling examples/pp4_gqa.json -o pp4_result.json
python -m transformer_modeling examples/moe_qwen3_30b_a3b.json -o moe_result.json
python -m transformer_modeling examples/kimi_k3_base_tp64.json -o kimi_k3_result.json
python -m transformer_modeling examples/deepseek_v4_pro_b200_nvl72.json -o v4_result.json
```

稳定的顶层调用入口：

```python
import json
from transformer_modeling import Config, estimate

data = json.load(open("examples/single_chip_gqa.json", encoding="utf-8"))
result = estimate(Config.from_dict(data), details=True)
```

算子目录与模型定义接口：

```python
from transformer_modeling.operators import get_operator_catalog
from transformer_modeling.models import resolve_model_definition

catalog = get_operator_catalog()
k3 = resolve_model_definition(preset_id="kimi-k3-official")
```

`Config`只接受 `schema_version=3`。旧的固定四槽位配置、`hidden_state_flow` 与 `residual_connections` 都会明确报“配置版本已过期”，不会被隐式转换。

## Schema v3 示例

```json
{
  "schema_version": 3,
  "model": {
    "id": "custom",
    "name": "自定义混合模型",
    "dimensions": {
      "layer_count": 64,
      "hidden_size": 7168,
      "intermediate_size": 28672,
      "vocab_size": 163840,
      "padded_vocab_size": 163840
    },
    "embedding": {"type": "token_embedding", "tied_lm_head": false},
    "layer_pattern": [
      {
        "repeat": 3,
        "operations": [
          {"id": "norm_pre", "operator": {"type": "rms_norm"}},
          {"id": "kda", "operator": {"type": "kda", "implementation": "chunkwise", "heads": 64, "key_dim": 128, "value_dim": 128, "short_conv_kernel_size": 4, "chunk_size": 256}},
          {"id": "attnres", "operator": {"type": "attnres", "block_size": 12}},
          {"id": "residual_attention", "operator": {"type": "standard_residual"}},
          {"id": "norm_post", "operator": {"type": "rms_norm"}},
          {"id": "moe", "operator": {"type": "moe", "implementation": "latent_moe_approx", "expert_count": 896, "experts_per_token": 16, "expert_intermediate_size": 2048, "shared_expert_intermediate_size": 2048, "activation": "swiglu"}},
          {"id": "residual_moe", "operator": {"type": "standard_residual"}}
        ]
      },
      {
        "repeat": 1,
        "operations": [
          {"id": "norm_pre", "operator": {"type": "rms_norm"}},
          {"id": "mla", "operator": {"type": "gated_mla", "query_heads": 64, "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128}},
          {"id": "attnres", "operator": {"type": "attnres", "block_size": 12}},
          {"id": "residual_attention", "operator": {"type": "standard_residual"}},
          {"id": "norm_post", "operator": {"type": "rms_norm"}},
          {"id": "moe", "operator": {"type": "moe", "implementation": "latent_moe_approx", "expert_count": 896, "experts_per_token": 16, "expert_intermediate_size": 2048, "shared_expert_intermediate_size": 2048}},
          {"id": "residual_moe", "operator": {"type": "standard_residual"}}
        ]
      }
    ],
    "output": {
      "norm": {"type": "rms_norm"},
      "head": {"type": "lm_head"},
      "sampling": {"type": "sampling"}
    },
    "dtype": {
      "weight": "mxfp4",
      "activation": "mxfp8",
      "kv_cache": "mxfp8",
      "state": "mxfp8",
      "accumulation": "bf16",
      "logits": "fp32"
    },
    "quantization": {"block_size": 32, "scale_bytes": 1},
    "extra": {"parameter_count": 0, "sharding": "tp_ep"}
  }
}
```

`layer_pattern`按顺序循环展开并在达到中间层预算时截断。可选的 `layer_prefix` 与 `layer_suffix` 使用相同层结构语法，分别按声明顺序展开一次并锚定模型开头和末尾；中间层数等于 `layer_count-prefix-suffix`，且必须至少保留一个循环层。这可紧凑表达“特殊前置层 + 周期主体 + 固定末层”的模型。

模型层统一为有序的 `operations` 列表。`Norm`、`Attention`、`MoE`、`Residual`、`mHC` 和 `AttnRes` 都是独立算子；算子的类型、实现方式与参数由后端注册表校验并估算。

当前估算器会按列表顺序执行并保留特殊算子的层位置，但普通独立算子的总 FLOPs、参数与峰值工作区仍按逐算子累加；尚未实现任意算子重排带来的张量存活期、邻接融合或执行依赖推导。因此，替换/增删算子会改变估算，而仅交换两个普通独立算子的顺序通常不会改变数值。这是部署性能近似的已知边界，不应视为可执行计算图。

```json
{"type": "standard_residual"}
{"type": "attnres", "block_size": 12}
{"type": "mhc", "channels": 4, "sinkhorn_iters": 20, "eps": 1e-6}
```

未知算子类型直接报错；已知但没有性能公式的结构应使用 `unmodeled`：

```json
{
  "type": "unmodeled",
  "name": "待补充结构",
  "parameter_count": 1000000,
  "state_bytes": 4096,
  "note": "只计容量，不虚构时延"
}
```

这会令 `performance_complete=false`，同时保留已建模代理结构的解析延迟；该数值不构成严格上界或下界。

## 成本与容量组合

算子本地时间：

```text
T_compute = executed_ops / (hardware_throughput × operator_efficiency × tile_efficiency)
T_memory  = HBM_payload_bytes / effective_HBM_bandwidth
T_operator = max(T_compute, T_memory)
           + rho × min(T_compute, T_memory)
           + kernel_count × launch_latency
```

模型容量按关键 rank 报告：

- 权重和持久状态求和。
- 临时激活与通信缓冲取关键路径最大值。
- 加入校准时不可用显存和用户运行时预留。
- PP按每个 Stage 独立计算容量。
- `capacity_feasible=false`只表示不能按当前驻留假设部署，不阻止理论性能计算。

结果会同时给出：

```json
{
  "capacity_feasible": false,
  "performance_is_theoretical": true
}
```

MXFP4/MXFP8按可配置 block scale 计量：

```text
Bytes = ceil(N / block_size) × (block_size × bits/8 + scale_bytes)
```

性能估算要求硬件提供对应的 `mxfp4_mxfp8_ops_per_second`，不会借用 INT4 吞吐。

权重精度可以按算子甚至按组件覆盖：算子级 `weight_dtype`，MoE 另有 `routed_expert_weight_dtype` 与 `shared_expert_weight_dtype`，缺省逐级回落到 `dtype.weight`。它既改变权重字节数（容量与 HBM 流量），也会写入对应 `WorkItem.compute_dtype`，由 Roofline 按 BF16/FP16、FP8 或 MXFP4/MXFP8 选择硬件吞吐；缺少对应吞吐时会标记为模型默认吞吐回退。容量结果的每个 Stage 会给出 `weights_by_dtype`，按精度分桶列出权重字节。

## 并行模型

总设备数必须满足：

```text
DeviceCount = TP × EP × PP
```

TP规则由算子声明。标准 Attention 和 Gated FFN采用 Column/Row Parallel，并在输出加入 All-Reduce；LM Head按词表分片并在采样前 All-Gather。

PP按完整 Transformer 层连续切分，默认用当前请求下的 Prefill+Decode估算成本做近似均衡；也可以通过 `pipeline_stage_boundaries` 提供 `PP-1` 个累计层边界。Stage间传输 hidden activation；发送占用互联而非算力，可与本 Stage 下一个微批的计算重叠。Decode 的稳态吞吐按流水稳态间隔（瓶颈 Stage 或链路）报告，可见 TPOT 仍按完整 makespan 报告。

EP只作用于MoE：

- `expert_count % EP == 0`。
- 每个专家内部仍可使用TP。
- Batch请求近似均匀分配，`active_ranks=min(EP, Batch)`。
- Dispatch与Combine各加入一次All-to-All。
- 假设专家负载理想均衡，不模拟热点和尾延迟。

All-to-All API payload近似：

```text
S = M_local × TopK × H × activation_bytes
```

## 预设与接口

预设只负责生成 Schema v3 算子组合，估算核心不检查厂商、模型名或 `family`：

- Qwen3-30B-A3B：标准 Attention + MoE，作为通用线性编辑起点。
- Kimi K3官方配置对齐版（`kimi-k3-official`）：93层，69×KDA + 24×Gated MLA，首层Dense SiTU-GLU、其余92层LatentMoE；第93层固定为MLA，Block AttnRes边界为L1/L13/…/L85。
- DeepSeek-V4（`deepseek-v4-pro`、`deepseek-v4-flash`）：Pro为31 HCA + 30 CSA + 0纯滑窗；Flash为2纯滑窗 + 20 HCA + 21 CSA；两者均为0 Standard Attention。每个 Block 在 Attention 和 MoE 后各使用一次 mHC，替代普通 Residual。

本地网页接口：

- `GET /api/example-config`
- `GET /api/operator-catalog`
- `GET /api/model-presets`
- `POST /api/resolve-preset`
- `POST /api/config-to-flowchart`
- `POST /api/flowchart-to-config`
- `POST /api/estimate`

DeepSeek-V4预设按官方配置和推理实现对齐文本主干：Pro对账到1.6T参数、Flash对账到284B。CSA/HCA都包含本地128滑窗分支，只有CSA包含Indexer；MTP作为已知外围模块展示但不进入性能估算。压缩Attention与mHC的性能公式仍明确标为近似，各算子的 `assumptions` 列出压缩比、top-k命中数与Sinkhorn融合等口径。

Kimi K3预设来自官方 `config.json` 与 `modeling_kimi_linear.py`，层数、层序、注意力类型、Dense/MoE分布和关键维度按公开字段对齐；参数锚点为2.78T。MoonViT/PatchMerger与DSpark作为已知外围模块展示但不进入估算；修改预设的算子列表或关键参数后应视为基于官方配置的自定义模型。

## 项目结构

```text
transformer_modeling/
├─ config/             # Schema v3、有序算子列表与跨字段校验
├─ operators/          # 通用算子接口、实现与注册表
├─ core/               # WorkItem与Roofline成本换算
├─ communication/      # TP/EP集合通信与拓扑公式
├─ parallel/pipeline/  # PP调度
├─ estimators/         # 算子组合、容量、阶段和端到端汇总
└─ models/             # 预设目录与HF字段映射
transformer_modeling/visual_app/
├─ server.py           # 唯一网页/API 服务
├─ flowchart_schema.py # Schema v3 与线性编辑视图往返
└─ static/             # 页面、参数表单、算子编辑与结果展示
```

## 测试

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

前端语法检查：

```powershell
Get-ChildItem transformer_modeling\visual_app\static -Recurse -Filter *.js | ForEach-Object { Get-Content -Raw -Encoding UTF8 $_.FullName | node --input-type=module --check }
```

测试覆盖单算子、有序 Pattern、K3 2.8T容量、KDA/MLA状态、AttnRes状态与PP载荷、TP Shape、四种拓扑、PP划分、EP All-to-All、容量不足继续计算、网页API和ES Module语法，以及DeepSeek-V4的层排列、Hash路由、压缩KV容量、mHC参数/四通道PP状态和按精度分桶的权重字节。
