# 基于可替换算子的 Transformer 性能评估模型

这是一个面向部署容量、性能数量级和瓶颈趋势的解析模型。它不根据模型名称选择公式，而是把模型写成固定 Transformer 骨架中的算子组合：

```text
Embedding
  → [Norm → Attention → Residual → Norm → FFN/MoE → Residual] × L
  → Final Norm → LM Head → Sampling
```

每个算子独立报告参数、持久状态、临时空间、逻辑/执行 Ops、HBM 流量、kernel 数量和通信请求；公共 Roofline 引擎再根据芯片吞吐、效率、带宽和启动延迟换算时间。模型目标是可解释的趋势评估，不是周期精确仿真。

## 功能范围

- Schema v2 固定骨架与可循环的 `layer_pattern`。
- Attention：标准 MHA/GQA/MQA、Flash Attention、KDA、Gated MLA。
- FFN：Dense、Gated/SwiGLU、普通 MoE、LatentMoE 近似。
- Norm：RMSNorm、LayerNorm。
- Residual：普通 Residual、AttnRes 近似。
- Embedding、LM Head、Sampling 和显式 `unmodeled` 算子。
- TP、连续层 PP、MoE 简化 EP，以及 Ring、Bus、Crossbar、2D Mesh 通信公式。
- 容量不足时仍返回 TTFT、TPOT 和吞吐，并标记为理论性能。
- 手动建模网页，可直接编辑模型、芯片、请求、经验效率和并行参数。
- Qwen、Llama、MoE 和 Kimi K3 参数化草案等预设；预设只生成算子组合。

暂不包含 CP、DP 模型并行、CPU/NVMe Offload、热点专家、网络拥塞、离散事件调度、服务排队和尾延迟。

## 启动网页

Python 3.10 及以上，无第三方依赖：

```powershell
git clone https://github.com/Sereinthur/Transformer-modeling.git
cd Transformer-modeling
python web_app.py
```

也可以双击 `启动可视化.bat`。默认地址为 `http://127.0.0.1:8000`；不自动打开浏览器时：

```powershell
python web_app.py --no-browser
```

手动模式模型页包含循环 Pattern 编辑器。每一行都可以设置重复次数，并独立选择 Norm、Attention、Residual 和 FFN 算子；KDA、MLA、AttnRes、MoE 与 MXFP 参数是通用算子参数，不依赖 Kimi K3 预设。

示例中的效率系数是依据公开评测资料约束校准的保守经验初值。公开资料提供的是端到端吞吐、TTFT、TPOT和容量分解，不能唯一反解单个算子的效率，因此这些数值不是实测真值：Dense示例使用`0.65/0.20`的Prefill/Decode GEMM效率、`0.50/0.15`的Attention效率、`0.15`的向量效率和`0.75`的HBM效率；MoE与KDA/MLA示例使用更保守的初值。用户应按硬件、框架、精度和Shape继续调整。

## 命令行与 Python API

```powershell
python -m transformer_modeling examples/single_chip_gqa.json -o result.json
python -m transformer_modeling examples/tp4_gqa.json -o tp4_result.json
python -m transformer_modeling examples/pp4_gqa.json -o pp4_result.json
python -m transformer_modeling examples/moe_qwen3_30b_a3b.json -o moe_result.json
python -m transformer_modeling examples/kimi_k3_base_tp64.json -o kimi_k3_result.json
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
k3 = resolve_model_definition(preset_id="kimi-k3-draft", scenario="base")
```

`Config`只接受 `schema_version=2`，不提供 v1 适配层。

## Schema v2 示例

```json
{
  "schema_version": 2,
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
        "norm": {"type": "rms_norm"},
        "attention": {
          "type": "kda",
          "implementation": "chunkwise",
          "heads": 64,
          "key_dim": 128,
          "value_dim": 128,
          "short_conv_kernel_size": 4,
          "chunk_size": 256
        },
        "residual": {"type": "attnres", "block_count": 8},
        "ffn": {
          "type": "moe",
          "implementation": "latent_moe_approx",
          "expert_count": 896,
          "experts_per_token": 16,
          "expert_intermediate_size": 2048,
          "shared_expert_intermediate_size": 2048,
          "activation": "swiglu"
        }
      },
      {
        "repeat": 1,
        "norm": {"type": "rms_norm"},
        "attention": {
          "type": "gated_mla",
          "query_heads": 64,
          "q_lora_rank": 1536,
          "kv_lora_rank": 512,
          "qk_nope_head_dim": 128,
          "qk_rope_head_dim": 64,
          "v_head_dim": 128
        },
        "residual": {"type": "attnres", "block_count": 8},
        "ffn": {"type": "moe", "implementation": "latent_moe_approx", "expert_count": 896, "experts_per_token": 16, "expert_intermediate_size": 2048, "shared_expert_intermediate_size": 2048}
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

`layer_pattern`按顺序循环展开并在达到 `layer_count` 时截断。未知算子类型直接报错；已知但没有性能公式的结构应使用 `unmodeled`：

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

## 并行模型

总设备数必须满足：

```text
DeviceCount = TP × EP × PP
```

TP规则由算子声明。标准 Attention 和 Gated FFN采用 Column/Row Parallel，并在输出加入 All-Reduce；LM Head按词表分片并在采样前 All-Gather。

PP按完整 Transformer 层连续切分，默认用当前请求下的 Prefill+Decode估算成本做近似均衡；也可以通过 `pipeline_stage_boundaries` 提供 `PP-1` 个累计层边界。Stage间传输 hidden activation。

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

预设只负责生成 Schema v2 算子组合，估算核心不检查厂商、模型名或 `family`：

- Llama/Qwen Dense：标准Attention + Gated FFN。
- 普通MoE：标准Attention + MoE。
- Kimi K3草案：3×KDA + 1×Gated MLA、AttnRes、LatentMoE近似。

本地网页接口：

- `GET /api/example`
- `GET /api/operator-catalog`
- `GET /api/model-presets`
- `POST /api/model-definitions/resolve`
- `POST /api/estimate`

Kimi K3预设是显式假设草案，不是官方 `config.json`。Base场景对账到2.8T参数，TP=1的含scale关键rank容量约1.35TiB；未归属参数计入容量但不虚构FLOPs。K3也使用普通算子组装入口，可以自行修改Pattern、启用PP或EP。

## 项目结构

```text
transformer_modeling/
├─ config/             # Schema v2、算子槽位和跨字段校验
├─ operators/          # 通用算子接口、实现与注册表
├─ core/               # WorkItem与Roofline成本换算
├─ communication/      # TP/EP集合通信与拓扑公式
├─ parallel/pipeline/  # PP调度
├─ estimators/         # 算子组合、容量、阶段和端到端汇总
└─ models/             # 预设目录与HF字段映射
static/
├─ index.html
└─ js/                 # 参数表单、Pattern编辑、结果展示
```

## 测试

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

前端语法检查：

```powershell
Get-ChildItem static -Recurse -Filter *.js | ForEach-Object { node --check $_.FullName }
```

测试覆盖单算子、混合Pattern、K3 2.8T容量、KDA/MLA状态、TP Shape、四种拓扑、PP划分、EP All-to-All、容量不足继续计算、网页API和ES Module语法。
