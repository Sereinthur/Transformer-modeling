"""算子参数Schema：供前端自动生成参数表单。

字段定义严格对齐 transformer_modeling.operators 各算子的 validate/estimate 实现，
默认值取算子源码中的缺省取值；required 表示算子必须读到该参数才能建模。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# 与 transformer_modeling.config.common.DTYPE_BYTES 保持一致。
DTYPE_CHOICES = [
    "fp32", "tf32", "bf16", "fp16", "fp8", "mxfp8", "int8", "int4", "mxfp4",
]


def _weight_dtype(label: str = "权重精度") -> dict[str, Any]:
    """算子级权重精度覆盖；留空表示回落到模型权重精度。"""

    return {
        "type": "enum",
        "choices": [""] + DTYPE_CHOICES,
        "default": "",
        "label": label,
        "required": False,
        "hint": "留空则继承模型权重精度；会影响参数/HBM容量及该工作项的硬件吞吐选择",
        "effect": "performance",
        "inherit_from": "model.dtype.weight",
    }


def _kv_cache_dtype() -> dict[str, Any]:
    """KV cache 精度标记字段；choices 与引擎 config.common.DTYPE_BYTES 的键一致。"""

    return {
        "type": "enum",
        "choices": [""] + DTYPE_CHOICES,
        "default": "",
        "label": "KV cache精度",
        "required": False,
        "hint": "留空则继承模型 KV 精度；覆盖后会改变 KV Cache 容量与读写带宽",
        "effect": "performance",
        "inherit_from": "model.dtype.kv_cache",
    }


_CSA_PARAMS: dict[str, Any] = {
    "query_heads": {"type": "int", "min": 1, "default": 64, "label": "Query头数", "required": True},
    "head_dim": {"type": "int", "min": 1, "default": 512, "label": "单头维度", "required": True},
    "kv_heads": {"type": "int", "min": 1, "default": 1, "label": "KV头数", "required": False,
                 "hint": "query_heads必须能被其整除"},
    "compress_ratio": {"type": "int", "min": 1, "default": 4, "label": "压缩比m", "required": False,
                       "hint": "每m个token压成一个entry"},
    "compress_overlap": {"type": "int", "min": 1, "default": 2, "label": "entry重叠倍数", "required": False},
    "selected_entries": {"type": "int", "min": 1, "default": 1024, "label": "选中entry数k", "required": False},
    "sliding_window": {"type": "int", "min": 0, "default": 128, "label": "滑窗长度", "required": False,
                       "hint": "0表示不叠加局部滑窗分支"},
    "selector": {"type": "enum", "choices": ["indexer", "uniform"], "default": "indexer",
                 "label": "选择器", "required": False},
    "indexer_heads": {"type": "int", "min": 1, "default": 64, "label": "Indexer头数", "required": False,
                      "hint": "selector=indexer时生效"},
    "indexer_head_dim": {"type": "int", "min": 1, "default": 128, "label": "Indexer头维度", "required": False,
                         "hint": "selector=indexer时生效"},
    "qk_rope_head_dim": {"type": "int", "min": 0, "default": 64, "label": "RoPE维度", "required": False,
                         "hint": "不能超过head_dim"},
    "q_lora_rank": {"type": "int", "min": 0, "default": 0, "label": "Q低秩rank", "required": False,
                    "hint": "0表示不使用低秩"},
    "o_lora_rank": {"type": "int", "min": 0, "default": 0, "label": "O低秩rank", "required": False,
                    "hint": "0表示不使用低秩"},
    "o_groups": {"type": "int", "min": 1, "default": 1, "label": "分组输出投影数", "required": False},
    "shared_kv_projection": {"type": "bool", "default": True, "label": "K/V共享投影", "required": False},
    "attention_sink": {"type": "bool", "default": False, "label": "Attention Sink", "required": False},
    "kv_cache_dtype": _kv_cache_dtype(),
    "weight_dtype": _weight_dtype(),
}


OPERATOR_SCHEMAS: dict[str, dict[str, Any]] = {
    # ---------------- embedding ----------------
    "token_embedding": {
        "slot": "embedding",
        "chinese_name": "词嵌入",
        "implementations": ["default"],
        "params": {
            "vocab_size": {
                "type": "int", "min": 0, "default": 0, "label": "Embedding 表行数", "required": False,
                "hint": "0 表示继承模型 padded_vocab_size；自定义值需能被 TP 整除",
            },
            "embedding_dim": {
                "type": "int", "min": 0, "default": 0, "label": "Embedding 输出宽度", "required": False,
                "hint": "0 表示继承 hidden_size；当前线性主干要求非零值等于 hidden_size",
            },
            "tied_lm_head": {
                "type": "bool", "default": True, "label": "与LM Head共享权重", "required": False,
            },
            "weight_dtype": _weight_dtype(),
        },
    },
    # ---------------- norm ----------------
    "rms_norm": {
        "slot": "norm",
        "chinese_name": "RMSNorm",
        "implementations": ["default"],
        "params": {
            "eps": {"type": "float", "min": 0.000000001, "default": 0.000001,
                    "label": "epsilon", "required": False,
                    "hint": "Numerical semantics only; it does not change the current performance formula."},
            "weight_dtype": _weight_dtype(),
        },
    },
    "layer_norm": {
        "slot": "norm",
        "chinese_name": "LayerNorm",
        "implementations": ["default"],
        "params": {
            "eps": {"type": "float", "min": 0.000000001, "default": 0.000001,
                    "label": "epsilon", "required": False,
                    "hint": "Numerical semantics only; it does not change the current performance formula."},
            "weight_dtype": _weight_dtype(),
        },
    },
    # ---------------- residual ----------------
    "standard_residual": {
        "slot": "residual",
        "chinese_name": "残差连接",
        "implementations": ["default"],
        "params": {},
        "description": "形状继承 model.hidden_size 与 model.dtype.activation；每次 x + f(x) 计两次读取、一次写回和一个 hidden 临时缓冲。",
    },
    "attnres": {
        "slot": "residual",
        "chinese_name": "AttnRes",
        "implementations": ["default"],
        "params": {
            "block_size": {"type": "int", "min": 1, "default": 12, "label": "每个Block层数", "required": False,
                           "hint": "Block数量由总层数向上取整得到"},
            "block_count": {"type": "int", "min": 1, "default": None, "label": "Block数量覆盖", "required": False,
                            "hint": "通常留空；填写时必须与总层数/block_size一致"},
        },
    },
    "mhc": {
        "slot": "residual",
        "chinese_name": "mHC",
        "implementations": ["default"],
        "params": {
            "channels": {"type": "int", "min": 1, "default": 4, "label": "残差通道数", "required": False},
            "sinkhorn_iters": {"type": "int", "min": 0, "default": 20, "label": "Sinkhorn迭代次数", "required": False},
            "eps": {"type": "float", "min": 0.000000001, "default": 0.000001, "label": "数值稳定eps", "required": False},
        },
    },
    # ---------------- attention ----------------
    "standard_attention": {
        "slot": "attention",
        "chinese_name": "标准Attention",
        "implementations": ["default", "standard", "flash_attention"],
        "params": {
            "query_heads": {"type": "int", "min": 1, "default": 32, "label": "Query头数", "required": True,
                            "hint": "必须能被TP整除"},
            "kv_heads": {"type": "int", "min": 1, "default": 8, "label": "KV头数", "required": False,
                         "hint": "缺省等于query_heads，且query_heads需被其整除"},
            "head_dim": {"type": "int", "min": 1, "default": 128, "label": "单头维度", "required": True},
            "sliding_window": {"type": "int", "min": 0, "default": 0, "label": "滑窗长度", "required": False,
                               "hint": "0表示全局注意力"},
            "query_width_equals_hidden": {
                "type": "bool", "default": True, "label": "Query宽度等于hidden_size", "required": False,
                "hint": "为真时校验query_heads×head_dim==hidden_size",
            },
            "q_lora_rank": {"type": "int", "min": 0, "default": 0, "label": "Q低秩rank", "required": False,
                            "hint": "0表示不使用低秩"},
            "o_lora_rank": {"type": "int", "min": 0, "default": 0, "label": "O低秩rank", "required": False,
                            "hint": "0表示不使用低秩"},
            "o_groups": {"type": "int", "min": 1, "default": 1, "label": "分组输出投影数", "required": False},
            "shared_kv_projection": {"type": "bool", "default": False, "label": "K/V共享投影", "required": False},
            "attention_sink": {"type": "bool", "default": False, "label": "Attention Sink", "required": False},
            "kv_cache_dtype": _kv_cache_dtype(),
            "weight_dtype": _weight_dtype(),
        },
    },
    "sliding_window_attention": {
        "slot": "attention",
        "chinese_name": "纯滑窗Attention",
        "implementations": ["default", "standard", "flash_attention"],
        "params": {
            "query_heads": {"type": "int", "min": 1, "default": 32, "label": "Query头数", "required": True,
                            "hint": "必须能被TP整除"},
            "kv_heads": {"type": "int", "min": 1, "default": 8, "label": "KV头数", "required": False},
            "head_dim": {"type": "int", "min": 1, "default": 128, "label": "单头维度", "required": True},
            "sliding_window": {"type": "int", "min": 1, "default": 128, "label": "滑窗长度", "required": True,
                               "hint": "纯滑窗算子必须大于0"},
            "query_width_equals_hidden": {
                "type": "bool", "default": True, "label": "Query宽度等于hidden_size", "required": False,
            },
            "q_lora_rank": {"type": "int", "min": 0, "default": 0, "label": "Q低秩rank", "required": False},
            "o_lora_rank": {"type": "int", "min": 0, "default": 0, "label": "O低秩rank", "required": False},
            "o_groups": {"type": "int", "min": 1, "default": 1, "label": "分组输出投影数", "required": False},
            "shared_kv_projection": {"type": "bool", "default": False, "label": "K/V共享投影", "required": False},
            "attention_sink": {"type": "bool", "default": False, "label": "Attention Sink", "required": False},
            "kv_cache_dtype": _kv_cache_dtype(),
            "weight_dtype": _weight_dtype(),
        },
    },
    "kda": {
        "slot": "attention",
        "chinese_name": "KDA线性注意力",
        "implementations": ["default", "chunkwise"],
        "params": {
            "heads": {"type": "int", "min": 1, "default": 64, "label": "头数", "required": True,
                      "hint": "必须能被TP整除"},
            "key_dim": {"type": "int", "min": 1, "default": 128, "label": "Key维度", "required": True},
            "value_dim": {"type": "int", "min": 1, "default": 128, "label": "Value维度", "required": True},
            "short_conv_kernel_size": {"type": "int", "min": 1, "default": 4, "label": "短卷积核大小", "required": True},
            "chunk_size": {"type": "int", "min": 1, "default": 256, "label": "Chunk大小", "required": True},
        },
    },
    "gated_mla": {
        "slot": "attention",
        "chinese_name": "Gated MLA",
        "implementations": ["default", "latent_cache"],
        "params": {
            "query_heads": {"type": "int", "min": 1, "default": 64, "label": "Query头数", "required": True,
                            "hint": "必须能被TP整除"},
            "q_lora_rank": {"type": "int", "min": 1, "default": 1536, "label": "Q低秩rank", "required": True},
            "kv_lora_rank": {"type": "int", "min": 1, "default": 512, "label": "KV低秩rank", "required": True},
            "qk_nope_head_dim": {"type": "int", "min": 1, "default": 128, "label": "QK非RoPE维度", "required": True},
            "qk_rope_head_dim": {"type": "int", "min": 1, "default": 64, "label": "QK RoPE维度", "required": True},
            "v_head_dim": {"type": "int", "min": 1, "default": 128, "label": "V头维度", "required": True},
        },
    },
    "dsa_attention": {
        "slot": "attention",
        "chinese_name": "DSA",
        "implementations": ["default", "mla_dsa"],
        "params": {
            "query_heads": {"type": "int", "min": 1, "default": 64, "label": "Query头数", "required": True,
                            "hint": "必须能被TP整除"},
            "q_lora_rank": {"type": "int", "min": 1, "default": 2048, "label": "Q低秩rank", "required": True},
            "kv_lora_rank": {"type": "int", "min": 1, "default": 512, "label": "KV低秩rank", "required": True},
            "qk_nope_head_dim": {"type": "int", "min": 1, "default": 192, "label": "QK非RoPE维度", "required": True},
            "qk_rope_head_dim": {"type": "int", "min": 1, "default": 64, "label": "QK RoPE维度", "required": True},
            "v_head_dim": {"type": "int", "min": 1, "default": 256, "label": "V头维度", "required": True},
            "indexer_mode": {"type": "enum", "choices": ["full", "shared"], "default": "full",
                              "label": "IndexShare模式", "required": True,
                              "hint": "full 计算Indexer与Top-K；shared复用最近full层的索引"},
            "indexer_heads": {"type": "int", "min": 1, "default": 32, "label": "Indexer头数", "required": True},
            "indexer_head_dim": {"type": "int", "min": 1, "default": 128, "label": "Indexer头维度", "required": True},
            "index_topk": {"type": "int", "min": 1, "default": 2048, "label": "Token Top-K", "required": True,
                           "hint": "DSA 在原始token粒度上选择的最大历史token数"},
            "weight_dtype": _weight_dtype(),
        },
    },
    "csa_attention": {
        "slot": "attention",
        "chinese_name": "CSA压缩稀疏注意力",
        "implementations": ["default", "compressed_kv"],
        "params": dict(_CSA_PARAMS),
    },
    "hca_attention": {
        "slot": "attention",
        "chinese_name": "HCA重度压缩注意力",
        "implementations": ["default", "compressed_kv"],
        "params": dict(
            _CSA_PARAMS,
            compress_ratio=dict(_CSA_PARAMS["compress_ratio"], default=128),
            compress_overlap=dict(_CSA_PARAMS["compress_overlap"], default=1),
            selector=dict(_CSA_PARAMS["selector"], default="uniform"),
            sliding_window=dict(_CSA_PARAMS["sliding_window"], default=128),
        ),
    },
    # ---------------- ffn ----------------
    "dense_ffn": {
        "slot": "ffn",
        "chinese_name": "Dense FFN",
        "implementations": ["default", "gelu"],
        "params": {
            "intermediate_size": {
                "type": "int", "min": 0, "default": 0, "label": "中间层宽度", "required": False,
                "hint": "缺省取模型intermediate_size，且必须能被TP整除",
            },
        },
    },
    "gated_ffn": {
        "slot": "ffn",
        "chinese_name": "Dense Gated FFN",
        "implementations": ["default", "swiglu", "fused_gate_up", "situ_glu"],
        "params": {
            "intermediate_size": {
                "type": "int", "min": 0, "default": 0, "label": "中间层宽度", "required": False,
                "hint": "缺省取模型intermediate_size，且必须能被TP整除",
            },
            "activation_situ_beta": {"type": "float", "min": 0.000001, "default": 4.0,
                                      "label": "SiTU beta", "required": False},
            "activation_situ_linear_beta": {"type": "float", "min": 0.000001, "default": 25.0,
                                             "label": "SiTU linear beta", "required": False},
            "situ_ops_per_element": {"type": "float", "min": 0.000001, "default": 6.0,
                                      "label": "SiTU每元素ops", "required": False},
        },
    },
    "moe": {
        "slot": "ffn",
        "chinese_name": "Mixture of Experts",
        "implementations": ["default", "standard_moe", "latent_moe_approx"],
        "params": {
            "expert_count": {"type": "int", "min": 1, "default": 128, "label": "专家总数", "required": True,
                             "hint": "必须能被EP整除"},
            "experts_per_token": {"type": "int", "min": 1, "default": 8, "label": "每token激活专家数", "required": True,
                                  "hint": "不能超过expert_count"},
            "expert_intermediate_size": {"type": "int", "min": 1, "default": 768, "label": "专家中间层宽度",
                                         "required": True, "hint": "必须能被TP整除"},
            "shared_expert_intermediate_size": {"type": "int", "min": 0, "default": 0, "label": "共享专家宽度",
                                                "required": False, "hint": "0表示无共享专家"},
            "shared_expert_count": {"type": "int", "min": 0, "default": 1, "label": "共享专家数量", "required": False},
            "latent_size": {"type": "int", "min": 0, "default": 0, "label": "Latent维度", "required": False,
                            "hint": "0表示专家在hidden域计算"},
            "routing": {"type": "enum", "choices": ["learned", "hash"], "default": "learned",
                        "label": "路由方式", "required": False},
            "gate_activation": {"type": "enum", "choices": ["softmax", "sigmoid", "sqrtsoftplus"],
                                "default": "softmax", "label": "打分函数", "required": False},
            "activation": {"type": "enum", "choices": ["swiglu", "gated", "situ", "gelu"], "default": "swiglu",
                           "label": "专家激活", "required": False,
                           "hint": "swiglu/gated按3矩阵计权重，其他按2矩阵"},
            "routed_scaling_factor": {"type": "float", "min": 0, "default": None, "label": "路由权重缩放",
                                      "required": False, "hint": "只缩放数值，不产生额外计算"},
            "swiglu_limit": {"type": "float", "min": 0, "default": 0, "label": "Clipped SwiGLU上限",
                              "required": False, "hint": "0表示不裁剪"},
            "situ_ops_per_element": {"type": "float", "min": 0, "default": 6.0, "label": "SiTU每元素ops",
                                     "required": False, "hint": "仅latent_moe_approx实现生效"},
            "activation_situ_beta": {"type": "float", "min": 0.000001, "default": 4.0,
                                      "label": "SiTU beta", "required": False},
            "activation_situ_linear_beta": {"type": "float", "min": 0.000001, "default": 25.0,
                                             "label": "SiTU linear beta", "required": False},
            "weight_dtype": _weight_dtype("Router/Latent权重精度"),
            "routed_expert_weight_dtype": _weight_dtype("路由专家权重精度"),
            "shared_expert_weight_dtype": _weight_dtype("共享专家权重精度"),
        },
    },
    # ---------------- output ----------------
    "lm_head": {
        "slot": "output",
        "chinese_name": "LM Head",
        "implementations": ["default"],
        "params": {},
    },
    "sampling": {
        "slot": "output",
        "chinese_name": "采样",
        "implementations": ["default"],
        "params": {},
    },
    # ---------------- 任意槽位 ----------------
    "unmodeled": {
        "slot": "any",
        "chinese_name": "未建模算子",
        "implementations": ["default"],
        "params": {
            "name": {"type": "string", "default": "未建模算子", "label": "显示名称", "required": False},
            "parameter_count": {"type": "int", "min": 0, "default": 0, "label": "参数量", "required": False},
            "state_bytes": {"type": "int", "min": 0, "default": 0, "label": "持久状态字节", "required": False},
            "note": {"type": "string", "default": "", "label": "备注", "required": False},
        },
    },
}


# 每个槽位可选的算子类型，供前端做拖拽/替换约束。
SLOT_OPERATORS: dict[str, list[str]] = {}
for _type_id, _schema in OPERATOR_SCHEMAS.items():
    SLOT_OPERATORS.setdefault(_schema["slot"], []).append(_type_id)


def operator_schema(type_id: str) -> dict[str, Any]:
    """取单个算子的参数Schema；未知类型直接报错。"""

    try:
        return OPERATOR_SCHEMAS[type_id]
    except KeyError as exc:
        raise ValueError(f"unknown operator type: {type_id}") from exc


def default_params(type_id: str) -> dict[str, Any]:
    """生成算子的默认参数字典：跳过无默认值和空字符串的可选覆盖项。"""

    params = {}
    for name, field in operator_schema(type_id)["params"].items():
        default = field.get("default")
        if default is None or default == "":
            continue
        params[name] = default
    return params


def chinese_name(type_id: str) -> str:
    schema = OPERATOR_SCHEMAS.get(type_id)
    return schema["chinese_name"] if schema else type_id


def schema_payload() -> dict[str, Any]:
    """/api/operator-schemas 的响应体。"""

    operators = deepcopy(OPERATOR_SCHEMAS)
    numerical_only = {
        "eps", "activation_situ_beta", "activation_situ_linear_beta",
        "routed_scaling_factor", "swiglu_limit",
    }
    inheritance_sources = {
        "weight_dtype": "model.dtype.weight",
        "routed_expert_weight_dtype": "model.dtype.weight",
        "shared_expert_weight_dtype": "model.dtype.weight",
        "kv_cache_dtype": "model.dtype.kv_cache",
        "intermediate_size": "model.dimensions.intermediate_size",
        "vocab_size": "model.dimensions.padded_vocab_size",
        "embedding_dim": "model.dimensions.hidden_size",
    }
    structural = {
        "query_heads", "kv_heads", "head_dim", "intermediate_size",
        "expert_count", "experts_per_token", "expert_intermediate_size",
        "shared_expert_intermediate_size", "shared_expert_count", "latent_size",
        "vocab_size", "embedding_dim", "channels", "block_size", "block_count",
        "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
        "v_head_dim", "o_lora_rank", "o_groups",
    }
    inheritance = {
        "weight_dtype", "routed_expert_weight_dtype", "shared_expert_weight_dtype",
        "kv_cache_dtype", "intermediate_size", "vocab_size", "embedding_dim",
    }
    for schema in operators.values():
        for key, field in schema.get("params", {}).items():
            field.setdefault("effect", "numerical" if key in numerical_only else "performance")
            field["performance_impact"] = key not in numerical_only
            field["structure_impact"] = key in structural
            field["numerical_semantics"] = key in numerical_only
            if key in inheritance:
                field["inherit_from"] = inheritance_sources[key]
    return {
        "schema_version": 3,
        "dtype_choices": DTYPE_CHOICES,
        "slot_operators": SLOT_OPERATORS,
        "operators": operators,
    }
