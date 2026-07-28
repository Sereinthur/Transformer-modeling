"""固定版本的真实模型预设；预设只生成算子组合。"""

from __future__ import annotations

from copy import deepcopy
from math import floor
from typing import Any


_DENSE = {
    "qwen3-0.6b": ("Qwen3-0.6B", 28, 1024, 3072, 151936, 16, 8, 128, True, 40960),
    "qwen3-1.7b": ("Qwen3-1.7B", 28, 2048, 6144, 151936, 16, 8, 128, True, 40960),
    "qwen3-4b": ("Qwen3-4B", 36, 2560, 9728, 151936, 32, 8, 128, True, 40960),
    "qwen3-8b": ("Qwen3-8B", 36, 4096, 12288, 151936, 32, 8, 128, False, 40960),
    "qwen3-14b": ("Qwen3-14B", 40, 5120, 17408, 151936, 40, 8, 128, False, 40960),
    "qwen3-32b": ("Qwen3-32B", 64, 5120, 25600, 151936, 64, 8, 128, False, 40960),
    "llama-3.2-1b": ("Llama 3.2 1B", 16, 2048, 8192, 128256, 32, 8, 64, True, 131072),
    "llama-3.2-3b": ("Llama 3.2 3B", 28, 3072, 8192, 128256, 24, 8, 128, True, 131072),
    "llama-3.1-8b": ("Llama 3.1 8B", 32, 4096, 14336, 128256, 32, 8, 128, False, 131072),
}

_K3_TARGET = 2_800_000_000_000
_K3_SCENARIOS = {
    "compact": (48, 6144, 0.75), "base": (64, 7168, 1.0),
    "deep_wide": (80, 8192, 1.25),
}


def _base_model(model_id: str, name: str, layers: int, hidden: int,
                intermediate: int, vocab: int, dtype: str = "bf16") -> dict[str, Any]:
    return {
        "id": model_id, "name": name,
        "dimensions": {
            "layer_count": layers, "hidden_size": hidden,
            "intermediate_size": intermediate, "vocab_size": vocab,
            "padded_vocab_size": vocab,
        },
        "dtype": {
            "weight": dtype, "activation": dtype, "kv_cache": dtype,
            "state": dtype, "accumulation": "fp32", "logits": "fp32",
        },
        "inference": {"prefill_logits_mode": "last_token"},
        "quantization": {"block_size": 32, "scale_bytes": 1},
    }


def dense_definition(preset_id: str) -> tuple[dict[str, Any], int]:
    name, layers, h, inter, vocab, qh, kvh, dim, tied, context = _DENSE[preset_id]
    model = _base_model(preset_id, name, layers, h, inter, vocab)
    model.update({
        "embedding": {"type": "token_embedding", "tied_lm_head": tied},
        "layer_pattern": [{
            "repeat": 1, "norm": {"type": "rms_norm"},
            "attention": {
                "type": "standard_attention", "implementation": "flash_attention",
                "query_heads": qh, "kv_heads": kvh, "head_dim": dim,
                "query_width_equals_hidden": qh * dim == h,
            },
            "residual": {"type": "standard_residual"},
            "ffn": {"type": "gated_ffn", "implementation": "swiglu", "intermediate_size": inter},
        }],
        "output": {"norm": {"type": "rms_norm"}, "head": {"type": "lm_head"}, "sampling": {"type": "sampling"}},
        "metadata": {
            "family": "Qwen3 Dense" if preset_id.startswith("qwen") else "Llama",
            "mapping_quality": "approximate" if preset_id.startswith("qwen") else "exact_for_supported_fields",
            "unsupported_features": ["QK-Norm少量向量开销"] if preset_id.startswith("qwen") else [],
        },
    })
    return model, context


def moe_definition() -> tuple[dict[str, Any], int]:
    model = _base_model("qwen3-30b-a3b", "Qwen3-30B-A3B", 48, 2048, 6144, 151936)
    model.update({
        "embedding": {"type": "token_embedding", "tied_lm_head": False},
        "layer_pattern": [{
            "repeat": 1, "norm": {"type": "rms_norm"},
            "attention": {"type": "standard_attention", "implementation": "flash_attention", "query_heads": 32, "kv_heads": 4, "head_dim": 128, "query_width_equals_hidden": False},
            "residual": {"type": "standard_residual"},
            "ffn": {"type": "moe", "implementation": "standard_moe", "expert_count": 128, "experts_per_token": 8, "expert_intermediate_size": 768, "shared_expert_intermediate_size": 0, "activation": "swiglu"},
        }],
        "output": {"norm": {"type": "rms_norm"}, "head": {"type": "lm_head"}, "sampling": {"type": "sampling"}},
        "metadata": {"family": "Qwen3 MoE", "mapping_quality": "approximate", "unsupported_features": ["QK-Norm少量向量开销"]},
    })
    return model, 40960


def kimi_k3_definition(scenario: str = "base") -> tuple[dict[str, Any], int]:
    if scenario not in _K3_SCENARIOS:
        raise ValueError(f"unknown Kimi K3 scenario: {scenario}")
    layers, hidden, scale = _K3_SCENARIOS[scenario]
    align = lambda value: max(64, floor(value * scale / 64) * 64)
    model = _base_model("kimi-k3-draft", f"Kimi K3参数化草案（{scenario}）", layers, hidden, 4 * hidden, 163840, "mxfp4")
    model["dtype"].update({"activation": "mxfp8", "kv_cache": "mxfp8", "state": "mxfp8", "accumulation": "bf16"})
    kda = {"type": "kda", "implementation": "chunkwise", "heads": 64, "key_dim": align(128), "value_dim": align(128), "short_conv_kernel_size": 4, "chunk_size": 256}
    mla = {"type": "gated_mla", "implementation": "latent_cache", "query_heads": 64, "q_lora_rank": align(1536), "kv_lora_rank": align(512), "qk_nope_head_dim": align(128), "qk_rope_head_dim": align(64), "v_head_dim": align(128)}
    # 先扣除注意力、Embedding、Norm、Router与AttnRes，再反解专家宽度。
    kda_per = hidden * (64 * (2 * align(128) + align(128)) + 2 * 64 * align(128)) + 64 * (2 * align(128) + align(128)) * 4
    q_width, kv_width = 64 * (align(128) + align(64)), 64 * (align(128) + align(128))
    mla_per = hidden * align(1536) + align(1536) * q_width + hidden * (align(512) + align(64)) + align(512) * kv_width + hidden * 64 + 64 * align(128) * hidden
    kda_layers = sum(1 for index in range(layers) if index % 4 != 3)
    mla_layers = layers - kda_layers
    # AttnRes每个子层包含一个pseudo-query和一个内部RMSNorm，末尾再聚合一次。
    attnres_parameters = (2 * layers + 1) * 2 * hidden
    fixed = kda_layers * kda_per + mla_layers * mla_per + 2 * 163840 * hidden + (2 * layers + 1) * hidden + layers * hidden * 896 + attnres_parameters
    coefficient = layers * 3 * hidden * (896 + 1)
    expert_width = max(256, floor((_K3_TARGET - fixed) / coefficient / 256) * 256)
    moe = {"type": "moe", "implementation": "latent_moe_approx", "expert_count": 896, "experts_per_token": 16, "expert_intermediate_size": expert_width, "shared_expert_intermediate_size": expert_width, "activation": "swiglu", "situ_ops_per_element": 6}
    modeled = fixed + coefficient * expert_width
    model.update({
        "embedding": {"type": "token_embedding", "tied_lm_head": False},
        "layer_pattern": [
            {"repeat": 3, "norm": {"type": "rms_norm"}, "attention": kda, "residual": {"type": "attnres", "block_count": 8}, "ffn": moe},
            {"repeat": 1, "norm": {"type": "rms_norm"}, "attention": mla, "residual": {"type": "attnres", "block_count": 8}, "ffn": moe},
        ],
        "output": {"norm": {"type": "rms_norm"}, "head": {"type": "lm_head"}, "sampling": {"type": "sampling"}},
        "extra": {"parameter_count": max(0, _K3_TARGET - modeled), "sharding": "tp_ep"},
        "metadata": {"family": "Kimi K3 Hybrid MoE", "scenario": scenario, "mapping_quality": "parameterized_draft", "parameter_target": _K3_TARGET, "unsupported_features": ["Stable LatentMoE与SiTU精确方程", "视觉前端"]},
    })
    return model, 1_048_576


def preset_catalog() -> dict[str, Any]:
    presets = []
    for preset_id in _DENSE:
        model, context = dense_definition(preset_id)
        presets.append({"id": preset_id, "name": model["name"], "family": model["metadata"]["family"], "mapping_quality": model["metadata"]["mapping_quality"], "unsupported_features": model["metadata"]["unsupported_features"], "model": model, "default_max_sequence_length": context})
    model, context = moe_definition()
    presets.append({"id": model["id"], "name": model["name"], "family": "Qwen3 MoE", "mapping_quality": "approximate", "unsupported_features": model["metadata"]["unsupported_features"], "model": model, "default_max_sequence_length": context})
    k3, context = kimi_k3_definition()
    presets.append({"id": "kimi-k3-draft", "name": "Kimi K3（参数化草案）", "family": "Kimi K3 Hybrid MoE", "mapping_quality": "parameterized_draft", "unsupported_features": k3["metadata"]["unsupported_features"], "model": k3, "scenarios": list(_K3_SCENARIOS), "default_scenario": "base", "default_max_sequence_length": context})
    return {"schema_version": 2, "snapshot_date": "2026-07-20", "presets": presets}


def get_preset(preset_id: str, scenario: str = "base") -> dict[str, Any]:
    if preset_id == "kimi-k3-draft":
        model, context = kimi_k3_definition(scenario)
        return {"id": preset_id, "name": model["name"], "model": model, "default_max_sequence_length": context}
    for item in preset_catalog()["presets"]:
        if item["id"] == preset_id:
            return deepcopy(item)
    raise ValueError(f"unknown model preset: {preset_id}")
