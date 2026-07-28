"""模型预设和Hugging Face config到固定槽位Schema的映射。"""

from __future__ import annotations

from typing import Any

from .catalog import get_preset


def _hf_definition(config: dict[str, Any]) -> dict[str, Any]:
    required = ("num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size", "num_attention_heads")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Hugging Face config缺少字段: {', '.join(missing)}")
    h = int(config["hidden_size"])
    qh = int(config["num_attention_heads"])
    kvh = int(config.get("num_key_value_heads", qh))
    dim = int(config.get("head_dim", h // qh))
    model_type = str(config.get("model_type", "unknown"))
    unsupported = []
    if config.get("num_local_experts"):
        ffn = {
            "type": "moe", "implementation": "standard_moe",
            "expert_count": int(config["num_local_experts"]),
            "experts_per_token": int(config.get("num_experts_per_tok", 1)),
            "expert_intermediate_size": int(config["intermediate_size"]),
            "shared_expert_intermediate_size": int(config.get("moe_intermediate_size", 0)),
            "activation": "swiglu",
        }
    else:
        ffn = {"type": "gated_ffn", "implementation": "swiglu", "intermediate_size": int(config["intermediate_size"])}
    known = {"llama", "qwen2", "qwen3", "mistral", "mixtral"}
    if model_type not in known:
        unsupported.append(f"未注册model_type={model_type}的专用结构，按标准Decoder骨架映射")
    return {
        "id": str(config.get("_name_or_path", "hf-custom")), "name": str(config.get("_name_or_path", "Hugging Face自定义模型")),
        "dimensions": {"layer_count": int(config["num_hidden_layers"]), "hidden_size": h, "intermediate_size": int(config["intermediate_size"]), "vocab_size": int(config["vocab_size"]), "padded_vocab_size": int(config.get("padded_vocab_size", config["vocab_size"]))},
        "embedding": {"type": "token_embedding", "tied_lm_head": bool(config.get("tie_word_embeddings", False))},
        "layer_pattern": [{"repeat": 1, "norm": {"type": "rms_norm"}, "attention": {"type": "standard_attention", "implementation": "flash_attention", "query_heads": qh, "kv_heads": kvh, "head_dim": dim, "query_width_equals_hidden": qh * dim == h}, "residual": {"type": "standard_residual"}, "ffn": ffn}],
        "output": {"norm": {"type": "rms_norm"}, "head": {"type": "lm_head"}, "sampling": {"type": "sampling"}},
        "dtype": {"weight": "bf16", "activation": "bf16", "kv_cache": "bf16", "state": "bf16", "accumulation": "fp32", "logits": "fp32"},
        "inference": {"prefill_logits_mode": "last_token"},
        "quantization": {"block_size": 32, "scale_bytes": 1},
        "metadata": {"family": model_type, "mapping_quality": "field_mapping", "unsupported_features": unsupported},
    }


def resolve_model_definition(*, preset_id: str | None = None,
                             hf_config: dict[str, Any] | None = None,
                             scenario: str = "base") -> dict[str, Any]:
    if (preset_id is None) == (hf_config is None):
        raise ValueError("preset_id和hf_config必须且只能提供一个")
    if preset_id is not None:
        preset = get_preset(preset_id, scenario)
        return {"resolved_model": preset["model"], "default_max_sequence_length": preset["default_max_sequence_length"], "warnings": preset["model"].get("metadata", {}).get("unsupported_features", [])}
    assert hf_config is not None
    model = _hf_definition(hf_config)
    return {"resolved_model": model, "default_max_sequence_length": int(hf_config.get("max_position_embeddings", 0)), "warnings": model["metadata"]["unsupported_features"]}
