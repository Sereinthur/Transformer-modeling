"""Built-in Schema v3 model presets expressed only as ordered operators."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_K3_TARGET = 2_780_000_000_000
_GLM52_TARGET = 744_000_000_000
_V4_VARIANTS = {
    "pro": {"layers": 61, "hidden": 7168, "query_heads": 128, "head_dim": 512, "q_lora_rank": 1536, "o_lora_rank": 1024, "o_groups": 16, "experts": 384, "expert_width": 3072, "selected_entries": 1024, "route_scale": 2.5, "parameter_reference": 1_600_000_000_000},
    "flash": {"layers": 43, "hidden": 4096, "query_heads": 64, "head_dim": 512, "q_lora_rank": 1024, "o_lora_rank": 1024, "o_groups": 8, "experts": 256, "expert_width": 2048, "selected_entries": 512, "route_scale": 1.5, "parameter_reference": 284_000_000_000},
}
_V4_VOCAB, _V4_WINDOW, _V4_ROPE, _V4_CHANNELS = 129280, 128, 64, 4
_V4_INDEXER_HEADS, _V4_INDEXER_DIM = 64, 128


def _base_model(model_id: str, name: str, layers: int, hidden: int, intermediate: int, vocab: int, dtype: str = "bf16") -> dict[str, Any]:
    return {
        "id": model_id, "name": name,
        "dimensions": {"layer_count": layers, "hidden_size": hidden, "intermediate_size": intermediate, "vocab_size": vocab, "padded_vocab_size": vocab},
        "dtype": {"weight": dtype, "activation": dtype, "kv_cache": dtype, "state": dtype, "accumulation": "fp32", "logits": "fp32"},
        "inference": {"prefill_logits_mode": "last_token"},
        "quantization": {"block_size": 32, "scale_bytes": 1},
    }


def _layer(repeat: int, *operators: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    operations = []
    for source in operators:
        operator = deepcopy(source)
        base = str(operator.get("type", "unmodeled"))
        counts[base] = counts.get(base, 0) + 1
        operations.append({"id": f"{base}_{counts[base]}", "operator": operator})
    return {"repeat": repeat, "operations": operations}


def _transformer_layer(repeat: int, attention: dict[str, Any], ffn: dict[str, Any], *special: dict[str, Any]) -> dict[str, Any]:
    return _layer(repeat, *special, {"type": "rms_norm"}, attention, {"type": "standard_residual"}, {"type": "rms_norm"}, ffn, {"type": "standard_residual"})


def _mhc_transformer_layer(repeat: int, attention: dict[str, Any], ffn: dict[str, Any], mhc: dict[str, Any]) -> dict[str, Any]:
    """DeepSeek-V4: mHC replaces residuals after Attention and MoE."""
    return _layer(repeat, {"type": "rms_norm"}, attention, mhc, {"type": "rms_norm"}, ffn, mhc)


def _output() -> dict[str, Any]:
    return {"norm": {"type": "rms_norm"}, "head": {"type": "lm_head"}, "sampling": {"type": "sampling"}}


def moe_definition() -> tuple[dict[str, Any], int]:
    model = _base_model("qwen3-30b-a3b", "Qwen3-30B-A3B", 48, 2048, 6144, 151936)
    attention = {"type": "standard_attention", "implementation": "flash_attention", "query_heads": 32, "kv_heads": 4, "head_dim": 128, "query_width_equals_hidden": False}
    moe = {"type": "moe", "implementation": "standard_moe", "expert_count": 128, "experts_per_token": 8, "expert_intermediate_size": 768, "shared_expert_intermediate_size": 0, "activation": "swiglu"}
    model.update({"embedding": {"type": "token_embedding", "tied_lm_head": False}, "layer_pattern": [_transformer_layer(1, attention, moe)], "output": _output(), "metadata": {"family": "Qwen3 MoE", "mapping_quality": "approximate", "unsupported_features": []}})
    return model, 40960


def kimi_k3_definition() -> tuple[dict[str, Any], int]:
    """93-layer public K3 sequence, expressed as independent v3 operators."""
    model = _base_model("kimi-k3-official", "Kimi K3 (official config aligned)", 93, 7168, 33792, 163840, "bf16")
    model["dtype"].update({"activation": "mxfp8", "kv_cache": "bf16", "state": "bf16", "accumulation": "bf16"})
    kda = {"type": "kda", "implementation": "chunkwise", "heads": 96, "key_dim": 128, "value_dim": 128, "short_conv_kernel_size": 4, "chunk_size": 256}
    mla = {"type": "gated_mla", "implementation": "latent_cache", "query_heads": 96, "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128}
    dense = {"type": "gated_ffn", "implementation": "situ_glu", "intermediate_size": 33792, "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0, "situ_ops_per_element": 6.0}
    moe = {"type": "moe", "implementation": "latent_moe_approx", "expert_count": 896, "experts_per_token": 16, "expert_intermediate_size": 3072, "shared_expert_intermediate_size": 3072, "shared_expert_count": 2, "latent_size": 3584, "activation": "situ", "situ_ops_per_element": 6.0, "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0, "weight_dtype": "bf16", "routed_expert_weight_dtype": "mxfp4", "shared_expert_weight_dtype": "bf16"}
    attnres = {"type": "attnres", "block_size": 12}
    model.update({
        "embedding": {"type": "token_embedding", "tied_lm_head": False},
        "layer_prefix": [_transformer_layer(1, kda, dense, attnres), _transformer_layer(2, kda, moe, attnres), _transformer_layer(1, mla, moe, attnres)],
        "layer_pattern": [_transformer_layer(3, kda, moe, attnres), _transformer_layer(1, mla, moe, attnres)],
        "layer_suffix": [_transformer_layer(1, mla, moe, attnres)],
        "output": _output(),
        "metadata": {"family": "Kimi K3 Hybrid MoE", "mapping_quality": "official_config_aligned", "structure_accuracy": "official_text_backbone_exact", "performance_formula_confidence": "mixed_medium_low", "published_parameter_reference": _K3_TARGET, "config_source": "moonshotai/Kimi-K3 config.json", "peripheral_modules": [{"name": "MoonViT-V2 + PatchMergerV2", "status": "shown_not_estimated"}, {"name": "DSpark", "status": "shown_not_estimated"}], "unsupported_features": ["MoonViT-V2, PatchMergerV2 and DSpark are not estimated."]},
    })
    return model, 1_048_576


def deepseek_v4_definition(variant: str = "pro") -> tuple[dict[str, Any], int]:
    if variant not in _V4_VARIANTS:
        raise ValueError(f"unknown DeepSeek-V4 variant: {variant}")
    values = _V4_VARIANTS[variant]
    layers, h = values["layers"], values["hidden"]
    model = _base_model(f"deepseek-v4-{variant}", f"DeepSeek-V4-{variant.capitalize()}", layers, h, 4 * h, _V4_VOCAB, "fp8")
    model["quantization"].update({"block_size": 128, "scale_bytes": 1})
    model["dtype"].update({"activation": "fp8", "kv_cache": "bf16", "state": "bf16", "accumulation": "bf16"})
    attention = {"query_heads": values["query_heads"], "kv_heads": 1, "head_dim": values["head_dim"], "q_lora_rank": values["q_lora_rank"], "o_lora_rank": values["o_lora_rank"], "o_groups": values["o_groups"], "weight_dtype": "fp8", "shared_kv_projection": True, "attention_sink": True, "qk_rope_head_dim": _V4_ROPE, "kv_cache_dtype": "bf16"}
    swa = dict(attention, type="sliding_window_attention", implementation="flash_attention", sliding_window=_V4_WINDOW, query_width_equals_hidden=False)
    csa = dict(attention, type="csa_attention", implementation="compressed_kv", compress_ratio=4, compress_overlap=2, selected_entries=values["selected_entries"], sliding_window=_V4_WINDOW, selector="indexer", indexer_heads=_V4_INDEXER_HEADS, indexer_head_dim=_V4_INDEXER_DIM)
    hca = dict(attention, type="hca_attention", implementation="compressed_kv", compress_ratio=128, compress_overlap=1, selected_entries=values["selected_entries"], sliding_window=_V4_WINDOW, selector="uniform")
    moe = {"type": "moe", "implementation": "standard_moe", "expert_count": values["experts"], "experts_per_token": 6, "expert_intermediate_size": values["expert_width"], "shared_expert_intermediate_size": values["expert_width"], "shared_expert_count": 1, "activation": "swiglu", "gate_activation": "sqrtsoftplus", "swiglu_limit": 10.0, "routed_scaling_factor": values["route_scale"], "routing": "learned", "weight_dtype": "fp8", "routed_expert_weight_dtype": "mxfp4", "shared_expert_weight_dtype": "fp8"}
    mhc = {"type": "mhc", "channels": _V4_CHANNELS, "sinkhorn_iters": 20, "eps": 1e-6}
    first = hca if variant == "pro" else swa
    model.update({
        "embedding": {"type": "token_embedding", "tied_lm_head": False},
        "layer_prefix": [_mhc_transformer_layer(2, first, dict(moe, routing="hash"), mhc), _mhc_transformer_layer(1, csa, dict(moe, routing="hash"), mhc)],
        "layer_pattern": [_mhc_transformer_layer(1, hca, moe, mhc), _mhc_transformer_layer(1, csa, moe, mhc)],
        "output": _output(),
        "metadata": {"family": "DeepSeek-V4", "variant": variant, "mapping_quality": "official_config_aligned", "structure_accuracy": "official_text_backbone_exact", "performance_formula_confidence": "mixed_medium_low", "published_parameter_reference": values["parameter_reference"], "config_source": f"deepseek-ai/DeepSeek-V4-{variant.capitalize()} config.json", "peripheral_modules": [{"name": "MTP Block ×1", "status": "shown_not_estimated"}], "unsupported_features": ["MTP is not included in the performance estimate."]},
    })
    return model, 1_048_576


def glm_5_2_definition() -> tuple[dict[str, Any], int]:
    """Official GLM-5.2 text backbone: DSA (with MLA) + IndexShare + Dense→MoE."""
    model = _base_model("glm-5.2", "GLM-5.2", 78, 6144, 12288, 154880, "bf16")
    model["dtype"].update({"activation": "bf16", "kv_cache": "bf16", "state": "bf16", "accumulation": "fp32"})
    model["quantization"].update({"block_size": 128, "scale_bytes": 1})
    common_dsa = {
        "implementation": "mla_dsa", "query_heads": 64,
        "q_lora_rank": 2048, "kv_lora_rank": 512,
        "qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
        "v_head_dim": 256, "indexer_heads": 32,
        "indexer_head_dim": 128, "index_topk": 2048,
        "weight_dtype": "bf16",
    }
    dsa_full = {"type": "dsa_attention", **common_dsa, "indexer_mode": "full"}
    dsa_shared = {"type": "dsa_attention", **common_dsa, "indexer_mode": "shared"}
    dense = {"type": "gated_ffn", "implementation": "swiglu", "intermediate_size": 12288}
    moe = {
        "type": "moe", "implementation": "standard_moe", "expert_count": 256,
        "experts_per_token": 8, "expert_intermediate_size": 2048,
        "shared_expert_count": 1, "shared_expert_intermediate_size": 2048,
        "activation": "swiglu", "gate_activation": "sigmoid", "routing": "learned",
        "routed_scaling_factor": 2.5, "weight_dtype": "bf16",
        "routed_expert_weight_dtype": "bf16", "shared_expert_weight_dtype": "bf16",
    }
    # IndexShare sequence: full×3 dense, shared×3 MoE, then
    # [full×1 → shared×3]×18.  This yields 21 full + 57 shared Indexers.
    model.update({
        "embedding": {"type": "token_embedding", "tied_lm_head": False},
        "layer_prefix": [
            _transformer_layer(3, dsa_full, dense),
            _transformer_layer(3, dsa_shared, moe),
        ],
        "layer_pattern": [
            _transformer_layer(1, dsa_full, moe),
            _transformer_layer(3, dsa_shared, moe),
        ],
        "output": _output(),
        "metadata": {
            "family": "GLM-5.2 DSA MoE", "mapping_quality": "official_config_aligned",
            "structure_accuracy": "official_text_backbone_exact",
            "performance_formula_confidence": "mixed_medium",
            "published_parameter_reference": _GLM52_TARGET,
            "config_source": "zai-org/GLM-5.2 config.json",
            "peripheral_modules": [
                {"name": "MTP ×1 (IndexShare + KVShare)", "status": "shown_not_estimated"},
            ],
            "unsupported_features": [
                "MTP speculative decoding, KVShare, and the FP32-only Indexer weights projection are not separately estimated.",
            ],
        },
    })
    return model, 1_048_576


def preset_catalog() -> dict[str, Any]:
    presets = []
    model, context = moe_definition()
    presets.append({"id": model["id"], "name": model["name"], "family": "Qwen3 MoE", "mapping_quality": "approximate", "unsupported_features": model["metadata"]["unsupported_features"], "model": model, "default_max_sequence_length": context})
    k3, context = kimi_k3_definition()
    presets.append({"id": k3["id"], "name": k3["name"], "family": "Kimi K3 Hybrid MoE", "mapping_quality": k3["metadata"]["mapping_quality"], "unsupported_features": k3["metadata"]["unsupported_features"], "model": k3, "default_max_sequence_length": context})
    for variant in _V4_VARIANTS:
        model, context = deepseek_v4_definition(variant)
        presets.append({"id": model["id"], "name": model["name"], "family": "DeepSeek-V4", "mapping_quality": model["metadata"]["mapping_quality"], "unsupported_features": model["metadata"]["unsupported_features"], "model": model, "default_max_sequence_length": context})
    glm, context = glm_5_2_definition()
    presets.append({"id": glm["id"], "name": glm["name"], "family": "GLM-5.2 DSA MoE", "mapping_quality": glm["metadata"]["mapping_quality"], "unsupported_features": glm["metadata"]["unsupported_features"], "model": glm, "default_max_sequence_length": context})
    return {"schema_version": 3, "snapshot_date": "2026-08-04", "presets": presets}


def get_preset(preset_id: str, scenario: str = "base") -> dict[str, Any]:
    if preset_id == "kimi-k3-draft":
        raise ValueError("preset kimi-k3-draft has been removed; use kimi-k3-official")
    for item in preset_catalog()["presets"]:
        if item["id"] == preset_id:
            return deepcopy(item)
    raise ValueError(f"unknown model preset: {preset_id}")
