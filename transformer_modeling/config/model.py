"""Schema v2固定Transformer骨架模型配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import dtype_bytes, positive
from .quantization import QuantizationSpec
from .operator import LayerPatternSpec, ModelOutputSpec, OperatorSpec


@dataclass(frozen=True)
class ModelDtypeSpec:
    weight: str
    activation: str
    kv_cache: str
    logits: str
    state: str
    accumulation: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelDtypeSpec":
        spec = cls(
            weight=str(data.get("weight", "fp16")).lower(),
            activation=str(data.get("activation", "fp16")).lower(),
            kv_cache=str(data.get("kv_cache", "fp16")).lower(),
            logits=str(data.get("logits", "fp32")).lower(),
            state=str(data.get("state", data.get("kda_state", "fp16"))).lower(),
            accumulation=str(data.get("accumulation", "fp32")).lower(),
        )
        for value in spec.__dict__.values():
            dtype_bytes(value)
        return spec


@dataclass(frozen=True)
class ModelSpec:
    """模型只描述维度与算子组合，不携带厂商专用执行分支。"""

    model_id: str
    name: str
    layer_count: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    padded_vocab_size: int
    embedding: OperatorSpec
    layer_pattern: tuple[LayerPatternSpec, ...]
    output: ModelOutputSpec
    dtype: ModelDtypeSpec
    quantization: QuantizationSpec
    prefill_logits_mode: str
    extra_parameters: int
    extra_parameter_sharding: str
    metadata: dict[str, Any]
    # 前置层Pattern：先按顺序展开这些层，剩下的层才按layer_pattern循环。
    layer_prefix: tuple[LayerPatternSpec, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        dimensions = data.get("dimensions", {})
        if not isinstance(dimensions, dict):
            raise ValueError("model.dimensions must be an object")
        patterns = data.get("layer_pattern")
        if not isinstance(patterns, list) or not patterns:
            raise ValueError("model.layer_pattern must be a non-empty array")
        inference = data.get("inference", {})
        extras = data.get("extra", {})
        prefixes = data.get("layer_prefix", [])
        if not isinstance(prefixes, list):
            raise ValueError("model.layer_prefix must be an array")
        spec = cls(
            model_id=str(data.get("id", data.get("model_id", "custom"))),
            name=str(data.get("name", "自定义Transformer")),
            layer_count=int(dimensions.get("layer_count", 0)),
            hidden_size=int(dimensions.get("hidden_size", 0)),
            intermediate_size=int(dimensions.get("intermediate_size", 0)),
            vocab_size=int(dimensions.get("vocab_size", 0)),
            padded_vocab_size=int(
                dimensions.get("padded_vocab_size", dimensions.get("vocab_size", 0))
            ),
            embedding=OperatorSpec.from_dict(data.get("embedding"), "token_embedding"),
            layer_pattern=tuple(LayerPatternSpec.from_dict(item) for item in patterns),
            output=ModelOutputSpec.from_dict(data.get("output", {})),
            dtype=ModelDtypeSpec.from_dict(data.get("dtype", {})),
            quantization=QuantizationSpec.from_dict(data.get("quantization", {})),
            prefill_logits_mode=str(
                inference.get("prefill_logits_mode", "last_token")
            ).lower(),
            extra_parameters=int(extras.get("parameter_count", 0)),
            extra_parameter_sharding=str(extras.get("sharding", "tp_ep")).lower(),
            metadata=dict(data.get("metadata", {})),
            layer_prefix=tuple(LayerPatternSpec.from_dict(item) for item in prefixes),
        )
        for field in (
            "layer_count", "hidden_size", "intermediate_size",
            "vocab_size", "padded_vocab_size",
        ):
            positive(f"model.dimensions.{field}", getattr(spec, field))
        if spec.padded_vocab_size < spec.vocab_size:
            raise ValueError("padded_vocab_size must be >= vocab_size")
        if spec.prefill_logits_mode not in {"last_token", "all_prompt_tokens"}:
            raise ValueError("model.inference.prefill_logits_mode is invalid")
        if spec.extra_parameters < 0:
            raise ValueError("model.extra.parameter_count cannot be negative")
        if spec.extra_parameter_sharding not in {"replicated", "tp", "ep", "tp_ep"}:
            raise ValueError("model.extra.sharding must be replicated, tp, ep, or tp_ep")
        prefix_length = sum(item.repeat for item in spec.layer_prefix)
        if prefix_length > spec.layer_count:
            raise ValueError(
                "model.layer_prefix expands to more layers than model.dimensions.layer_count"
            )
        return spec

    def expanded_layers(self) -> tuple[LayerPatternSpec, ...]:
        """先展开前置层，剩余层循环展开Pattern，直到得到精确的Transformer层数。"""

        prefix = tuple(
            item for pattern in self.layer_prefix for item in (pattern,) * pattern.repeat
        )[:self.layer_count]
        cycle = tuple(
            item for pattern in self.layer_pattern for item in (pattern,) * pattern.repeat
        )
        rest = self.layer_count - len(prefix)
        return prefix + tuple(cycle[index % len(cycle)] for index in range(rest))

    def operators(self) -> tuple[OperatorSpec, ...]:
        items: list[OperatorSpec] = [self.embedding]
        layers = self.expanded_layers()
        for layer in layers:
            if layer.residual.type == "attnres":
                items.extend((layer.residual, layer.norm, layer.attention, layer.residual, layer.norm, layer.ffn))
            else:
                items.extend((layer.norm, layer.attention, layer.residual, layer.norm, layer.ffn, layer.residual))
        if layers and layers[-1].residual.type == "attnres":
            items.append(layers[-1].residual)
        items.extend((self.output.norm, self.output.head, self.output.sampling))
        return tuple(items)

    def uses_operator(self, type_id: str) -> bool:
        return any(item.type == type_id for item in self.operators())

    @property
    def weight_dtype(self) -> str:
        return self.dtype.weight

    @property
    def activation_dtype(self) -> str:
        return self.dtype.activation

    @property
    def kv_dtype(self) -> str:
        return self.dtype.kv_cache

    @property
    def logits_dtype(self) -> str:
        return self.dtype.logits

    @property
    def kda_state_dtype(self) -> str:
        return self.dtype.state

    @property
    def weight_bytes(self) -> float:
        return dtype_bytes(self.weight_dtype)

    @property
    def activation_bytes(self) -> float:
        return dtype_bytes(self.activation_dtype)

    @property
    def kv_bytes(self) -> float:
        return dtype_bytes(self.kv_dtype)

    @property
    def logits_bytes(self) -> float:
        return dtype_bytes(self.logits_dtype)

    @property
    def kda_state_bytes(self) -> float:
        return dtype_bytes(self.kda_state_dtype)

    def standard_attention_defaults(self) -> dict[str, Any]:
        for layer in self.expanded_layers():
            if layer.attention.type == "standard_attention":
                return layer.attention.params
        return self.layer_pattern[0].attention.params

    @property
    def query_heads(self) -> int:
        return int(self.standard_attention_defaults().get("query_heads", 1))

    @property
    def kv_heads(self) -> int:
        return int(self.standard_attention_defaults().get("kv_heads", self.query_heads))

    @property
    def head_dim(self) -> int:
        return int(
            self.standard_attention_defaults().get(
                "head_dim", self.hidden_size // max(1, self.query_heads)
            )
        )

    @property
    def query_size(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def qkv_output_size(self) -> int:
        return self.query_size + 2 * self.kv_size
