"""内置算子注册表。"""

from __future__ import annotations

from typing import Any

from .attention import GatedMLAOperator, KDAOperator, StandardAttentionOperator
from .base import TransformerOperator
from .compressed_attention import CSAOperator, HCAOperator
from .ffn import DenseFFNOperator, GatedFFNOperator, MoEOperator
from .standard import (
    AttnResOperator, LayerNormOperator, LMHeadOperator, MHCOperator,
    RMSNormOperator, SamplingOperator, StandardResidualOperator,
    TokenEmbeddingOperator, UnmodeledOperator,
)


_OPERATORS: dict[str, TransformerOperator] = {}


def register(operator: TransformerOperator) -> None:
    if operator.type_id in _OPERATORS:
        raise ValueError(f"duplicate operator type: {operator.type_id}")
    _OPERATORS[operator.type_id] = operator


for _operator in (
    TokenEmbeddingOperator(), RMSNormOperator(), LayerNormOperator(),
    StandardResidualOperator(), AttnResOperator(), MHCOperator(),
    StandardAttentionOperator(), KDAOperator(), GatedMLAOperator(),
    CSAOperator(), HCAOperator(), DenseFFNOperator(), GatedFFNOperator(),
    MoEOperator(), LMHeadOperator(), SamplingOperator(), UnmodeledOperator(),
):
    register(_operator)


def get_operator(type_id: str) -> TransformerOperator:
    try:
        return _OPERATORS[type_id]
    except KeyError as exc:
        raise ValueError(f"unknown operator type: {type_id}") from exc


def get_operator_catalog() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "operators": [
            {
                "type": operator.type_id,
                "chinese_name": operator.chinese_name,
                "slot": operator.slot,
                "implementations": sorted(operator.implementations),
            }
            for operator in _OPERATORS.values()
        ],
    }
