"""同构Prefill/Decode分离与混合Prefix Cache配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import positive


@dataclass(frozen=True)
class PrefixCacheSpec:
    kda_state_hit_rate: float = 0.0
    kda_cached_prefix_tokens: int = 0
    mla_prefix_hit_rate: float = 0.0
    mla_average_matched_tokens: int = 0
    block_tokens: int = 16
    metadata_bytes_per_block: int = 16

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrefixCacheSpec":
        spec = cls(
            kda_state_hit_rate=float(data.get("kda_state_hit_rate", 0.0)),
            kda_cached_prefix_tokens=int(data.get("kda_cached_prefix_tokens", 0)),
            mla_prefix_hit_rate=float(data.get("mla_prefix_hit_rate", 0.0)),
            mla_average_matched_tokens=int(data.get("mla_average_matched_tokens", 0)),
            block_tokens=int(data.get("block_tokens", 16)),
            metadata_bytes_per_block=int(data.get("metadata_bytes_per_block", 16)),
        )
        for name in ("kda_state_hit_rate", "mla_prefix_hit_rate"):
            value = getattr(spec, name)
            if not 0 <= value <= 1:
                raise ValueError(f"serving.prefix_cache.{name} must be in [0, 1]")
        for name in ("kda_cached_prefix_tokens", "mla_average_matched_tokens", "metadata_bytes_per_block"):
            if getattr(spec, name) < 0:
                raise ValueError(f"serving.prefix_cache.{name} cannot be negative")
        positive("serving.prefix_cache.block_tokens", spec.block_tokens)
        return spec


@dataclass(frozen=True)
class DeploymentSpec:
    mode: str = "aggregated"
    prefill_replicas: int = 1
    decode_replicas: int = 1
    transfer_bandwidth_bytes_per_second: float | None = None
    transfer_latency_seconds: float = 0.0
    overlap_rho: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeploymentSpec":
        transfer = data.get("transfer", {})
        spec = cls(
            mode=str(data.get("mode", "aggregated")).lower(),
            prefill_replicas=int(data.get("prefill_replicas", 1)),
            decode_replicas=int(data.get("decode_replicas", 1)),
            transfer_bandwidth_bytes_per_second=(
                float(transfer["effective_bandwidth_bytes_per_second"])
                if transfer.get("effective_bandwidth_bytes_per_second") is not None
                else None
            ),
            transfer_latency_seconds=float(transfer.get("latency_seconds", 0.0)),
            overlap_rho=float(transfer.get("overlap_rho", 1.0)),
        )
        if spec.mode not in {"aggregated", "disaggregated"}:
            raise ValueError("deployment.mode must be aggregated or disaggregated")
        positive("deployment.prefill_replicas", spec.prefill_replicas)
        positive("deployment.decode_replicas", spec.decode_replicas)
        if spec.transfer_latency_seconds < 0:
            raise ValueError("deployment.transfer.latency_seconds cannot be negative")
        if not 0 <= spec.overlap_rho <= 1:
            raise ValueError("deployment.transfer.overlap_rho must be in [0, 1]")
        if spec.mode == "disaggregated":
            if spec.transfer_bandwidth_bytes_per_second is None:
                raise ValueError("PD分离时必须填写deployment.transfer有效带宽")
            positive(
                "deployment.transfer.effective_bandwidth_bytes_per_second",
                spec.transfer_bandwidth_bytes_per_second,
            )
        return spec

