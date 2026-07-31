"""压缩注意力算子：CSA（压缩稀疏）与HCA（重度压缩）共享同一账单骨架。"""

from __future__ import annotations

from math import ceil

from ..core.work_item import WorkItem
from ..parallel.tensor import require_divisible, tp_all_reduce_hidden
from .base import (
    OperatorEstimate, TransformerOperator, blocked_bytes, effective_element_bytes,
    gemm, int_param, local_kv_heads, optional_int, paged_kv_tokens,
    resolve_weight_dtype, windowed_tokens,
)

SELECTORS = {"indexer", "uniform"}


class CompressedAttentionOperator(TransformerOperator):
    """Compressor把每m个token压成一个entry，再按Indexer或等距采样选k个entry。"""

    type_id, chinese_name, slot = "compressed_attention", "压缩注意力", "attention"
    implementations = {"default", "compressed_kv"}
    defaults: dict[str, object] = {
        "kv_heads": 1, "compress_ratio": 4, "compress_overlap": 2,
        "selected_entries": 1024, "sliding_window": 128, "selector": "indexer",
        "indexer_heads": 128, "indexer_head_dim": 128, "qk_rope_head_dim": 64,
    }

    def _int(self, spec, name: str) -> int:
        return optional_int(spec, name, int(self.defaults.get(name, 0)))

    def _selector(self, spec) -> str:
        return str(spec.get("selector", self.defaults["selector"])).lower()

    def validate(self, spec, config):
        super().validate(spec, config)
        qh = int_param(spec, "query_heads")
        kvh = self._int(spec, "kv_heads")
        dim = int_param(spec, "head_dim")
        if kvh <= 0 or qh % kvh:
            raise ValueError(f"{self.type_id} query_heads must be divisible by kv_heads")
        require_divisible(f"{self.type_id} query_heads", qh, config.parallelism.tensor_parallel)
        if self._int(spec, "qk_rope_head_dim") > dim:
            raise ValueError(f"{self.type_id} qk_rope_head_dim cannot exceed head_dim")
        for field in ("compress_ratio", "compress_overlap", "selected_entries"):
            if self._int(spec, field) <= 0:
                raise ValueError(f"{self.type_id}.{field} must be greater than zero")
        self._int(spec, "sliding_window")
        for field in ("q_lora_rank", "o_lora_rank"):
            optional_int(spec, field)
        selector = self._selector(spec)
        if selector not in SELECTORS:
            choices = ", ".join(sorted(SELECTORS))
            raise ValueError(f"{self.type_id}.selector must be one of: {choices}")
        if selector == "indexer":
            for field in ("indexer_heads", "indexer_head_dim"):
                if self._int(spec, field) <= 0:
                    raise ValueError(f"{self.type_id}.{field} must be greater than zero")
        resolve_weight_dtype(config, spec)

    def estimate(self, spec, ctx):
        model, count, h = ctx.model, ctx.occurrence_count, ctx.model.hidden_size
        qh, kvh = int_param(spec, "query_heads"), self._int(spec, "kv_heads")
        dim = int_param(spec, "head_dim")
        ratio, overlap = self._int(spec, "compress_ratio"), self._int(spec, "compress_overlap")
        selected, window = self._int(spec, "selected_entries"), self._int(spec, "sliding_window")
        rope_dim = self._int(spec, "qk_rope_head_dim")
        qr, orank = optional_int(spec, "q_lora_rank"), optional_int(spec, "o_lora_rank")
        selector = self._selector(spec)
        idx_heads, idx_dim = self._int(spec, "indexer_heads"), self._int(spec, "indexer_head_dim")
        wdtype = resolve_weight_dtype(ctx.config, spec)
        local_qh, local_kvh = qh // ctx.tp, local_kv_heads(kvh, ctx.tp)
        q, kv, rows = local_qh * dim, local_kvh * dim, ctx.rows
        entry, global_entry = overlap * kv, overlap * kvh * dim
        ba = effective_element_bytes(ctx.config, model.activation_dtype)
        bkv = effective_element_bytes(ctx.config, model.kv_dtype)
        local_idx = idx_heads // ctx.tp if idx_heads and idx_heads % ctx.tp == 0 else idx_heads

        items = []
        if qr:
            items.append(gemm("layers.csa_q_a", rows, h, qr, ctx, weight_dtype=wdtype))
            items.append(gemm("layers.csa_q_b", rows, qr, q, ctx, weight_dtype=wdtype))
        else:
            items.append(gemm("layers.csa_q_projection", rows, h, q, ctx, weight_dtype=wdtype))
        # Compressor：wkv与wgate合并成一次投影，再由门控池化写出压缩entry。
        items.append(gemm("layers.csa_compress_kv", rows, h, 2 * entry, ctx, weight_dtype=wdtype))
        pool_ops = 5 * rows * entry * count
        items.append(WorkItem(
            "layers.csa_compress_pool", "vector", pool_ops,
            (2 * rows * entry * ba + ceil(rows / ratio) * entry * bkv) * count,
            count, pool_ops,
        ))
        entries_total = ceil(max(1, ctx.attention_length) / ratio)
        # Prefill每个query平均只能看到一半的历史entry，Decode能看到全部。
        entries_avg = ceil(entries_total / 2) if ctx.phase == "prefill" else entries_total
        if selector == "indexer":
            items.append(gemm(
                "layers.csa_indexer_q", rows, h, local_idx * idx_dim, ctx, weight_dtype=wdtype
            ))
            items.append(gemm(
                "layers.csa_indexer_key", rows, h, idx_dim, ctx, weight_dtype=wdtype
            ))
            score_ops = 2 * rows * entries_avg * local_idx * idx_dim * count
            items.append(WorkItem(
                "layers.csa_indexer_score", "attention", score_ops,
                (rows * local_idx * idx_dim * ba
                 + ctx.batch_size * entries_total * idx_dim * bkv
                 + rows * entries_avg * ba) * count,
                count, score_ops,
            ))
            topk_ops = 4 * rows * entries_avg * count
            items.append(WorkItem(
                "layers.csa_topk_select", "vector", topk_ops,
                2 * rows * entries_avg * ba * count, count, topk_ops,
            ))
        # 滑窗在压缩注意力里是叠加的局部分支，0表示不开这条分支（而非退化为全局）。
        visible_window = windowed_tokens(ctx.attention_length, window) if window else 0
        attended = min(selected, entries_avg) + visible_window
        attn_ops = 4 * rows * q * attended * count
        entries_read = ctx.batch_size * (
            entries_total if ctx.phase == "prefill" else min(selected, entries_total)
        )
        items.append(WorkItem(
            "layers.csa_attention", "attention", attn_ops,
            (2 * rows * q * ba + entries_read * 2 * entry * bkv
             + 2 * ctx.batch_size * visible_window * kv * bkv) * count,
            count, attn_ops,
        ))
        if rope_dim and not ctx.config.execution.rope_kv_write_fused:
            rope_ops = 3 * rows * (local_qh + local_kvh) * rope_dim * count
            items.append(WorkItem(
                "layers.csa_rope", "vector", rope_ops,
                2 * rows * (local_qh + local_kvh) * rope_dim * ba * count, count, rope_ops,
            ))
        if orank:
            items.append(gemm("layers.csa_o_a", rows, q, orank, ctx, weight_dtype=wdtype))
            items.append(gemm("layers.csa_o_b", rows, orank, h, ctx, weight_dtype=wdtype))
        else:
            items.append(gemm("layers.csa_o_projection", rows, q, h, ctx, weight_dtype=wdtype))

        # 低秩输入/输出侧与Compressor、Indexer-key在TP rank间复制，Q/O按头切分。
        q_global = h * qr + qr * qh * dim if qr else h * qh * dim
        q_local = h * qr + qr * q if qr else h * q
        o_global = qh * dim * orank + orank * h if orank else qh * dim * h
        o_local = q * orank + orank * h if orank else q * h
        compress_global = h * 2 * global_entry + ratio * global_entry
        compress_local = h * 2 * entry + ratio * entry
        index_global = index_local = 0
        if selector == "indexer":
            index_global = h * idx_heads * idx_dim + h * idx_dim + idx_heads
            index_local = h * local_idx * idx_dim + h * idx_dim + local_idx
        global_params = count * (q_global + o_global + compress_global + index_global)
        local_params = count * (q_local + o_local + compress_local + index_local)

        stored = paged_kv_tokens(
            ctx.config,
            ctx.config.serving.prompt_length + ctx.config.serving.output_length - 1,
        )
        stored_entries = ceil(stored / ratio)
        state_breakdown = {
            "compressed_kv_cache_bytes": blocked_bytes(
                ctx.config, ctx.batch_size * count * stored_entries * 2 * entry, model.kv_dtype
            ),
            "sliding_window_kv_cache_bytes": blocked_bytes(
                ctx.config,
                ctx.batch_size * count * (windowed_tokens(stored, window) if window else 0) * 2 * kv,
                model.kv_dtype,
            ),
        }
        if selector == "indexer":
            state_breakdown["indexer_key_cache_bytes"] = blocked_bytes(
                ctx.config, ctx.batch_size * count * stored_entries * idx_dim, model.kv_dtype
            )
        state = sum(state_breakdown.values())
        assumptions = [
            "Compressor与Indexer Key按每token摊销，不单独建模跨token的重复压缩。",
            "Top-K命中数按当前可见entry的平均值取min(k, entries)。",
            "位置等距采样与Top-K选择的成本按同一量级计。",
            "Indexer的Hadamard变换与低精度量化按等效GEMM近似。",
        ]
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, global_params, local_params,
            {f"{self.type_id}_parameters": local_params}, state, state_breakdown,
            ceil(max(rows * (q + 2 * entry) * ba, rows * h * ba)), items,
            tp_all_reduce_hidden(ctx, self.type_id), assumptions, "low",
            local_parameters_by_dtype={wdtype: local_params},
        )


class CSAOperator(CompressedAttentionOperator):
    """细粒度压缩（m小、entry重叠）配Lightning Indexer做Top-K稀疏选择。"""

    type_id, chinese_name = "csa_attention", "CSA压缩稀疏注意力"
    defaults = dict(
        CompressedAttentionOperator.defaults,
        compress_ratio=4, compress_overlap=2, selector="indexer",
    )


class HCAOperator(CompressedAttentionOperator):
    """重度压缩（m大、无重叠）用位置等距采样替代Indexer。"""

    type_id, chinese_name = "hca_attention", "HCA重度压缩注意力"
    defaults = dict(
        CompressedAttentionOperator.defaults,
        compress_ratio=128, compress_overlap=1, selector="uniform", sliding_window=0,
    )
