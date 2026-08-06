"""GLM DSA: MLA projections plus token-level sparse attention and IndexShare."""

from __future__ import annotations

from math import ceil

from ..core.work_item import WorkItem
from ..parallel.tensor import require_divisible, tp_all_reduce_hidden
from .base import (
    OperatorContext, OperatorEstimate, TransformerOperator, blocked_bytes,
    effective_element_bytes, gemm, int_param, paged_kv_tokens,
    resolve_weight_dtype,
)


class DSAAttentionOperator(TransformerOperator):
    """GLM-5.x DSA with MLA KV compression.

    ``indexer_mode=full`` estimates the Indexer projections and token-level
    score/top-k pass. ``shared`` reuses the nearest full layer's selected
    indices (IndexShare), so only MLA + sparse attention remains in that layer.
    """

    # Keep the card name identical to the architecture/configuration term.
    # MLA is the implementation used by DSA here, rather than a second editor
    # operator that users need to place in the linear sequence.
    type_id, chinese_name, slot = "dsa_attention", "DSA", "attention"
    implementations = {"default", "mla_dsa"}
    _INDEXER_MODES = {"full", "shared"}

    def validate(self, spec, config):
        super().validate(spec, config)
        heads = int_param(spec, "query_heads")
        require_divisible("dsa_attention query_heads", heads, config.parallelism.tensor_parallel)
        for field in (
            "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
            "qk_rope_head_dim", "v_head_dim", "indexer_heads",
            "indexer_head_dim", "index_topk",
        ):
            if int_param(spec, field) <= 0:
                raise ValueError(f"dsa_attention.{field} must be greater than zero")
        mode = str(spec.get("indexer_mode", "full")).lower()
        if mode not in self._INDEXER_MODES:
            raise ValueError("dsa_attention.indexer_mode must be full or shared")
        resolve_weight_dtype(config, spec)

    def estimate(self, spec, ctx):
        model, count, h = ctx.model, ctx.occurrence_count, ctx.model.hidden_size
        heads_global = int_param(spec, "query_heads")
        heads = heads_global // ctx.tp
        qr, kr = int_param(spec, "q_lora_rank"), int_param(spec, "kv_lora_rank")
        nope, rope = int_param(spec, "qk_nope_head_dim"), int_param(spec, "qk_rope_head_dim")
        vd, dq = int_param(spec, "v_head_dim"), nope + rope
        idx_heads_global = int_param(spec, "indexer_heads")
        if idx_heads_global % ctx.tp:
            # The official 32 Indexer heads can be replicated when TP does not
            # divide 32; that is less surprising than rejecting otherwise legal
            # deployment plans for this approximate performance operator.
            idx_heads = idx_heads_global
        else:
            idx_heads = idx_heads_global // ctx.tp
        idx_dim, topk = int_param(spec, "indexer_head_dim"), int_param(spec, "index_topk")
        mode = str(spec.get("indexer_mode", "full")).lower()
        is_full = mode == "full"
        wdtype = resolve_weight_dtype(ctx.config, spec)
        ba = effective_element_bytes(ctx.config, model.activation_dtype)
        bkv = effective_element_bytes(ctx.config, model.kv_dtype)
        rows, length = ctx.rows, ctx.token_length
        local_ctx = OperatorContext(
            ctx.config, ctx.phase, ctx.batch_size, length, ctx.attention_length, count
        )

        q_width_global = heads_global * dq
        kv_width_global = heads_global * (nope + vd)
        mla_global = (
            h * qr + qr * q_width_global + h * (kr + rope)
            + kr * kv_width_global + h * heads_global + heads_global * vd * h
        )
        mla_replicated = h * (qr + kr + rope)
        mla_local = mla_replicated + (mla_global - mla_replicated) // ctx.tp

        # The Indexer reuses MLA q_resid, projects a shared K, produces 32
        # head weights, then scores every token only in full layers.
        index_global = qr * idx_heads_global * idx_dim + h * idx_dim + h * idx_heads_global + idx_dim
        index_local = qr * idx_heads * idx_dim + h * idx_dim + h * idx_heads + idx_dim
        global_per = mla_global + (index_global if is_full else 0)
        local_per = mla_local + (index_local if is_full else 0)

        items = [
            gemm("layers.dsa_mla_q_a", rows, h, qr, local_ctx, weight_dtype=wdtype),
            gemm("layers.dsa_mla_q_b", rows, qr, heads * dq, local_ctx, weight_dtype=wdtype),
            gemm("layers.dsa_mla_kv_a", rows, h, kr + rope, local_ctx, bkv, wdtype),
            gemm("layers.dsa_mla_kv_b", rows, kr, heads * (nope + vd), local_ctx, weight_dtype=wdtype),
        ]
        if is_full:
            items.extend((
                gemm("layers.dsa_indexer_q", rows, qr, idx_heads * idx_dim, local_ctx, weight_dtype=wdtype),
                gemm("layers.dsa_indexer_k", rows, h, idx_dim, local_ctx, weight_dtype=wdtype),
                gemm("layers.dsa_indexer_weights", rows, h, idx_heads, local_ctx, weight_dtype=wdtype),
            ))
            if ctx.phase == "prefill":
                score_ops = ctx.batch_size * idx_heads * idx_dim * length * (length + 1) * count
                visible_for_topk = max(1, length // 2)
            else:
                score_ops = 2 * ctx.batch_size * idx_heads * idx_dim * ctx.attention_length * count
                visible_for_topk = max(1, ctx.attention_length)
            items.append(WorkItem(
                "layers.dsa_indexer_score", "attention", score_ops,
                (rows * idx_heads * idx_dim * ba
                 + ctx.batch_size * visible_for_topk * idx_dim * bkv) * count,
                count, score_ops,
            ))
            select_ops = 4 * rows * visible_for_topk * count
            items.append(WorkItem(
                "layers.dsa_topk_select", "vector", select_ops,
                2 * rows * visible_for_topk * ba * count, count, select_ops,
            ))

        if ctx.phase == "prefill":
            # Causal top-k is capped by the available prefix at each position.
            average_keys = min(topk, max(1, (length + 1) // 2))
            sparse_ops = 2 * ctx.batch_size * heads * length * average_keys * (dq + vd) * count
            cache_tokens = average_keys
        else:
            visible = min(topk, max(1, ctx.attention_length))
            sparse_ops = 2 * ctx.batch_size * heads * visible * (dq + vd) * count
            cache_tokens = visible
        cache_read = ctx.batch_size * cache_tokens * (kr + rope) * bkv
        items.append(WorkItem(
            "layers.dsa_sparse_attention", "attention", sparse_ops,
            (rows * heads * (dq + vd) * ba + cache_read) * count,
            count, sparse_ops,
        ))
        items.append(gemm("layers.dsa_output", rows, heads * vd, h, local_ctx, weight_dtype=wdtype))

        stored = paged_kv_tokens(
            ctx.config, ctx.config.serving.prompt_length + ctx.config.serving.output_length - 1
        )
        mla_state = blocked_bytes(
            ctx.config, ctx.batch_size * count * stored * (kr + rope), model.kv_dtype
        )
        state_breakdown = {"dsa_mla_latent_kv_cache_bytes": mla_state}
        if is_full:
            index_state = blocked_bytes(
                ctx.config, ctx.batch_size * count * stored * idx_dim, model.kv_dtype
            )
            state_breakdown["dsa_indexer_key_cache_bytes"] = index_state
        state = sum(state_breakdown.values())
        assumptions = [
            "DSA 主注意力按 token-level Top-K 精确注意力近似；不展开选择索引的访存布局。",
            "IndexShare shared 层复用最近 full 层的 Top-K 索引，因此不重复计 Indexer 投影、打分或 Indexer Key cache。",
        ]
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, global_per * count, local_per * count,
            {
                "dsa_mla_parameters": mla_local * count,
                "dsa_indexer_parameters": (index_local * count) if is_full else 0,
            }, state, state_breakdown,
            ceil(rows * max(qr, heads * dq, idx_heads * idx_dim) * ba), items,
            tp_all_reduce_hidden(local_ctx, "dsa_attention"), assumptions, "medium",
            local_parameters_by_dtype={wdtype: local_per * count},
        )
