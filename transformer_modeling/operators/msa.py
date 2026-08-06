"""MiniMax Sparse Attention: GQA plus block-indexed sparse Softmax attention."""

from __future__ import annotations

from math import ceil

from ..core.work_item import WorkItem
from ..parallel.tensor import require_divisible, tp_all_reduce_hidden
from .base import (
    OperatorEstimate, TransformerOperator, blocked_bytes, effective_element_bytes,
    gemm, int_param, local_kv_heads, optional_int, paged_kv_tokens,
    resolve_kv_cache_dtype, resolve_weight_dtype,
)


class MiniMaxSparseAttentionOperator(TransformerOperator):
    """MSA selects KV blocks with an Index Branch before exact GQA attention."""

    type_id, chinese_name, slot = "minimax_sparse_attention", "MiniMax Sparse Attention (MSA)", "attention"
    implementations = {"default", "block_sparse"}

    def validate(self, spec, config):
        super().validate(spec, config)
        qh, kvh = int_param(spec, "query_heads"), int_param(spec, "kv_heads")
        if qh % kvh:
            raise ValueError("minimax_sparse_attention query_heads must be divisible by kv_heads")
        require_divisible("minimax_sparse_attention query_heads", qh, config.parallelism.tensor_parallel)
        if kvh % config.parallelism.tensor_parallel and config.parallelism.tensor_parallel % kvh:
            raise ValueError("minimax_sparse_attention KV heads cannot shard or group-replicate for TP")
        for field in ("head_dim", "indexer_heads", "indexer_head_dim", "block_size", "topk_blocks"):
            int_param(spec, field)
        rope_dim = optional_int(spec, "rope_dim")
        if rope_dim > int_param(spec, "head_dim"):
            raise ValueError("minimax_sparse_attention rope_dim cannot exceed head_dim")
        resolve_weight_dtype(config, spec)
        resolve_kv_cache_dtype(config, spec)

    def estimate(self, spec, ctx):
        model, count, h = ctx.model, ctx.occurrence_count, ctx.model.hidden_size
        qh, kvh, dim = int_param(spec, "query_heads"), int_param(spec, "kv_heads"), int_param(spec, "head_dim")
        index_heads, index_dim = int_param(spec, "indexer_heads"), int_param(spec, "indexer_head_dim")
        block_size, topk_blocks = int_param(spec, "block_size"), int_param(spec, "topk_blocks")
        rope_dim = optional_int(spec, "rope_dim")
        local_qh, local_kvh = qh // ctx.tp, local_kv_heads(kvh, ctx.tp)
        local_index_heads = index_heads // ctx.tp if index_heads % ctx.tp == 0 else index_heads
        q, kv, rows = local_qh * dim, local_kvh * dim, ctx.rows
        wdtype, kv_dtype = resolve_weight_dtype(ctx.config, spec), resolve_kv_cache_dtype(ctx.config, spec)
        ba = effective_element_bytes(ctx.config, model.activation_dtype)
        bkv = effective_element_bytes(ctx.config, kv_dtype)
        items = [
            gemm("layers.msa_qkv_projection", rows, h, q + 2 * kv, ctx, weight_dtype=wdtype),
            gemm("layers.msa_indexer_q", rows, h, local_index_heads * index_dim, ctx, weight_dtype=wdtype),
            gemm("layers.msa_indexer_k", rows, h, index_dim, ctx, weight_dtype=wdtype),
        ]
        if ctx.phase == "prefill":
            length = ctx.token_length
            pairs = length * (length + 1) // 2
            index_ops = 2 * ctx.batch_size * local_index_heads * index_dim * pairs * count
            average_blocks = min(topk_blocks, max(1, ceil((length + 1) / (2 * block_size))))
            main_keys = min(topk_blocks * block_size, max(1, (length + 1) // 2))
            index_read = ctx.batch_size * pairs * index_dim * bkv
        else:
            visible = max(1, ctx.attention_length)
            blocks = ceil(visible / block_size)
            index_ops = 2 * ctx.batch_size * local_index_heads * index_dim * visible * count
            average_blocks = min(topk_blocks, blocks)
            main_keys = min(topk_blocks * block_size, visible)
            index_read = ctx.batch_size * visible * index_dim * bkv
        items.append(WorkItem(
            "layers.msa_indexer_score", "attention", index_ops,
            (rows * local_index_heads * index_dim * ba + index_read) * count,
            count, index_ops,
        ))
        block_count = ceil(max(1, ctx.attention_length) / block_size)
        select_ops = 4 * rows * block_count * count
        items.append(WorkItem(
            "layers.msa_block_topk", "vector", select_ops,
            2 * rows * block_count * ba * count, count, select_ops,
        ))
        main_ops = 4 * rows * q * main_keys * count
        main_read = rows * main_keys * 2 * kv * bkv
        items.append(WorkItem(
            "layers.msa_sparse_attention", "attention", main_ops,
            (2 * rows * q * ba + main_read) * count, count, main_ops,
        ))
        if rope_dim and not ctx.config.execution.rope_kv_write_fused:
            rope_ops = 3 * rows * (q + kv) * rope_dim * count
            items.append(WorkItem(
                "layers.msa_rope_kv_write", "vector", rope_ops,
                (2 * rows * (q + kv) * ba + 2 * rows * kv * bkv) * count,
                count, rope_ops,
            ))
        items.append(gemm("layers.msa_output_projection", rows, q, h, ctx, weight_dtype=wdtype))
        stored = paged_kv_tokens(
            ctx.config, ctx.config.serving.prompt_length + ctx.config.serving.output_length - 1
        )
        state_breakdown = {
            "msa_gqa_kv_cache_bytes": blocked_bytes(
                ctx.config, ctx.batch_size * count * stored * 2 * kv, kv_dtype
            ),
            "msa_index_key_cache_bytes": blocked_bytes(
                ctx.config, ctx.batch_size * count * stored * index_dim, kv_dtype
            ),
        }
        global_params = h * (qh * dim + 2 * kvh * dim + index_heads * index_dim + index_dim) + qh * dim * h
        local_params = h * (q + 2 * kv + local_index_heads * index_dim + index_dim) + q * h
        assumptions = [
            f"MSA按{block_size}-token KV block的max-score Top-{topk_blocks}选择建模；当前block包含在Top-K预算内。",
            "Main Branch仅对选中block内token执行精确GQA Softmax Attention；所有GQA KV和独立Indexer K均驻留cache。",
            "Indexer按每个query扫描可见Index-K cache近似，未单独建模Flash block-sparse kernel的KV-outer gather-Q复用增益。",
            f"Prefill按平均命中{average_blocks}个block估计Main Branch；Decode按当前可见block上限截断。",
        ]
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, global_params * count, local_params * count,
            {"msa_attention_parameters": local_params * count}, sum(state_breakdown.values()),
            state_breakdown, ceil(rows * max(q + 2 * kv, local_index_heads * index_dim) * ba), items,
            tp_all_reduce_hidden(ctx, "minimax_sparse_attention"), assumptions, "medium",
            local_parameters_by_dtype={wdtype: local_params * count},
        )
