"""Gated MLA及latent KV cache账单。"""

from __future__ import annotations

from math import ceil

from ..core.work_item import WorkItem
from ..parallel.tensor import require_divisible, tp_all_reduce_hidden
from .base import (
    OperatorContext, OperatorEstimate, TransformerOperator, blocked_bytes,
    effective_element_bytes, gemm, int_param, paged_kv_tokens,
)


class GatedMLAOperator(TransformerOperator):
    type_id, chinese_name, slot = "gated_mla", "Gated MLA", "attention"
    implementations = {"default", "latent_cache"}

    def validate(self, spec, config):
        super().validate(spec, config)
        heads = int_param(spec, "query_heads")
        require_divisible("gated_mla query_heads", heads, config.parallelism.tensor_parallel)
        for field in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
            int_param(spec, field)

    def estimate(self, spec, ctx):
        model, count, h = ctx.model, ctx.occurrence_count, ctx.model.hidden_size
        heads_global = int_param(spec, "query_heads")
        heads = heads_global // ctx.tp
        qr, kr = int_param(spec, "q_lora_rank"), int_param(spec, "kv_lora_rank")
        nope, rope = int_param(spec, "qk_nope_head_dim"), int_param(spec, "qk_rope_head_dim")
        vd, dq = int_param(spec, "v_head_dim"), nope + rope
        cache = ctx.config.serving.prefix_cache
        length = ctx.token_length
        if ctx.phase == "prefill":
            saved = round(cache.mla_prefix_hit_rate * min(length, cache.mla_average_matched_tokens))
            length = max(1, length - saved)
        rows = ctx.batch_size * length
        local_ctx = OperatorContext(ctx.config, ctx.phase, ctx.batch_size, length,
                                    ctx.attention_length, count)
        q_width_global = heads_global * dq
        kv_width_global = heads_global * (nope + vd)
        global_per = h * qr + qr * q_width_global + h * (kr + rope) + kr * kv_width_global + h * heads_global + heads_global * vd * h
        replicated = h * (qr + kr + rope)
        local_per = replicated + (global_per - replicated) // ctx.tp
        ba = effective_element_bytes(ctx.config, model.activation_dtype)
        bkv = effective_element_bytes(ctx.config, model.kv_dtype)
        items = [
            gemm("layers.mla_q_a", rows, h, qr, local_ctx),
            gemm("layers.mla_q_b", rows, qr, heads * dq, local_ctx),
            gemm("layers.mla_kv_a", rows, h, kr + rope, local_ctx, bkv),
            gemm("layers.mla_kv_b", rows, kr, heads * (nope + vd), local_ctx),
        ]
        if ctx.phase == "prefill":
            attn_ops = ctx.batch_size * heads * length * (length + 1) * (dq + vd)
            cache_tokens = length
        else:
            attn_ops = 2 * ctx.batch_size * heads * ctx.attention_length * (dq + vd)
            cache_tokens = ctx.attention_length
        cache_read = ctx.batch_size * cache_tokens * (kr + rope) * bkv
        items.append(WorkItem(
            "layers.mla_attention", "attention", attn_ops * count,
            (rows * heads * (dq + vd) * ba + cache_read) * count,
            count, attn_ops * count,
        ))
        gate_ops = 6 * rows * heads * vd * count
        items.append(WorkItem("layers.mla_gate", "vector", gate_ops,
                              2 * rows * heads * vd * ba * count, count, gate_ops))
        items.append(gemm("layers.mla_output", rows, heads * vd, h, local_ctx))
        stored = paged_kv_tokens(
            ctx.config,
            ctx.config.serving.prompt_length + ctx.config.serving.output_length - 1,
        )
        state = blocked_bytes(
            ctx.config, ctx.batch_size * count * stored * (kr + rope), model.kv_dtype
        )
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, global_per * count,
            local_per * count, {"mla_parameters": local_per * count}, state,
            {"mla_latent_kv_cache_bytes": state}, ceil(rows * max(qr, kr, heads * dq) * ba),
            items, tp_all_reduce_hidden(local_ctx, "mla_attention"),
            ["MLA latent cache在TP rank间复制。"], "low",
        )
