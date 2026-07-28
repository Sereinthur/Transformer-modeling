"""标准Attention、KDA与Gated MLA粗粒度算子。"""

from __future__ import annotations

from math import ceil

from ..core.work_item import WorkItem
from ..parallel.tensor import require_divisible, tp_all_reduce_hidden
from .base import (
    OperatorContext, OperatorEstimate,
    TransformerOperator, blocked_bytes, effective_element_bytes,
    gemm, int_param, local_kv_heads,
)


class StandardAttentionOperator(TransformerOperator):
    type_id, chinese_name, slot = "standard_attention", "标准Attention", "attention"
    implementations = {"default", "standard", "flash_attention"}

    def validate(self, spec, config):
        super().validate(spec, config)
        qh = int_param(spec, "query_heads")
        kvh = int_param(spec, "kv_heads", qh)
        dim = int_param(spec, "head_dim")
        if qh % kvh:
            raise ValueError("standard_attention query_heads must be divisible by kv_heads")
        tp = config.parallelism.tensor_parallel
        require_divisible("standard_attention query_heads", qh, tp)
        if kvh % tp and tp % kvh:
            raise ValueError("standard_attention KV heads cannot shard or group-replicate for TP")
        if bool(spec.get("query_width_equals_hidden", True)) and qh * dim != config.model.hidden_size:
            raise ValueError("standard_attention query_heads * head_dim must equal hidden_size")

    def estimate(self, spec, ctx):
        model, count = ctx.model, ctx.occurrence_count
        qh = int_param(spec, "query_heads")
        kvh = int_param(spec, "kv_heads", qh)
        dim = int_param(spec, "head_dim")
        local_qh, local_kvh = qh // ctx.tp, local_kv_heads(kvh, ctx.tp)
        q, kv, h, rows = local_qh * dim, local_kvh * dim, model.hidden_size, ctx.rows
        ba = effective_element_bytes(ctx.config, model.activation_dtype)
        bkv = effective_element_bytes(ctx.config, model.kv_dtype)
        global_q, global_kv = qh * dim, kvh * dim
        global_params = count * (h * (global_q + 2 * global_kv) + global_q * h)
        local_params = count * (h * (q + 2 * kv) + q * h)

        qkv = gemm("layers.qkv_projection", rows, h, q + 2 * kv, ctx)
        output = gemm("layers.attention_output_projection", rows, q, h, ctx)
        flash = spec.implementation == "flash_attention" or (
            spec.implementation == "default" and ctx.config.execution.flash_attention
        )
        if ctx.phase == "prefill":
            length = ctx.token_length
            logical = 2 * ctx.batch_size * q * length * (length + 1) * count
            executed = logical if flash else 4 * ctx.batch_size * length * length * q * count
            cache_tokens = length
        else:
            logical = executed = 4 * ctx.batch_size * q * ctx.attention_length * count
            cache_tokens = ctx.attention_length
        attention_bytes = (
            rows * q * ba + 2 * ctx.batch_size * cache_tokens * kv * bkv
            + rows * q * ba
        ) * count
        if not flash and ctx.phase == "prefill":
            attention_bytes += (
                2 * ctx.batch_size * local_qh * ctx.token_length ** 2
                * model.logits_bytes * count
            )
        attention = WorkItem(
            "layers.attention", "attention", executed, attention_bytes, count, logical
        )
        stored_tokens = ctx.config.serving.prompt_length + ctx.config.serving.output_length - 1
        state = blocked_bytes(
            ctx.config,
            ctx.batch_size * count * stored_tokens * 2 * kv,
            model.kv_dtype,
        )
        temp = ceil(max(rows * (q + 2 * kv) * ba, rows * h * ba))
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, global_params, local_params,
            {"attention_parameters": local_params}, state,
            {"kv_cache_bytes": state}, temp,
            [qkv, attention, output], tp_all_reduce_hidden(ctx, "attention"),
            ["KV Head按均匀分片或等组复制放置。"], "high" if flash else "medium",
        )


class KDAOperator(TransformerOperator):
    type_id, chinese_name, slot = "kda", "KDA线性注意力", "attention"
    implementations = {"default", "chunkwise"}

    def validate(self, spec, config):
        super().validate(spec, config)
        heads = int_param(spec, "heads")
        require_divisible("kda heads", heads, config.parallelism.tensor_parallel)
        for field in ("key_dim", "value_dim", "short_conv_kernel_size", "chunk_size"):
            int_param(spec, field)

    def estimate(self, spec, ctx):
        h, count = ctx.model.hidden_size, ctx.occurrence_count
        heads_global = int_param(spec, "heads")
        heads = heads_global // ctx.tp
        key = int_param(spec, "key_dim")
        value = int_param(spec, "value_dim")
        conv_size = int_param(spec, "short_conv_kernel_size")
        chunk = int_param(spec, "chunk_size")
        cache = ctx.config.serving.prefix_cache
        length = ctx.token_length
        if ctx.phase == "prefill":
            saved = round(cache.kda_state_hit_rate * min(length, cache.kda_cached_prefix_tokens))
            length = max(1, length - saved)
        rows = ctx.batch_size * length
        ba = effective_element_bytes(ctx.config, ctx.model.activation_dtype)
        bw = effective_element_bytes(ctx.config, ctx.model.weight_dtype)
        bs = effective_element_bytes(ctx.config, ctx.model.kda_state_dtype)
        projected_global = heads_global * (2 * key + value)
        projected = heads * (2 * key + value)
        global_params = count * (
            h * (projected_global + 2 * heads_global * value)
            + projected_global * conv_size
        )
        local_params = count * (
            h * (projected + 2 * heads * value) + projected * conv_size
        )
        local_ctx = OperatorContext(ctx.config, ctx.phase, ctx.batch_size, length,
                                    ctx.attention_length, count)
        projection = gemm("layers.kda_projection", rows, h, projected, local_ctx)
        gate = gemm("layers.kda_gate", rows, h, heads * value, local_ctx)
        output = gemm("layers.kda_output", rows, heads * value, h, local_ctx)
        conv_ops = 2 * rows * projected * conv_size * count
        conv = WorkItem(
            "layers.kda_short_conv", "vector", conv_ops,
            (2 * rows * projected * ba + projected * conv_size * bw) * count,
            count, conv_ops,
        )
        state_elements = ctx.batch_size * heads * key * value
        state_visits = ceil(length / chunk) if ctx.phase == "prefill" else 1
        recurrence_ops = 7 * rows * heads * key * value * count
        recurrence = WorkItem(
            "layers.kda_recurrence", "attention", recurrence_ops,
            (rows * projected * ba + 2 * state_visits * state_elements * bs
             + rows * heads * value * ba) * count,
            count * state_visits, recurrence_ops,
        )
        conv_state = ctx.batch_size * heads * (2 * key + value) * max(0, conv_size - 1)
        state = blocked_bytes(
            ctx.config, count * (state_elements + conv_state), ctx.model.kda_state_dtype
        )
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, global_params, local_params,
            {"kda_parameters": local_params}, state, {"kda_state_bytes": state},
            ceil(rows * projected * ba), [projection, gate, conv, recurrence, output],
            tp_all_reduce_hidden(local_ctx, "kda_attention"),
            ["KDA Prefill按Chunkwise Scan、Decode按固定recurrent state近似。"], "low",
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
        stored = ctx.config.serving.prompt_length + ctx.config.serving.output_length - 1
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
