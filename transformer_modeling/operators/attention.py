"""标准Attention、KDA与Gated MLA粗粒度算子。"""

from __future__ import annotations

from math import ceil

from ..core.work_item import WorkItem
from ..parallel.tensor import require_divisible, tp_all_reduce_hidden
from .base import (
    OperatorContext, OperatorEstimate,
    TransformerOperator, blocked_bytes, effective_element_bytes,
    gemm, int_param, local_kv_heads, optional_int, paged_kv_tokens,
    resolve_kv_cache_dtype, resolve_weight_dtype, windowed_pairs, windowed_tokens,
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
        for field in ("sliding_window", "q_lora_rank", "o_lora_rank"):
            optional_int(spec, field)
        groups = int(spec.get("o_groups", 1))
        if groups <= 0 or qh % groups:
            raise ValueError("standard_attention o_groups must divide query_heads")
        if groups > 1 and groups % tp:
            raise ValueError("standard_attention o_groups must be divisible by tensor_parallel")
        resolve_weight_dtype(config, spec)
        resolve_kv_cache_dtype(config, spec)

    def estimate(self, spec, ctx):
        model, count = ctx.model, ctx.occurrence_count
        qh = int_param(spec, "query_heads")
        kvh = int_param(spec, "kv_heads", qh)
        dim = int_param(spec, "head_dim")
        window = optional_int(spec, "sliding_window")
        qr, orank = optional_int(spec, "q_lora_rank"), optional_int(spec, "o_lora_rank")
        groups = int(spec.get("o_groups", 1))
        shared_kv = bool(spec.get("shared_kv_projection", False))
        attention_sink = bool(spec.get("attention_sink", False))
        wdtype = resolve_weight_dtype(ctx.config, spec)
        kv_dtype = resolve_kv_cache_dtype(ctx.config, spec)
        local_qh, local_kvh = qh // ctx.tp, local_kv_heads(kvh, ctx.tp)
        q, kv, h, rows = local_qh * dim, local_kvh * dim, model.hidden_size, ctx.rows
        ba = effective_element_bytes(ctx.config, model.activation_dtype)
        bkv = effective_element_bytes(ctx.config, kv_dtype)
        global_q, global_kv = qh * dim, kvh * dim
        # 低秩时输入侧（h×qr）与输出侧（orank×h）权重在TP rank间复制，与MLA一致。
        q_global = h * qr + qr * global_q if qr else h * global_q
        q_local = h * qr + qr * q if qr else h * q
        local_groups = groups // ctx.tp if groups % ctx.tp == 0 else 1
        o_global = (
            global_q * orank + groups * orank * h if orank else global_q * h
        )
        o_local = q * orank + local_groups * orank * h if orank else q * h
        kv_paths = 1 if shared_kv else 2
        norms_and_sink = (
            (qr if qr else 0) + (global_kv if shared_kv else 0)
            + (qh if attention_sink else 0)
        )
        local_norms_and_sink = (
            (qr if qr else 0) + (kv if shared_kv else 0)
            + (local_qh if attention_sink else 0)
        )
        global_params = count * (
            q_global + kv_paths * h * global_kv + o_global + norms_and_sink
        )
        local_params = count * (
            q_local + kv_paths * h * kv + o_local + local_norms_and_sink
        )

        if qr:
            items = [
                gemm("layers.attn_q_a", rows, h, qr, ctx, weight_dtype=wdtype),
                gemm("layers.attn_q_b", rows, qr, q, ctx, weight_dtype=wdtype),
                gemm(
                    "layers.attn_kv_projection", rows, h, kv_paths * kv,
                    ctx, weight_dtype=wdtype,
                ),
            ]
        else:
            items = [gemm(
                "layers.qkv_projection", rows, h, q + kv_paths * kv,
                ctx, weight_dtype=wdtype,
            )]
        if orank:
            output = [
                gemm("layers.attn_o_a", rows, q, orank, ctx, weight_dtype=wdtype),
                gemm(
                    "layers.attn_o_b", rows, local_groups * orank, h,
                    ctx, weight_dtype=wdtype,
                ),
            ]
        else:
            output = [gemm("layers.attention_output_projection", rows, q, h, ctx, weight_dtype=wdtype)]
        flash = spec.implementation == "flash_attention" or (
            spec.implementation == "default" and ctx.config.execution.flash_attention
        )
        sparsity = 1.0
        if ctx.phase == "prefill":
            length = ctx.token_length
            pairs = windowed_pairs(length, window)
            logical = 4 * ctx.batch_size * q * pairs * count
            sparsity = pairs / (length * (length + 1) / 2)
            executed = logical if flash else round(
                4 * ctx.batch_size * length * length * q * count * sparsity
            )
            cache_tokens = length
        else:
            visible = windowed_tokens(ctx.attention_length, window)
            logical = executed = 4 * ctx.batch_size * q * visible * count
            cache_tokens = visible
        attention_bytes = (
            rows * q * ba + kv_paths * ctx.batch_size * cache_tokens * kv * bkv
            + rows * q * ba
        ) * count
        if not flash and ctx.phase == "prefill":
            attention_bytes += (
                2 * ctx.batch_size * local_qh * ctx.token_length ** 2
                * model.logits_bytes * count * sparsity
            )
        attention = WorkItem(
            "layers.attention", "attention", executed, attention_bytes, count, logical
        )
        items.append(attention)
        if not ctx.config.execution.rope_kv_write_fused:
            # 未融合时RoPE旋转与KV写入是独立的vector kernel：读写Q/K并写KV cache。
            rope_ops = 3 * rows * (q + kv) * count
            items.insert(len(items) - 1, WorkItem(
                "layers.rope_kv_write", "vector", rope_ops,
                (2 * rows * (q + kv) * ba + rows * kv_paths * kv * bkv) * count,
                count, rope_ops,
            ))
        items.extend(output)
        stored_tokens = paged_kv_tokens(
            ctx.config,
            windowed_tokens(
                ctx.config.serving.prompt_length + ctx.config.serving.output_length - 1,
                window,
            ),
        )
        state = blocked_bytes(
            ctx.config,
            ctx.batch_size * count * stored_tokens * kv_paths * kv,
            kv_dtype,
        )
        temp = ceil(max(rows * (q + 2 * kv) * ba, rows * h * ba))
        assumptions = ["KV Head按均匀分片或等组复制放置。"]
        if window:
            assumptions.append(
                f"滑窗{window} token：KV cache、注意力对数与读量均按窗口上限截断。"
            )
        if shared_kv:
            assumptions.append("K与V共享单个MQA投影和cache向量。")
        if groups > 1:
            assumptions.append(f"输出投影按{groups}组执行低秩wo_a，再由wo_b合并。")
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, global_params, local_params,
            {"attention_parameters": local_params}, state,
            {"kv_cache_bytes": state}, temp,
            items, tp_all_reduce_hidden(ctx, "attention"),
            assumptions, "high" if flash else "medium",
            local_parameters_by_dtype={wdtype: local_params},
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
