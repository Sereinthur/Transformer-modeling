"""Dense、Gated与MoE FFN算子。"""

from __future__ import annotations

from math import ceil

from ..core.work_item import WorkItem
from ..parallel.tensor import require_divisible, tp_all_reduce_hidden
from .base import (
    CommunicationRequest, OperatorContext, OperatorEstimate,
    TransformerOperator, effective_element_bytes, gemm, int_param,
)


class DenseFFNOperator(TransformerOperator):
    type_id, chinese_name, slot = "dense_ffn", "Dense FFN", "ffn"
    implementations = {"default", "gelu"}

    def validate(self, spec, config):
        super().validate(spec, config)
        width = int(spec.get("intermediate_size", config.model.intermediate_size))
        if width <= 0:
            raise ValueError("dense_ffn intermediate_size must be positive")
        require_divisible("dense_ffn intermediate_size", width, config.parallelism.tensor_parallel)

    def estimate(self, spec, ctx):
        h, count = ctx.model.hidden_size, ctx.occurrence_count
        width = int(spec.get("intermediate_size", ctx.model.intermediate_size))
        local = width // ctx.tp
        params = 2 * h * width * count
        local_params = 2 * h * local * count
        items = [
            gemm("layers.ffn_up", ctx.rows, h, local, ctx),
            gemm("layers.ffn_down", ctx.rows, local, h, ctx),
        ]
        ba = effective_element_bytes(ctx.config, ctx.model.activation_dtype)
        activation_ops = 8 * ctx.rows * local * count
        items.insert(1, WorkItem(
            "layers.ffn_activation", "vector", activation_ops,
            2 * ctx.rows * local * ba * count, count, activation_ops,
        ))
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, params, local_params,
            {"dense_ffn_parameters": local_params},
            temporary_bytes=ceil(ctx.rows * local * ba), work_items=items,
            communication_requests=tp_all_reduce_hidden(ctx, "ffn"), confidence="high",
        )


class GatedFFNOperator(TransformerOperator):
    type_id, chinese_name, slot = "gated_ffn", "Gated/SwiGLU FFN", "ffn"
    implementations = {"default", "swiglu", "fused_gate_up"}

    def validate(self, spec, config):
        super().validate(spec, config)
        width = int(spec.get("intermediate_size", config.model.intermediate_size))
        if width <= 0 or width % config.parallelism.tensor_parallel:
            raise ValueError("gated_ffn intermediate_size must be positive and divisible by TP")

    def estimate(self, spec, ctx):
        h, count = ctx.model.hidden_size, ctx.occurrence_count
        width = int(spec.get("intermediate_size", ctx.model.intermediate_size))
        local = width // ctx.tp
        params, local_params = 3 * h * width * count, 3 * h * local * count
        gate_up = gemm("layers.ffn_gate_up", ctx.rows, h, 2 * local, ctx)
        down = gemm("layers.ffn_down", ctx.rows, local, h, ctx)
        ba = effective_element_bytes(ctx.config, ctx.model.activation_dtype)
        ops = 8 * ctx.rows * local * count
        activation = WorkItem(
            "layers.ffn_swiglu", "vector", ops,
            3 * ctx.rows * local * ba * count, count, ops,
        )
        temp_factor = 1 if (
            spec.implementation == "fused_gate_up" or ctx.config.execution.gated_mlp_fused
        ) else 2
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, params, local_params,
            {"gated_ffn_parameters": local_params},
            temporary_bytes=ceil(temp_factor * ctx.rows * local * ba),
            work_items=[gate_up, activation, down],
            communication_requests=tp_all_reduce_hidden(ctx, "ffn"), confidence="high",
        )


class MoEOperator(TransformerOperator):
    type_id, chinese_name, slot = "moe", "Mixture of Experts", "ffn"
    implementations = {"default", "standard_moe", "latent_moe_approx"}

    def validate(self, spec, config):
        super().validate(spec, config)
        experts = int_param(spec, "expert_count")
        topk = int_param(spec, "experts_per_token")
        width = int_param(spec, "expert_intermediate_size")
        shared = int(spec.get("shared_expert_intermediate_size", 0))
        if topk > experts:
            raise ValueError("moe experts_per_token cannot exceed expert_count")
        if experts % config.parallelism.expert_parallel:
            raise ValueError("moe expert_count must be divisible by EP")
        require_divisible("moe expert_intermediate_size", width, config.parallelism.tensor_parallel)
        if shared < 0:
            raise ValueError("moe shared_expert_intermediate_size must be non-negative")
        if shared:
            require_divisible("moe shared_expert_intermediate_size", shared, config.parallelism.tensor_parallel)
        latent = int(spec.get("latent_size", 0))
        if latent < 0:
            raise ValueError("moe latent_size must be non-negative")
        if latent:
            require_divisible("moe latent_size", latent, config.parallelism.tensor_parallel)

    def estimate(self, spec, ctx):
        model, h, count = ctx.model, ctx.model.hidden_size, ctx.occurrence_count
        experts = int_param(spec, "expert_count")
        topk = int_param(spec, "experts_per_token")
        width = int_param(spec, "expert_intermediate_size")
        shared = int(spec.get("shared_expert_intermediate_size", 0))
        latent = int(spec.get("latent_size", 0))
        gated = str(spec.get("activation", "swiglu")).lower() in {"swiglu", "gated"}
        factor = 3 if gated else 2
        io = latent or h  # 专家I/O维：LatentMoE在latent域计算，否则为hidden。
        local_experts, local_width = experts // ctx.ep, width // ctx.tp
        local_shared = shared // ctx.tp
        local_latent = latent // ctx.tp
        latent_params = 2 * h * latent if latent else 0
        global_params = count * (factor * io * (experts * width + shared) + h * experts + latent_params)
        local_expert_params = factor * io * local_experts * local_width
        local_shared_params = factor * io * local_shared
        local_router_params = h * experts
        local_latent_params = 2 * h * local_latent
        local_params = count * (
            local_expert_params + local_shared_params + local_router_params + local_latent_params
        )
        ba = effective_element_bytes(ctx.config, model.activation_dtype)
        bw = effective_element_bytes(ctx.config, model.weight_dtype)
        assignments = ctx.rows * topk
        active_experts = min(local_experts, max(1, assignments))

        router = gemm("layers.moe_router", ctx.rows, h, experts, ctx, model.logits_bytes)
        route_ops = 5 * ctx.rows * experts * count
        route = WorkItem(
            "layers.moe_topk_dispatch", "vector", route_ops,
            (ctx.rows * experts * model.logits_bytes + assignments * io * ba) * count,
            count, route_ops,
        )
        first_n = (2 if gated else 1) * local_width
        first_ops = 2 * assignments * io * first_n * count
        first_bytes = (
            assignments * io * ba + active_experts * io * first_n * bw
            + assignments * first_n * ba
        ) * count
        first = WorkItem(
            "layers.moe_expert_gate_up", "gemm", first_ops, first_bytes, count,
            first_ops, (max(1, assignments), first_n, io),
        )
        down_ops = 2 * assignments * local_width * io * count
        down_bytes = (
            assignments * local_width * ba + active_experts * local_width * io * bw
            + assignments * io * ba
        ) * count
        down = WorkItem(
            "layers.moe_expert_down", "gemm", down_ops, down_bytes, count,
            down_ops, (max(1, assignments), io, local_width),
        )
        combine_ops = 3 * assignments * io * count
        combine = WorkItem(
            "layers.moe_combine", "vector", combine_ops,
            2 * assignments * io * ba * count, count, combine_ops,
        )
        items = [router, route, first, down, combine]
        if latent:
            items.insert(1, gemm("layers.moe_latent_down", ctx.rows, h, local_latent, ctx))
            items.append(gemm("layers.moe_latent_up", ctx.rows, local_latent, h, ctx))
        if local_shared:
            shared_ctx = OperatorContext(ctx.config, ctx.phase, ctx.batch_size,
                                         ctx.token_length, ctx.attention_length, count)
            shared_first = gemm("layers.moe_shared_gate_up", ctx.rows, io,
                                (2 if gated else 1) * local_shared, shared_ctx)
            shared_down = gemm("layers.moe_shared_down", ctx.rows, local_shared, io, shared_ctx)
            items.extend((shared_first, shared_down))
        if spec.implementation == "latent_moe_approx":
            situ = float(spec.get("situ_ops_per_element", 6.0))
            elements = assignments * local_width * count
            items.append(WorkItem(
                "layers.moe_situ_latent", "vector", situ * elements,
                2 * elements * ba, count, situ * elements,
            ))

        comm = tp_all_reduce_hidden(ctx, "moe")
        if ctx.ep > 1:
            payload = ctx.rows * topk * io * ba
            comm.extend((
                CommunicationRequest("layers.moe_dispatch_all_to_all", "专家Dispatch", "all_to_all", "ep", payload, count),
                CommunicationRequest("layers.moe_combine_all_to_all", "专家Combine", "all_to_all", "ep", payload, count),
            ))
        temp = ceil(max(
            assignments * io * ba, assignments * first_n * ba, ctx.rows * local_latent * ba,
        ))
        assumptions = ["EP专家负载采用理想平均分配，不模拟热点专家。"]
        if latent:
            assumptions.append("Latent投影按TP列切分建模，latent域的All-Gather开销未单独建模。")
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, global_params, local_params,
            {
                "routed_expert_parameters": local_expert_params * count,
                "shared_expert_parameters": local_shared_params * count,
                "router_parameters": local_router_params * count,
                "latent_projection_parameters": local_latent_params * count,
            }, temporary_bytes=temp, work_items=items, communication_requests=comm,
            assumptions=assumptions, confidence="medium",
        )
