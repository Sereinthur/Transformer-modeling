"""Dense、Gated与MoE FFN算子。"""

from __future__ import annotations

from math import ceil

from ..core.work_item import WorkItem
from ..parallel.tensor import require_divisible, tp_all_reduce_hidden
from .base import (
    CommunicationRequest, OperatorContext, OperatorEstimate,
    TransformerOperator, effective_element_bytes, gemm, int_param,
    resolve_weight_dtype,
)

ROUTINGS = {"learned", "hash"}
# 路由打分的vector ops常数：打分函数越复杂，每个专家得分的元素级开销越高。
GATE_OPS = {"sigmoid": 4, "softmax": 5, "sqrtsoftplus": 7}


class DenseFFNOperator(TransformerOperator):
    type_id, chinese_name, slot = "dense_ffn", "Dense FFN", "ffn"
    implementations = {"default", "gelu"}

    def validate(self, spec, config):
        super().validate(spec, config)
        width = int(spec.get("intermediate_size", 0) or config.model.intermediate_size)
        if width <= 0:
            raise ValueError("dense_ffn intermediate_size must be positive")
        require_divisible("dense_ffn intermediate_size", width, config.parallelism.tensor_parallel)

    def estimate(self, spec, ctx):
        h, count = ctx.model.hidden_size, ctx.occurrence_count
        width = int(spec.get("intermediate_size", 0) or ctx.model.intermediate_size)
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
    type_id, chinese_name, slot = "gated_ffn", "Dense Gated FFN", "ffn"
    implementations = {"default", "swiglu", "fused_gate_up", "situ_glu"}

    def validate(self, spec, config):
        super().validate(spec, config)
        width = int(spec.get("intermediate_size", 0) or config.model.intermediate_size)
        if width <= 0 or width % config.parallelism.tensor_parallel:
            raise ValueError("gated_ffn intermediate_size must be positive and divisible by TP")
        if spec.implementation == "situ_glu":
            for field, default in (
                ("activation_situ_beta", 4.0),
                ("activation_situ_linear_beta", 25.0),
                ("situ_ops_per_element", 6.0),
            ):
                if float(spec.get(field, default)) <= 0:
                    raise ValueError(f"gated_ffn {field} must be positive")

    def estimate(self, spec, ctx):
        h, count = ctx.model.hidden_size, ctx.occurrence_count
        width = int(spec.get("intermediate_size", 0) or ctx.model.intermediate_size)
        local = width // ctx.tp
        params, local_params = 3 * h * width * count, 3 * h * local * count
        gate_up = gemm("layers.ffn_gate_up", ctx.rows, h, 2 * local, ctx)
        down = gemm("layers.ffn_down", ctx.rows, local, h, ctx)
        ba = effective_element_bytes(ctx.config, ctx.model.activation_dtype)
        ops_per_element = (
            float(spec.get("situ_ops_per_element", 6.0))
            if spec.implementation == "situ_glu" else 8.0
        )
        ops = int(ops_per_element * ctx.rows * local * count)
        activation = WorkItem(
            "layers.ffn_situ_glu" if spec.implementation == "situ_glu" else "layers.ffn_swiglu",
            "vector", ops,
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
        shared_count = int(spec.get("shared_expert_count", 1))
        if shared_count < 0:
            raise ValueError("moe shared_expert_count must be non-negative")
        shared *= shared_count
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
        if str(spec.get("routing", "learned")).lower() not in ROUTINGS:
            raise ValueError("moe routing must be learned or hash")
        if str(spec.get("gate_activation", "softmax")).lower() not in GATE_OPS:
            choices = ", ".join(sorted(GATE_OPS))
            raise ValueError(f"moe gate_activation must be one of: {choices}")
        if float(spec.get("swiglu_limit", 0.0)) < 0:
            raise ValueError("moe swiglu_limit cannot be negative")
        for field in ("weight_dtype", "routed_expert_weight_dtype", "shared_expert_weight_dtype"):
            resolve_weight_dtype(config, spec, field)

    def estimate(self, spec, ctx):
        model, h, count = ctx.model, ctx.model.hidden_size, ctx.occurrence_count
        experts = int_param(spec, "expert_count")
        topk = int_param(spec, "experts_per_token")
        width = int_param(spec, "expert_intermediate_size")
        shared = int(spec.get("shared_expert_intermediate_size", 0)) * int(
            spec.get("shared_expert_count", 1)
        )
        latent = int(spec.get("latent_size", 0))
        routing = str(spec.get("routing", "learned")).lower()
        gate_const = GATE_OPS[str(spec.get("gate_activation", "softmax")).lower()]
        wdtype = resolve_weight_dtype(ctx.config, spec)
        rdtype = resolve_weight_dtype(ctx.config, spec, "routed_expert_weight_dtype", wdtype)
        sdtype = resolve_weight_dtype(ctx.config, spec, "shared_expert_weight_dtype", wdtype)
        gated = str(spec.get("activation", "swiglu")).lower() in {
            "swiglu", "gated", "situ",
        }
        factor = 3 if gated else 2
        io = latent or h  # 专家I/O维：LatentMoE在latent域计算，否则为hidden。
        local_experts, local_width = experts // ctx.ep, width // ctx.tp
        local_shared = shared // ctx.tp
        local_latent = latent // ctx.tp
        latent_params = 2 * h * latent if latent else 0
        router_width = 0 if routing == "hash" else experts
        global_params = count * (
            factor * io * (experts * width + shared) + h * router_width + latent_params
        )
        local_expert_params = factor * io * local_experts * local_width
        local_shared_params = factor * io * local_shared
        local_router_params = h * router_width
        local_latent_params = 2 * h * local_latent
        local_params = count * (
            local_expert_params + local_shared_params + local_router_params + local_latent_params
        )
        ba = effective_element_bytes(ctx.config, model.activation_dtype)
        brw = effective_element_bytes(ctx.config, rdtype)
        global_assignments = ctx.rows * topk
        active_ep_ranks = min(ctx.ep, global_assignments)
        # Dense backbone inputs are replicated across EP ranks.  Only routed
        # expert work is divided after the idealized all-to-all dispatch.
        assignments = ceil(global_assignments / active_ep_ranks)
        global_active_experts = min(experts, max(1, global_assignments))
        active_experts = min(
            local_experts, max(1, ceil(global_active_experts / active_ep_ranks))
        )
        # tile padding发生在每个专家内部的grouped GEMM上，
        # 用平均每专家行数作为效率口径，而不是所有专家的总行数。
        per_expert_rows = max(1, ceil(assignments / active_experts))

        if routing == "hash":
            # Hash路由无可学习打分：只做token到专家的确定映射与分派。
            route_ops = 3 * ctx.rows * topk * count
            route_bytes = global_assignments * io * ba * count
        else:
            route_ops = gate_const * ctx.rows * experts * count
            route_bytes = (
                ctx.rows * experts * model.logits_bytes + global_assignments * io * ba
            ) * count
        route = WorkItem(
            "layers.moe_topk_dispatch", "vector", route_ops, route_bytes, count, route_ops,
        )
        first_n = (2 if gated else 1) * local_width
        first_ops = 2 * assignments * io * first_n * count
        first_bytes = (
            assignments * io * ba + active_experts * io * first_n * brw
            + assignments * first_n * ba
        ) * count
        first = WorkItem(
            "layers.moe_expert_gate_up", "gemm", first_ops, first_bytes, count,
            first_ops, (per_expert_rows, first_n, io), compute_dtype=rdtype,
        )
        down_ops = 2 * assignments * local_width * io * count
        down_bytes = (
            assignments * local_width * ba + active_experts * local_width * io * brw
            + assignments * io * ba
        ) * count
        down = WorkItem(
            "layers.moe_expert_down", "gemm", down_ops, down_bytes, count,
            down_ops, (per_expert_rows, io, local_width), compute_dtype=rdtype,
        )
        combine_ops = 3 * assignments * io * count
        combine = WorkItem(
            "layers.moe_combine", "vector", combine_ops,
            2 * assignments * io * ba * count, count, combine_ops,
        )
        items = [route, first, down, combine]
        if gated:
            limit = float(spec.get("swiglu_limit", 0.0))
            activation_ops = (10 if limit > 0 else 8) * assignments * local_width * count
            items.insert(2, WorkItem(
                "layers.moe_clipped_swiglu" if limit > 0 else "layers.moe_swiglu",
                "vector", activation_ops,
                2 * assignments * local_width * ba * count,
                count, activation_ops,
            ))
        if routing != "hash":
            items.insert(0, gemm(
                "layers.moe_router", ctx.rows, h, experts, ctx, model.logits_bytes,
                weight_dtype=wdtype,
            ))
        if latent:
            items.insert(1, gemm("layers.moe_latent_down", ctx.rows, h, local_latent, ctx,
                                 weight_dtype=wdtype))
            items.append(gemm("layers.moe_latent_up", ctx.rows, local_latent, h, ctx,
                              weight_dtype=wdtype))
        if local_shared:
            shared_ctx = OperatorContext(ctx.config, ctx.phase, ctx.batch_size,
                                         ctx.token_length, ctx.attention_length, count)
            shared_first = gemm("layers.moe_shared_gate_up", ctx.rows, io,
                                (2 if gated else 1) * local_shared, shared_ctx,
                                weight_dtype=sdtype)
            shared_down = gemm("layers.moe_shared_down", ctx.rows, local_shared, io,
                               shared_ctx, weight_dtype=sdtype)
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
            payload = global_assignments * io * ba
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
        if routing == "hash":
            assumptions.append("Hash路由不含可学习打分权重，分派按确定映射计成本。")
        if spec.get("routed_scaling_factor") is not None:
            assumptions.append("routed_scaling_factor只缩放权重数值，不产生额外计算或访存。")
        limit = float(spec.get("swiglu_limit", 0.0))
        if limit > 0:
            assumptions.append(f"SwiGLU的gate/up按官方上限{limit:g}执行clamp。")
        hash_state = (
            ctx.model.vocab_size * topk * 4 * count if routing == "hash" else 0
        )
        buckets = {}
        for dtype, value in (
            (rdtype, local_expert_params * count),
            (sdtype, local_shared_params * count),
            (wdtype, (local_router_params + local_latent_params) * count),
        ):
            if value:
                buckets[dtype] = buckets.get(dtype, 0) + value
        return OperatorEstimate(
            self.type_id, self.chinese_name, count, global_params, local_params,
            {
                "routed_expert_parameters": local_expert_params * count,
                "shared_expert_parameters": local_shared_params * count,
                "router_parameters": local_router_params * count,
                "latent_projection_parameters": local_latent_params * count,
            }, persistent_state_bytes=hash_state,
            state_breakdown=(
                {"hash_route_table_bytes": hash_state} if hash_state else {}
            ), temporary_bytes=temp, work_items=items, communication_requests=comm,
            assumptions=assumptions, confidence="medium",
            local_parameters_by_dtype=buckets,
        )
