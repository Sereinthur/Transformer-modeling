"""Tensor Parallel 通信声明与切分校验。

集中处理粗粒度解析模型中 TP 相关的通信 payload 计算与整除性校验，
避免各算子重复编写几乎相同的 All-Reduce / All-Gather 构造代码。

切分后的 GEMM 维度（影响 ops/bytes 账单）仍由各算子自行计算，
因为切哪个维度取决于算子自身的几何结构。
"""

from __future__ import annotations

from ...operators.base import (
    CommunicationRequest, OperatorContext, effective_element_bytes,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          通信声明构造函数                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tp_all_reduce_hidden(
    ctx: OperatorContext,
    prefix: str,
    occurrences: int | None = None,
) -> list[CommunicationRequest]:
    """Attention/FFN 输出后对 hidden 维度做 All-Reduce 的通信声明。

    payload = rows × hidden × activation_bytes
    TP=1 时返回空列表（无需集合通信）。
    """

    if ctx.tp == 1:
        return []
    payload = ctx.rows * ctx.model.hidden_size * effective_element_bytes(
        ctx.config, ctx.model.activation_dtype
    )
    return [CommunicationRequest(
        f"layers.{prefix}_all_reduce", f"{prefix}输出归约", "all_reduce", "tp",
        payload, occurrences if occurrences is not None else ctx.occurrence_count,
    )]


def tp_embedding_all_reduce(ctx: OperatorContext) -> list[CommunicationRequest]:
    """词表并行 Embedding 输出后的 hidden All-Reduce。

    各 rank 只对本地词表命中的 token 产生非零向量，随后归约得到
    每个 rank 都可继续使用的完整 hidden state。
    """

    if ctx.tp == 1:
        return []
    payload = ctx.rows * ctx.model.hidden_size * effective_element_bytes(
        ctx.config, ctx.model.activation_dtype
    )
    return [CommunicationRequest(
        "embedding_hidden_all_reduce", "词表并行Embedding归约",
        "all_reduce", "tp", payload, 1,
    )]


def tp_all_gather_logits(
    ctx: OperatorContext,
    local_vocab: int,
    rows: int,
) -> list[CommunicationRequest]:
    """LM Head 后对 logits 做 All-Gather 的通信声明。

    payload = rows × local_vocab × logits_bytes
    与 Attention/FFN 的 All-Reduce 不同，这里收集的是词表维度的 logits。
    """

    if ctx.tp == 1:
        return []
    return [CommunicationRequest(
        "lm_head_logits_all_gather", "词表Logits收集", "all_gather", "tp",
        rows * local_vocab * ctx.model.logits_bytes, 1,
    )]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            切分整除性校验                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def require_divisible(name: str, value: int, tp: int) -> None:
    """要求 value 能被 TP 整除，否则抛出带算子名的 ValueError。"""

    if value % tp:
        raise ValueError(f"{name} must be divisible by TP")
