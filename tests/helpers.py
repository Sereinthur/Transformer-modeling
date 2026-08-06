import copy
import json
from pathlib import Path

from transformer_modeling.models import get_preset


ROOT = Path(__file__).resolve().parents[1]


def example(name="single_chip_gqa"):
    data = json.loads((ROOT / "examples" / f"{name}.json").read_text(encoding="utf-8"))
    # Scenario files from schema v2 are intentionally not accepted by Config.
    # Tests reuse their hardware/request envelope but source the model from v3 presets.
    preset = {
        "moe_qwen3_30b_a3b": "qwen3-30b-a3b",
        "kimi_k3_base_tp64": "kimi-k3-official",
        "kimi_k3_capacity_proxy_b200_nvl72": "kimi-k3-official",
        "kimi_k3_performance_proxy_b200_nvl72_8k": "kimi-k3-official",
        "deepseek_v4_pro_b200_nvl72": "deepseek-v4-pro",
    }.get(name)
    if preset:
        data["schema_version"] = 3
        data["model"] = get_preset(preset)["model"]
    return data


def parallel(data, *, tp=1, ep=1, pp=1, microbatches=1):
    result = copy.deepcopy(data)
    result["parallelism"].update(
        tensor_parallel=tp, expert_parallel=ep, pipeline_parallel=pp,
        pipeline_microbatches=microbatches,
    )
    result["hardware"]["device_count"] = tp * ep * pp
    return result


def phase_operators(phase):
    return [operator for stage in phase.get("stages", []) for operator in stage.get("operators", [])]
