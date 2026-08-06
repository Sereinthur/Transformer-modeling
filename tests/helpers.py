import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def example(name="single_chip_gqa"):
    return json.loads((ROOT / "examples" / f"{name}.json").read_text(encoding="utf-8"))


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
