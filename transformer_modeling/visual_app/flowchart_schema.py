"""Schema v3 config <-> ordered-operator flowchart conversion.

The visual editor persists only a vertical sequence of independently editable
operators.  Group boxes, layer coverage and edges are derived view data; old
fixed slots, hidden-state templates and user-authored graph edges are not part
of the configuration contract.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .operator_schemas import OPERATOR_SCHEMAS, chinese_name


OUTPUT_LAYOUT = (
    ("output_norm", "output_norm", "norm", "norm", "rms_norm", "输出 Norm"),
    ("output_head", "output_head", "output", "head", "lm_head", "LM Head"),
    ("output_sampling", "output_sampling", "output", "sampling", "sampling", "Sampling"),
)
MODEL_PASSTHROUGH = ("dtype", "quantization", "inference", "extra", "metadata")


def _split_operator(data: Any, default_type: str) -> tuple[str, str, dict[str, Any]]:
    if data is None:
        source: dict[str, Any] = {}
    elif isinstance(data, dict):
        source = dict(data)
    else:
        raise ValueError(f"operator {default_type} must be an object")
    type_id = str(source.pop("type", default_type)).lower()
    implementation = str(source.pop("implementation", "default")).lower()
    nested = source.pop("params", {})
    if not isinstance(nested, dict):
        raise ValueError(f"operator {type_id}.params must be an object")
    source.pop("_说明", None)
    return type_id, implementation, {**nested, **source}


def _operator_slot(type_id: str) -> str:
    schema = OPERATOR_SCHEMAS.get(type_id)
    return str(schema.get("slot", "any")) if schema else "any"


def _operator_node(
    node_id: str,
    data: Any,
    default_type: str,
    role: str,
    label: str | None = None,
) -> dict[str, Any]:
    type_id, implementation, params = _split_operator(data, default_type)
    return {
        "id": node_id,
        "type": type_id,
        "slot": _operator_slot(type_id),
        "role": role,
        "label": label or chinese_name(type_id),
        "implementation": implementation,
        "params": params,
    }


def _group_node(
    group_id: str, segment: dict[str, Any], label: str, kind: str
) -> dict[str, Any]:
    if not isinstance(segment, dict):
        raise ValueError("layer segment must be an object")
    expired = {"norm", "attention", "residual", "ffn", "residual_connections"} & set(segment)
    if expired:
        raise ValueError(
            "configuration version expired: fixed layer fields are unsupported; use operations"
        )
    repeat = int(segment.get("repeat", 1))
    if repeat <= 0:
        raise ValueError("layer segment repeat must be greater than zero")
    raw_operations = segment.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise ValueError("layer segment operations must be a non-empty array")

    children: list[dict[str, Any]] = []
    operation_ids: set[str] = set()
    for index, operation in enumerate(raw_operations):
        if not isinstance(operation, dict):
            raise ValueError("layer operation must be an object")
        operation_id = str(operation.get("id", "")).strip()
        if not operation_id:
            raise ValueError("layer operation.id is required")
        if operation_id in operation_ids:
            raise ValueError("layer operation ids must be unique")
        operation_ids.add(operation_id)
        operator = operation.get("operator")
        if not isinstance(operator, dict):
            raise ValueError(f"layer operation {operation_id}.operator must be an object")
        child = _operator_node(
            f"{group_id}_{operation_id}", operator, "unmodeled", operation_id
        )
        child["operation_id"] = operation_id
        child["order"] = index
        children.append(child)

    return {
        "id": group_id,
        "type": "layer_group",
        "slot": "layer",
        "label": label,
        "repeat": repeat,
        "group_kind": kind,
        "children": children,
    }


def _model_info(model: dict[str, Any], nodes: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = model.get("dimensions", {})
    if not isinstance(dimensions, dict):
        raise ValueError("model.dimensions must be an object")
    groups = [node for node in nodes if node.get("type") == "layer_group"]
    prefix_layers = sum(int(node["repeat"]) for node in groups if node["group_kind"] == "prefix")
    cycle_layers = sum(int(node["repeat"]) for node in groups if node["group_kind"] == "pattern")
    suffix_layers = sum(int(node["repeat"]) for node in groups if node["group_kind"] == "suffix")
    layer_count = int(dimensions.get("layer_count", 0))
    remaining = max(0, layer_count - prefix_layers - suffix_layers)
    full_cycles, remainder = divmod(remaining, cycle_layers) if cycle_layers else (0, 0)
    vocab = int(dimensions.get("vocab_size", 0))
    info: dict[str, Any] = {
        "model_id": str(model.get("id", model.get("model_id", "custom"))),
        "name": str(model.get("name", "Custom Transformer")),
        "layer_count": layer_count,
        "hidden_size": int(dimensions.get("hidden_size", 0)),
        "intermediate_size": int(dimensions.get("intermediate_size", 0)),
        "vocab_size": vocab,
        "padded_vocab_size": int(dimensions.get("padded_vocab_size", vocab)),
        "prefix_layer_count": prefix_layers,
        "pattern_cycle_length": cycle_layers,
        "suffix_layer_count": suffix_layers,
        "structure": {
            "prefix_layer_count": prefix_layers,
            "pattern_cycle_length": cycle_layers,
            "full_cycle_count": full_cycles,
            "partial_cycle_layers": remainder,
            "suffix_layer_count": suffix_layers,
        },
    }
    for key in MODEL_PASSTHROUGH:
        value = model.get(key)
        if isinstance(value, dict):
            info[key] = deepcopy(value)
    return info


def _annotate_stage_coverage(nodes: list[dict[str, Any]], info: dict[str, Any]) -> None:
    groups = [node for node in nodes if node.get("type") == "layer_group"]
    prefixes = [node for node in groups if node["group_kind"] == "prefix"]
    patterns = [node for node in groups if node["group_kind"] == "pattern"]
    suffixes = [node for node in groups if node["group_kind"] == "suffix"]
    layer_count = int(info["layer_count"])
    prefix_count = sum(int(node["repeat"]) for node in prefixes)
    cycle_length = sum(int(node["repeat"]) for node in patterns)
    suffix_count = sum(int(node["repeat"]) for node in suffixes)
    remaining = max(0, layer_count - prefix_count - suffix_count)
    full_cycles, remainder = divmod(remaining, cycle_length) if cycle_length else (0, 0)

    cursor = 0
    for node in prefixes:
        repeat = int(node["repeat"])
        node["_structure"] = {
            "kind": "prefix", "occurrences": repeat,
            "first_layer": cursor + 1, "last_layer": cursor + repeat,
            "repeat_per_cycle": None,
        }
        cursor += repeat

    offset = 0
    for node in patterns:
        repeat = int(node["repeat"])
        partial = max(0, min(repeat, remainder - offset))
        occurrences = full_cycles * repeat + partial
        last_layer = 0
        if occurrences:
            last_layer = (
                prefix_count + full_cycles * cycle_length + offset + partial
                if partial
                else prefix_count + (full_cycles - 1) * cycle_length + offset + repeat
            )
        node["_structure"] = {
            "kind": "pattern", "occurrences": occurrences,
            "first_layer": prefix_count + offset + 1, "last_layer": last_layer,
            "repeat_per_cycle": repeat,
            "cycle_position_start": offset + 1,
            "cycle_position_end": offset + repeat,
            "full_cycle_count": full_cycles,
            "partial_occurrences": partial,
        }
        offset += repeat

    cursor = layer_count - suffix_count
    for node in suffixes:
        repeat = int(node["repeat"])
        node["_structure"] = {
            "kind": "suffix", "occurrences": repeat,
            "first_layer": cursor + 1, "last_layer": cursor + repeat,
            "repeat_per_cycle": None,
        }
        cursor += repeat


def config_to_flowchart(config_dict: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config_dict, dict):
        raise ValueError("config must be an object")
    if int(config_dict.get("schema_version", 0)) != 3:
        raise ValueError("configuration version expired: visual modeling accepts schema_version 3 only")
    model = config_dict.get("model")
    if not isinstance(model, dict):
        raise ValueError("config.model must be an object")
    patterns = model.get("layer_pattern")
    prefixes = model.get("layer_prefix", [])
    suffixes = model.get("layer_suffix", [])
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("model.layer_pattern must be a non-empty array")
    if not isinstance(prefixes, list) or not isinstance(suffixes, list):
        raise ValueError("model.layer_prefix and model.layer_suffix must be arrays")

    nodes: list[dict[str, Any]] = [
        _operator_node("embedding_0", model.get("embedding"), "token_embedding", "embedding")
    ]
    nodes.extend(
        _group_node(f"prefix_{index}", segment, f"前置层 {index + 1}", "prefix")
        for index, segment in enumerate(prefixes)
    )
    nodes.extend(
        _group_node(f"group_{index}", segment, f"循环 Pattern {index + 1}", "pattern")
        for index, segment in enumerate(patterns)
    )
    nodes.extend(
        _group_node(f"suffix_{index}", segment, f"尾部层 {index + 1}", "suffix")
        for index, segment in enumerate(suffixes)
    )

    output = model.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("model.output must be an object")
    for node_id, role, _slot, key, default_type, label in OUTPUT_LAYOUT:
        nodes.append(_operator_node(node_id, output.get(key), default_type, role, label))

    info = _model_info(model, nodes)
    _annotate_stage_coverage(nodes, info)
    return {
        "schema_version": 3,
        "nodes": nodes,
        "edges": [
            {"from": nodes[index]["id"], "to": nodes[index + 1]["id"]}
            for index in range(len(nodes) - 1)
        ],
        "model_info": info,
    }


def _operator_config(node: dict[str, Any]) -> dict[str, Any]:
    type_id = node.get("type")
    if not isinstance(type_id, str) or not type_id:
        raise ValueError("flowchart operator node requires type")
    params = node.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"node {node.get('id', type_id)} params must be an object")
    result: dict[str, Any] = {"type": type_id.lower()}
    implementation = str(node.get("implementation", "default") or "default").lower()
    if implementation != "default":
        result["implementation"] = implementation
    result.update({key: value for key, value in params.items() if value is not None and value != ""})
    return result


def _pattern_config(group: dict[str, Any]) -> dict[str, Any]:
    children = group.get("children")
    if not isinstance(children, list) or not children:
        raise ValueError(f"layer group {group.get('id', '?')} requires operator children")
    repeat = int(group.get("repeat", 1))
    if repeat <= 0:
        raise ValueError(f"layer group {group.get('id', '?')} repeat must be greater than zero")
    operations: list[dict[str, Any]] = []
    ids: set[str] = set()
    for child in children:
        if not isinstance(child, dict):
            raise ValueError("layer group children must be objects")
        operation_id = str(child.get("operation_id") or child.get("role") or "").strip()
        if not operation_id or operation_id in ids:
            raise ValueError("layer operation ids must be present and unique")
        ids.add(operation_id)
        operations.append({"id": operation_id, "operator": _operator_config(child)})
    return {"repeat": repeat, "operations": operations}


def _apply_model_info(model: dict[str, Any], info: dict[str, Any]) -> None:
    dimensions = dict(model.get("dimensions", {}))
    for key in ("layer_count", "hidden_size", "intermediate_size", "vocab_size", "padded_vocab_size"):
        if info.get(key) is not None:
            dimensions[key] = int(info[key])
    model["dimensions"] = dimensions
    if info.get("name"):
        model["name"] = str(info["name"])
    if info.get("model_id"):
        model["id"] = str(info["model_id"])
    for key in MODEL_PASSTHROUGH:
        value = info.get(key)
        if isinstance(value, dict):
            model[key] = deepcopy(value)


def _validate_linear_flowchart(nodes: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    groups: list[dict[str, Any]] = []
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id or node_id in ids:
            raise ValueError("flowchart node ids must be present and unique")
        ids.add(node_id)
        if node.get("type") == "layer_group":
            groups.append(node)
    if not nodes or nodes[0].get("role") != "embedding":
        raise ValueError("linear flowchart must begin with Embedding")
    if [str(node.get("role") or "") for node in nodes[-3:]] != [
        "output_norm", "output_head", "output_sampling"
    ]:
        raise ValueError("linear flowchart must end with output Norm, LM Head and Sampling")
    if not groups:
        raise ValueError("flowchart requires at least one layer_pattern group")

    rank = {"prefix": 0, "pattern": 1, "suffix": 2}
    previous = 0
    pattern_count = 0
    for group in groups:
        if "residual_connections" in group or "sequence_mode" in group:
            raise ValueError("legacy flowchart fields are unsupported; use operator children")
        kind = str(group.get("group_kind", "pattern"))
        if kind not in rank or rank[kind] < previous:
            raise ValueError("layer groups must be ordered prefix, pattern, suffix")
        previous = rank[kind]
        pattern_count += int(kind == "pattern")
        _pattern_config(group)
    if not pattern_count:
        raise ValueError("flowchart requires at least one layer_pattern group")


def flowchart_to_config(
    flowchart_json: dict[str, Any], base_config_dict: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(flowchart_json, dict) or not isinstance(base_config_dict, dict):
        raise ValueError("flowchart and base_config must be objects")
    if int(flowchart_json.get("schema_version", 0)) != 3:
        raise ValueError("configuration version expired: flowchart schema_version 3 is required")
    nodes = flowchart_json.get("nodes")
    if not isinstance(nodes, list) or not nodes or not all(isinstance(node, dict) for node in nodes):
        raise ValueError("flowchart.nodes must be a non-empty object array")
    _validate_linear_flowchart(nodes)

    config = deepcopy(base_config_dict)
    config["schema_version"] = 3
    model = deepcopy(config.get("model")) if isinstance(config.get("model"), dict) else {}
    prefixes: list[dict[str, Any]] = []
    patterns: list[dict[str, Any]] = []
    suffixes: list[dict[str, Any]] = []
    output: dict[str, Any] = {}
    embedding: dict[str, Any] | None = None
    output_roles = {role: key for _, role, _, key, _, _ in OUTPUT_LAYOUT}

    for node in nodes:
        if node.get("type") == "layer_group":
            segment = _pattern_config(node)
            kind = str(node.get("group_kind", "pattern"))
            (prefixes if kind == "prefix" else suffixes if kind == "suffix" else patterns).append(segment)
        elif node.get("role") == "embedding":
            embedding = _operator_config(node)
        elif node.get("role") in output_roles:
            output[output_roles[str(node["role"])]] = _operator_config(node)
        else:
            raise ValueError(f"unrecognized top-level node {node.get('id', node.get('type'))}")

    model["embedding"] = embedding or {"type": "token_embedding"}
    model["layer_pattern"] = patterns
    if prefixes:
        model["layer_prefix"] = prefixes
    else:
        model.pop("layer_prefix", None)
    if suffixes:
        model["layer_suffix"] = suffixes
    else:
        model.pop("layer_suffix", None)
    model["output"] = output
    info = flowchart_json.get("model_info")
    if isinstance(info, dict):
        _apply_model_info(model, info)
    config["model"] = model
    return config
