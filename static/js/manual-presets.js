/** Schema v2模型预设应用、状态追踪和跨页兼容提示。 */

import { $ } from "./dom.js";
import { clonePresetValue, loadModelPresets, presetDescription, replacePresetOptions } from "./preset-catalog.js";

const snapshot = (config) => JSON.stringify({
  model: config.model,
  max_sequence_length: Number(config.serving.max_sequence_length),
});

function patternOperators(model, slot) {
  return (model.layer_pattern || []).map((item) => item[slot]).filter(Boolean);
}

export function compatibilityMessages(config) {
  const messages = [
    { severity: "advisory", text: `当前按 ${config.model.dtype.weight.toUpperCase()} 权重格式解释芯片吞吐，请确认硬件口径。` },
    { severity: "advisory", text: "GEMM、Attention与向量效率会复用于所选算子，请按实际实现校准。" },
  ];
  const d = config.model.dimensions;
  const tp = Number(config.parallelism.tensor_parallel);
  const ep = Number(config.parallelism.expert_parallel || 1);
  const pp = Number(config.parallelism.pipeline_parallel);
  const required = Number(config.serving.prompt_length.value ?? config.serving.prompt_length)
    + Number(config.serving.output_length.value ?? config.serving.output_length) - 1;
  if (required > Number(config.serving.max_sequence_length)) messages.push({ severity: "critical", text: `请求需要${required} token，超过最大上下文。` });
  if (pp > Number(d.layer_count)) messages.push({ severity: "critical", text: `PP=${pp}超过层数${d.layer_count}。` });
  if ((config.model.layer_prefix || []).length) messages.push({ severity: "advisory", text: "该预设含前置层（layer_prefix），编辑器只展示循环部分，前置层按预设原样参与计算。" });
  patternOperators(config.model, "attention").forEach((operator) => {
    const heads = Number(operator.query_heads || operator.heads || 0);
    if (heads && heads % tp) messages.push({ severity: "critical", text: `${operator.type}的Heads=${heads}不能被TP=${tp}整除。` });
    const kv = Number(operator.kv_heads || 0);
    if (kv && kv % tp && tp % kv) messages.push({ severity: "critical", text: `${operator.type}的KV Heads=${kv}无法在TP=${tp}下分片或等组复制。` });
    if (Number(operator.selected_entries) && !Number(operator.compress_ratio)) messages.push({ severity: "critical", text: `${operator.type}的压缩率m必须大于0。` });
  });
  patternOperators(config.model, "ffn").forEach((operator) => {
    const width = Number(operator.expert_intermediate_size || operator.intermediate_size || d.intermediate_size);
    if (width % tp) messages.push({ severity: "critical", text: `${operator.type}中间维度${width}不能被TP=${tp}整除。` });
    if (operator.type === "moe" && Number(operator.expert_count) % ep) messages.push({ severity: "critical", text: `专家数${operator.expert_count}不能被EP=${ep}整除。` });
  });
  return messages;
}

export async function initializeManualPresets({ getConfig, applyConfig }) {
  const presets = await loadModelPresets();
  const select = $("#manual-model-preset");
  const applyButton = $("#apply-manual-preset");
  const description = $("#manual-preset-description");
  const source = $("#manual-preset-source");
  const status = $("#manual-preset-status");
  const guidance = $("#manual-preset-guidance");
  const list = $("#manual-preset-guidance-list");
  let activePreset = null;
  let activeSnapshot = null;
  replacePresetOptions(select, presets, "当前参数 / 自定义");

  const selected = () => presets.find((item) => item.id === select.value) || null;
  const scenario = () => $("#k3-draft-scenario").value;
  const pending = () => {
    const item = selected();
    if (!item) return false;
    if (item.id !== activePreset?.id) return true;
    return item.id === "kimi-k3-draft" && scenario() !== getConfig().model.metadata?.scenario;
  };

  function renderSelection() {
    const item = selected();
    $("#k3-draft-scenario-control").hidden = item?.id !== "kimi-k3-draft";
    applyButton.disabled = !item;
    description.textContent = item ? presetDescription(item) : "保留当前手动算子组合。";
    source.hidden = !item?.source_url;
    if (item?.source_url) source.href = item.source_url;
  }

  function renderState() {
    list.replaceChildren();
    if (pending()) {
      status.textContent = `已选择 ${selected().name}，尚未应用`;
      const li = document.createElement("li"); li.className = "critical";
      li.textContent = "请点击“应用预设”后再计算。"; list.appendChild(li);
      guidance.hidden = false; return;
    }
    if (!activePreset) { status.textContent = "当前参数 / 自定义"; guidance.hidden = true; return; }
    status.textContent = snapshot(getConfig()) === activeSnapshot
      ? `已应用 ${activePreset.name}` : `基于 ${activePreset.name}，已调整`;
    compatibilityMessages(getConfig()).forEach((message) => {
      const li = document.createElement("li"); li.className = message.severity;
      li.textContent = message.text; list.appendChild(li);
    });
    guidance.hidden = false;
  }

  async function applySelected() {
    const item = selected(); if (!item) return;
    const response = await fetch("/api/model-definitions/resolve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset_id: item.id, scenario: scenario() }),
    });
    const resolved = await response.json();
    if (!response.ok) throw new Error(resolved.error || "模型定义解析失败");
    const config = getConfig();
    config.model = clonePresetValue(resolved.resolved_model);
    config.serving.max_sequence_length = resolved.default_max_sequence_length;
    applyConfig(config);
    activePreset = item; activeSnapshot = snapshot(getConfig());
    renderSelection(); renderState();
  }

  function syncFromConfig(config) {
    activePreset = presets.find((item) => item.id === config.model.id) || null;
    select.value = activePreset?.id || "custom";
    if (activePreset?.id === "kimi-k3-draft") $("#k3-draft-scenario").value = config.model.metadata?.scenario || "base";
    activeSnapshot = activePreset ? snapshot(getConfig()) : null;
    renderSelection(); renderState();
  }

  select.addEventListener("change", () => { renderSelection(); renderState(); });
  $("#k3-draft-scenario").addEventListener("change", renderState);
  applyButton.addEventListener("click", () => applySelected().catch((error) => { status.textContent = `无法应用预设：${error.message}`; }));
  $("#config-form").addEventListener("input", renderState);
  $("#config-form").addEventListener("change", renderState);
  renderSelection(); renderState();
  return { syncFromConfig, validationError: () => pending() ? `已选择“${selected().name}”但尚未应用。` : null };
}
