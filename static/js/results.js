/** Schema v2统一容量、性能、通信、流水和算子结果展示。 */

import { $ } from "./dom.js";
import { renderBarChart } from "./charts.js";
import { formatBytes, formatDuration, formatParameters, formatRate } from "./format.js";

let latestResult = null;
const BOTTLENECK_LABELS = { compute: "计算", memory: "HBM访存", launch: "启动", communication: "通信", local: "本地算子" };

function criticalCapacity(capacity) {
  return capacity.per_stage[capacity.critical_stage_index];
}

export function renderCapacity(capacity) {
  const critical = criticalCapacity(capacity);
  const total = capacity.available_bytes_per_rank;
  const parts = [
    ["weights", "权重", critical.weights_bytes, "var(--cyan)"],
    ["state", "KV/持久状态", critical.persistent_state_bytes, "var(--blue)"],
    ["activation", "峰值激活", critical.activations_peak_bytes, "var(--purple)"],
    ["communication", "通信缓冲", critical.communication_buffer_peak_bytes, "var(--red)"],
    ["unavailable", "校准时不可用", critical.baseline_unavailable_bytes, "var(--dim)"],
    ["reserved", "运行时预留", critical.runtime_reserved_bytes, "var(--orange)"],
  ].filter((item) => item[2] > 0);
  $("#capacity-total").textContent = `${formatBytes(critical.peak_total_bytes)} / ${formatBytes(total)}`;
  $("#capacity-track").innerHTML = parts.map(([key, label, value]) =>
    `<span class="capacity-segment ${key}" style="width:${Math.max(0.2, value / total * 100)}%" title="${label}: ${formatBytes(value)}"></span>`
  ).join("");
  $("#capacity-legend").innerHTML = parts.map(([, label, value, color]) =>
    `<div class="legend-item"><span><i style="background:${color}"></i>${label}</span><strong>${formatBytes(value)}</strong></div>`
  ).join("");
}

function flattenOperators(phase) {
  return (phase?.stages || []).flatMap((stage) => stage.operators || []);
}

export function selectedOperators(result) {
  if (!result?.performance) return [];
  const phase = $("#operator-phase").value;
  if (phase === "prefill") return flattenOperators(result.performance.prefill);
  const step = phase === "first_decode"
    ? result.performance.decode.first_step : result.performance.decode.last_step;
  return flattenOperators(step);
}

export function renderOperatorTable() {
  const rows = selectedOperators(latestResult);
  const body = $("#operator-table");
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--dim)">当前阶段没有算子数据</td></tr>';
    return;
  }
  body.innerHTML = rows.map((operator) => {
    const compute = operator.suboperators.reduce((sum, item) => sum + item.time_seconds.compute, 0);
    const memory = operator.suboperators.reduce((sum, item) => sum + item.time_seconds.memory, 0);
    const bottleneck = operator.time_seconds.communication > operator.time_seconds.local
      ? "communication" : (compute >= memory ? "compute" : "memory");
    return `<tr>
      <td>${operator["中文名称"]} × ${operator.occurrences}<br><small>${operator.type} · ${operator.confidence}</small></td>
      <td><span class="bottleneck-badge ${bottleneck}">${BOTTLENECK_LABELS[bottleneck]}</span></td>
      <td>${formatDuration(compute)}</td><td>${formatDuration(memory)}</td>
      <td>${formatDuration(operator.time_seconds.estimated)}</td></tr>`;
  }).join("");
}

function allCommunications(phase) {
  return flattenOperators(phase).flatMap((operator) => operator.communication || []);
}

export function renderCommunication(prefill, decode, parallelism) {
  const block = $("#communication-block");
  block.hidden = parallelism.device_count <= 1;
  if (block.hidden) return;
  const firstDecode = decode.first_step;
  const groups = [
    ["Prefill", allCommunications(prefill)],
    ["首次Decode", allCommunications(firstDecode)],
  ];
  const rows = groups.map(([label, items]) => ({
    label, value: items.reduce((sum, item) => sum + item.time_seconds.estimated, 0),
    display: (items.reduce((sum, item) => sum + item.time_seconds.estimated, 0) * 1e3).toFixed(2),
  }));
  const a2a = groups.flatMap(([, items]) => items).filter((item) => item.collective.type === "all_to_all").length;
  $("#communication-summary").textContent = `PP${parallelism.pipeline_parallel} × EP${parallelism.expert_parallel} × TP${parallelism.tensor_parallel} · ${parallelism.topology} · All-to-All明细 ${a2a} 项`;
  renderBarChart($("#communication-chart"), rows);
}

export function renderPipeline(prefill, pp) {
  const block = $("#pipeline-block");
  block.hidden = pp <= 1;
  if (block.hidden) return;
  const schedule = prefill.pipeline_schedule;
  $("#pipeline-summary").textContent = `${prefill.stages.length}个Stage · 利用率 ${(schedule.average_stage_utilization * 100).toFixed(1)}% · 气泡 ${(schedule.bubble_fraction * 100).toFixed(1)}%`;
  renderBarChart($("#pipeline-chart"), schedule.stage_service_seconds.map((seconds, index) => ({
    label: `Stage ${index} · ${prefill.stages[index]?.operators?.reduce((sum, item) => sum + item.occurrences, 0) || 0}个算子实例`,
    value: seconds, display: (seconds * 1e3).toFixed(2),
  })));
}

function renderHybrid(result) {
  const block = $("#hybrid-result-block");
  const mix = result.model.operator_mix || {};
  const hybrid = Boolean(mix.kda || mix.gated_mla || mix.attnres);
  block.hidden = !hybrid;
  if (!hybrid) return;
  $("#hybrid-result-summary").textContent = `${mix.kda || 0}层 KDA · ${mix.gated_mla || 0}层 Gated MLA · ${mix.attnres || 0}次 AttnRes`;
  const critical = criticalCapacity(result.capacity);
  const states = critical.states_by_operator || {};
  const rows = [
    ["每rank权重", formatBytes(critical.weights_bytes)],
    ["KDA State", formatBytes(states.kda_state_bytes || 0)],
    ["MLA Latent KV", formatBytes(states.mla_latent_kv_cache_bytes || 0)],
    ["AttnRes State", formatBytes(states.attnres_state_bytes || 0)],
    ["未建模参数", formatParameters(critical.weights_by_operator.unmodeled_parameters || 0)],
    ["性能完整性", result.validity.performance_complete ? "完整" : "已建模代理结果"],
  ];
  $("#hybrid-result-grid").innerHTML = rows.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  $("#hybrid-result-warnings").innerHTML = (result.warnings || []).map((item) => `<li class="critical">${item}</li>`).join("");
}

export function renderResult(result) {
  latestResult = result;
  $("#empty-state").hidden = true; $("#loading-state").hidden = true;
  $("#error-state").hidden = true; $("#result-content").hidden = false;
  $("#download-result").hidden = false;
  const capacity = result.capacity;
  const critical = criticalCapacity(capacity);
  const banner = $("#fit-banner");
  banner.classList.toggle("fail", !capacity.capacity_feasible);
  banner.innerHTML = capacity.capacity_feasible
    ? `<strong>✓ 关键rank容量可行</strong><span>剩余 ${formatBytes(capacity.headroom_bytes)}</span>`
    : `<strong>× 容量不足；下方继续显示理论性能</strong><span>缺口 ${formatBytes(capacity.capacity_shortfall_bytes)}</span>`;
  renderCapacity(capacity);
  $("#metric-parameters").textContent = formatParameters(result.model.parameters);
  $("#metric-model-kind").textContent = Object.entries(result.model.operator_mix).map(([name, count]) => `${name}×${count}`).join(" · ");
  renderHybrid(result);
  const performance = result.performance;
  if (!performance) {
    ["metric-ttft", "metric-tpot", "metric-prefill", "metric-decode", "metric-completion", "metric-speedup", "metric-parallel-efficiency"].forEach((id) => { document.getElementById(id).textContent = "未计算"; });
    $("#communication-block").hidden = true; $("#pipeline-block").hidden = true;
    renderOperatorTable(); $("#raw-json").textContent = JSON.stringify(result, null, 2); return;
  }
  const prefill = performance.prefill;
  const decode = performance.decode;
  const tpot = decode.device_inter_token_interval.mean_seconds;
  $("#metric-ttft").textContent = formatDuration(performance.first_token.ttft_seconds);
  $("#metric-tpot").textContent = formatDuration(tpot);
  $("#metric-prefill").textContent = formatRate(result.workload.batch_size * result.workload.prompt_length / prefill.latency_seconds);
  $("#metric-decode").textContent = formatRate(decode.steady_state_output_tokens_per_second);
  $("#metric-completion").textContent = formatDuration(performance.request.completion_latency_seconds);
  const scaling = performance.scaling;
  $("#metric-speedup").textContent = scaling?.available ? `${scaling.speedup.toFixed(2)}×` : (result.parallelism.device_count === 1 ? "1.00×" : "无基线");
  $("#metric-parallel-efficiency").textContent = scaling?.available ? `${(scaling.parallel_efficiency * 100).toFixed(1)}%` : (result.parallelism.device_count === 1 ? "100.0%" : "不适用");
  renderBarChart($("#latency-chart"), [
    { label: "TTFT", value: prefill.latency_seconds, display: (prefill.latency_seconds * 1e3).toFixed(2) },
    { label: "首次Decode", value: decode.first_step?.latency_seconds || 0, display: decode.first_step ? (decode.first_step.latency_seconds * 1e3).toFixed(2) : "—" },
    { label: "平均TPOT", value: tpot || 0, display: tpot ? (tpot * 1e3).toFixed(2) : "—" },
    { label: "末次Decode", value: decode.last_step?.latency_seconds || 0, display: decode.last_step ? (decode.last_step.latency_seconds * 1e3).toFixed(2) : "—" },
  ]);
  const effective = prefill.hbm.effective_bandwidth_bytes_per_second / 1e9;
  const prefillBw = prefill.hbm.average_achieved_bandwidth_bytes_per_second / 1e9;
  const decodeBw = decode.first_step ? decode.first_step.hbm.average_achieved_bandwidth_bytes_per_second / 1e9 : 0;
  renderBarChart($("#bandwidth-chart"), [
    { label: "有效上限", value: effective, display: effective.toFixed(1) },
    { label: "Prefill平均", value: prefillBw, display: prefillBw.toFixed(1) },
    { label: "首次Decode", value: decodeBw, display: decodeBw ? decodeBw.toFixed(1) : "—" },
  ]);
  renderCommunication(prefill, decode, result.parallelism);
  renderPipeline(prefill, result.parallelism.pipeline_parallel);
  renderOperatorTable(); $("#raw-json").textContent = JSON.stringify(result, null, 2);
}

export function downloadResult() {
  if (!latestResult) return;
  const blob = new Blob([JSON.stringify(latestResult, null, 2)], { type: "application/json;charset=utf-8" });
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob);
  link.download = "transformer_performance_result_v2.json"; link.click(); URL.revokeObjectURL(link.href);
}
