/** Schema v2算子组合表单的填充、读取与页签交互。 */

import { $, $$, numberValue, optionalNumber, setValue } from "./dom.js";
import { nearestMeshDimensions, updateTopologyFields } from "./topology.js";
import { patternUses, readPatternRows, writePatternRows } from "./pattern-editor.js";

const DTYPE_OPTIONS = ["fp32", "tf32", "bf16", "fp16", "fp8", "mxfp8", "int8", "int4", "mxfp4"];
let modelMetadata = {};
let modelLayerPrefix = [];
let modelIdentity = { id: "custom", name: "自定义Transformer" };

export function populateDtypes() {
  [
    "weight-dtype", "activation-dtype", "kv-dtype", "logits-dtype", "kda-state-dtype",
    "accumulation-dtype", "attention-weight-dtype", "routed-expert-dtype", "shared-expert-dtype",
  ].forEach((id) => {
    const select = document.getElementById(id);
    select.innerHTML = DTYPE_OPTIONS.map((dtype) => `<option value="${dtype}">${dtype.toUpperCase()}</option>`).join("");
  });
}

function firstOperator(model, slot) {
  return model.layer_pattern.map((item) => item[slot]).find(Boolean) || {};
}

function operatorOfType(model, type) {
  return model.layer_pattern.map((item) => item.attention).find((item) => item?.type === type) || {};
}

export function fillForm(config) {
  const h = config.hardware;
  const m = config.model;
  const d = m.dimensions;
  const s = config.serving;
  const e = config.execution;
  const p = config.parallelism || {};
  const interconnect = h.interconnect || {};
  const throughput = h.compute.throughput;
  const measured = h.compute.measured_throughput || {};
  modelMetadata = JSON.parse(JSON.stringify(m.metadata || {}));
  // 预设的前置层不可在编辑器里表达，原样保留避免丢失层排列。
  modelLayerPrefix = JSON.parse(JSON.stringify(m.layer_prefix || []));
  modelIdentity = { id: m.id || "custom", name: m.name || "自定义Transformer" };

  setValue("hardware-name", h.name);
  setValue("peak-tflops", Object.values(throughput).find((value) => value != null) / 1e12);
  const measuredValue = Object.values(measured).find((value) => value != null);
  setValue("measured-tflops", measuredValue == null ? "" : measuredValue / 1e12);
  setValue("memory-gib", h.device_memory.capacity_bytes / 2 ** 30);
  setValue("peak-bandwidth", h.device_memory.peak_bandwidth_bytes_per_second / 1e9);
  const measuredBw = h.device_memory.measured_read_bandwidth_bytes_per_second;
  setValue("measured-bandwidth", measuredBw == null ? "" : measuredBw / 1e9);
  setValue("launch-us", h.runtime.kernel_launch_latency_seconds * 1e6);
  setValue("baseline-unavailable-gib", (h.device_memory.baseline_unavailable_bytes || 0) / 2 ** 30);
  setValue("reserved-gib", h.device_memory.reserved_capacity_bytes / 2 ** 30);
  setValue("tile-m", h.compute.matrix_tile.m);
  setValue("tile-n", h.compute.matrix_tile.n);
  setValue("tile-k", h.compute.matrix_tile.k);

  setValue("layer-count", d.layer_count);
  setValue("hidden-size", d.hidden_size);
  setValue("intermediate-size", d.intermediate_size);
  setValue("vocab-size", d.vocab_size);
  setValue("padded-vocab-size", d.padded_vocab_size);
  const standard = operatorOfType(m, "standard_attention");
  const kda = operatorOfType(m, "kda");
  const mla = operatorOfType(m, "gated_mla");
  const csa = operatorOfType(m, "csa_attention");
  const hca = operatorOfType(m, "hca_attention");
  const compressed = csa.type ? csa : hca;
  const unmodeled = m.layer_pattern.flatMap((item) => [item.norm, item.attention, item.residual, item.ffn]).find((item) => item?.type === "unmodeled") || {};
  const attention = kda.type ? kda : (mla.type ? mla : standard);
  writePatternRows(m.layer_pattern);
  setValue("query-heads", standard.query_heads ?? mla.query_heads ?? kda.heads ?? 1);
  setValue("kv-heads", standard.kv_heads ?? standard.query_heads ?? mla.query_heads ?? kda.heads ?? 1);
  setValue("head-dim", standard.head_dim ?? Math.max(1, Math.floor(d.hidden_size / (standard.query_heads || mla.query_heads || kda.heads || 1))));
  setValue("query-width-mode", standard.query_width_equals_hidden === false ? "heads_times_head_dim" : "hidden");
  setValue("hybrid-kda-heads", kda.heads ?? attention.query_heads ?? 64);
  setValue("kda-key-dim", kda.key_dim ?? 128);
  setValue("kda-value-dim", kda.value_dim ?? 128);
  setValue("hybrid-kda-conv-size", kda.short_conv_kernel_size ?? 4);
  setValue("kda-chunk-size", kda.chunk_size ?? 256);
  setValue("mla-q-rank", mla.q_lora_rank ?? 1536);
  setValue("mla-kv-rank", mla.kv_lora_rank ?? 512);
  setValue("mla-nope-dim", mla.qk_nope_head_dim ?? 128);
  setValue("mla-rope-dim", mla.qk_rope_head_dim ?? 64);
  setValue("mla-v-dim", mla.v_head_dim ?? 128);
  const residual = firstOperator(m, "residual");
  setValue("attnres-blocks", residual.block_count ?? 8);
  setValue("attention-sliding-window", standard.sliding_window ?? compressed.sliding_window ?? 0);
  setValue("attention-q-lora-rank", standard.q_lora_rank ?? compressed.q_lora_rank ?? 0);
  setValue("attention-o-lora-rank", standard.o_lora_rank ?? compressed.o_lora_rank ?? 0);
  setValue("csa-compress-ratio", csa.compress_ratio ?? 4);
  setValue("csa-compress-overlap", csa.compress_overlap ?? 2);
  setValue("csa-selected-entries", csa.selected_entries ?? 1024);
  setValue("csa-indexer-heads", csa.indexer_heads ?? 128);
  setValue("csa-indexer-head-dim", csa.indexer_head_dim ?? 128);
  setValue("hca-compress-ratio", hca.compress_ratio ?? 128);
  setValue("hca-selected-entries", hca.selected_entries ?? 1024);
  setValue("mhc-channels", residual.channels ?? 4);
  setValue("mhc-sinkhorn-iters", residual.sinkhorn_iters ?? 20);

  const ffn = firstOperator(m, "ffn");
  const moe = ffn.type === "moe";
  setValue("ffn-type", ffn.activation ?? (ffn.type === "dense_ffn" ? "dense" : "swiglu"));
  setValue("expert-count", ffn.expert_count ?? 8);
  setValue("experts-per-token", ffn.experts_per_token ?? 2);
  setValue("expert-intermediate-size", ffn.expert_intermediate_size ?? d.intermediate_size);
  setValue("shared-expert-intermediate-size", ffn.shared_expert_intermediate_size ?? 0);
  setValue("shared-expert-count", ffn.shared_expert_count ?? 1);
  setValue("moe-routing", ffn.routing ?? "learned");
  setValue("routing-imbalance-factor", 1);
  setValue("hybrid-moe-variant", ffn.implementation === "latent_moe_approx" ? "latent_moe_approximation" : "standard_moe");
  setValue("situ-ops", ffn.situ_ops_per_element ?? 6);
  setValue("unmodeled-name", unmodeled.name ?? "待补充结构");
  setValue("unmodeled-operator-parameters-b", (unmodeled.parameter_count || 0) / 1e9);
  setValue("unmodeled-operator-state-gib", (unmodeled.state_bytes || 0) / 2 ** 30);
  setValue("tied-lm-head", Boolean(m.embedding.tied_lm_head));
  setValue("causal", true);
  setValue("weight-dtype", m.dtype.weight);
  setValue("activation-dtype", m.dtype.activation);
  setValue("kv-dtype", m.dtype.kv_cache);
  setValue("logits-dtype", m.dtype.logits);
  setValue("kda-state-dtype", m.dtype.state ?? m.dtype.kv_cache);
  setValue("accumulation-dtype", m.dtype.accumulation ?? "fp32");
  setValue("attention-weight-dtype", standard.weight_dtype ?? compressed.weight_dtype ?? m.dtype.weight);
  setValue("routed-expert-dtype", ffn.routed_expert_weight_dtype ?? ffn.weight_dtype ?? m.dtype.weight);
  setValue("shared-expert-dtype", ffn.shared_expert_weight_dtype ?? ffn.weight_dtype ?? m.dtype.weight);
  setValue("quant-block-size", m.quantization?.block_size ?? 32);
  setValue("quant-scale-bytes", m.quantization?.scale_bytes ?? 1);
  setValue("hybrid-parameter-target-b", (m.metadata?.parameter_target || 0) / 1e9);
  setValue("hybrid-unattributed-b", (m.extra?.parameter_count || 0) / 1e9);
  $("#k3-provenance-note").hidden = m.metadata?.mapping_quality !== "parameterized_draft";

  setValue("batch-size", s.batch_size);
  setValue("prompt-length", s.prompt_length.value ?? s.prompt_length);
  setValue("output-length", s.output_length.value ?? s.output_length);
  setValue("max-sequence-length", s.max_sequence_length);
  const prefix = s.prefix_cache || {};
  setValue("kda-state-hit-rate", prefix.kda_state_hit_rate ?? 0);
  setValue("kda-cached-prefix-tokens", prefix.kda_cached_prefix_tokens ?? 0);
  setValue("mla-prefix-hit-rate", prefix.mla_prefix_hit_rate ?? 0);
  setValue("mla-matched-tokens", prefix.mla_average_matched_tokens ?? 0);
  setValue("hybrid-cache-block-tokens", prefix.block_tokens ?? 16);
  const deployment = config.deployment || {};
  const transfer = deployment.transfer || {};
  setValue("deployment-mode", deployment.mode ?? "aggregated");
  setValue("prefill-replicas", deployment.prefill_replicas ?? 1);
  setValue("decode-replicas", deployment.decode_replicas ?? 1);
  setValue("pd-bandwidth", (transfer.effective_bandwidth_bytes_per_second ?? 100e9) / 1e9);
  setValue("pd-latency", (transfer.latency_seconds ?? 5e-6) * 1e6);
  setValue("pd-overlap-rho", transfer.overlap_rho ?? 0.5);

  setValue("prefill-gemm", e.efficiencies.prefill_gemm);
  setValue("decode-gemm", e.efficiencies.decode_gemm);
  setValue("prefill-attention", e.efficiencies.prefill_attention);
  setValue("decode-attention", e.efficiencies.decode_attention);
  setValue("vector-efficiency", e.efficiencies.vector);
  setValue("hbm-efficiency", e.efficiencies.hbm);
  setValue("overlap-rho", e.overlap.interpolation_rho);
  setValue("prefill-tp-overlap-rho", e.overlap.prefill_tp_interpolation_rho ?? 0.5);
  setValue("decode-tp-overlap-rho", e.overlap.decode_tp_interpolation_rho ?? 1);
  setValue("ep-overlap-rho", e.overlap.ep_interpolation_rho ?? 1);
  setValue("kv-page-tokens", e.memory.kv_page_tokens);
  setValue("prefill-logits-mode", m.inference?.prefill_logits_mode ?? "last_token");
  setValue("flash-attention", e.fusion.flash_attention);
  setValue("rope-kv-write", e.fusion.rope_kv_write);
  setValue("gated-mlp", e.fusion.gated_mlp);
  setValue("rmsnorm-linear", e.fusion.rmsnorm_linear);
  setValue("kv-paged", e.memory.kv_paged);

  setValue("device-count", h.device_count);
  setValue("tensor-parallel", p.tensor_parallel ?? 1);
  setValue("expert-parallel", p.expert_parallel ?? 1);
  setValue("pipeline-parallel", p.pipeline_parallel ?? 1);
  setValue("pipeline-microbatches", p.pipeline_microbatches ?? 1);
  setValue("pipeline-stage-boundaries", (p.pipeline_stage_boundaries || []).join(", "));
  setValue("kv-head-policy", p.kv_head_policy ?? "shard_or_group_replicate");
  setValue("interconnect-topology", interconnect.topology ?? "ring");
  const [rows, columns] = nearestMeshDimensions(p.tensor_parallel ?? 1);
  setValue("mesh-rows", interconnect.mesh_rows ?? rows);
  setValue("mesh-columns", interconnect.mesh_columns ?? columns);
  setValue("interconnect-bandwidth", (interconnect.effective_channel_bandwidth_bytes_per_second ?? 100e9) / 1e9);
  setValue("collective-step-latency", (interconnect.collective_step_latency_seconds ?? 2e-6) * 1e6);
  setValue("pipeline-bandwidth", (interconnect.pipeline_effective_bandwidth_bytes_per_second ?? interconnect.effective_channel_bandwidth_bytes_per_second ?? 100e9) / 1e9);
  setValue("pipeline-transfer-latency", (interconnect.pipeline_transfer_latency_seconds ?? interconnect.collective_step_latency_seconds ?? 2e-6) * 1e6);
  updateTopologyFields(); updateFfnFields(); updateAttentionArchitectureFields(); updateDecodePreview();
}

export function updateFfnFields() {
  const moe = patternUses("moe", "ffn");
  $("#moe-fields").hidden = !moe;
  $("#intermediate-size-label").textContent = moe ? "默认FFN维度" : "FFN维度 I";
  $("#ffn-architecture-note").textContent = moe
    ? "MoE算子包含Router、Top-K、专家GEMM和Combine；EP>1时加入两次All-to-All。"
    : "使用Dense或Gated FFN算子。";
}

export function updateAttentionArchitectureFields() {
  const sequenceState = patternUses("kda", "attention") || patternUses("gated_mla", "attention");
  const specialized = sequenceState || patternUses("attnres", "residual") || patternUses("unmodeled");
  const csa = patternUses("csa_attention", "attention");
  const hca = patternUses("hca_attention", "attention");
  const mhc = patternUses("mhc", "residual");
  const standard = patternUses("standard_attention", "attention");
  $("#hybrid-fields").hidden = !specialized;
  $("#hybrid-serving-fields").hidden = !sequenceState;
  $("#compressed-attention-fields").hidden = !(standard || csa || hca || mhc);
  ["csa-compress-ratio", "csa-compress-overlap", "csa-selected-entries", "csa-indexer-heads", "csa-indexer-head-dim"]
    .forEach((id) => { document.getElementById(id).disabled = !csa; });
  ["hca-compress-ratio", "hca-selected-entries"].forEach((id) => { document.getElementById(id).disabled = !hca; });
  ["mhc-channels", "mhc-sinkhorn-iters"].forEach((id) => { document.getElementById(id).disabled = !mhc; });
  $("#attention-sliding-window").disabled = !(standard || csa || hca);
  $("#attnres-blocks").disabled = !patternUses("attnres", "residual");
  $("#situ-ops").disabled = $("#hybrid-moe-variant").value === "standard_moe";
}

function ffnSpec(type) {
  if (type === "unmodeled") return unmodeledSpec();
  if (type !== "moe") return { type, implementation: type === "dense_ffn" ? "gelu" : "swiglu", intermediate_size: numberValue("intermediate-size") };
  return { type: "moe", implementation: $("#hybrid-moe-variant").value === "latent_moe_approximation" ? "latent_moe_approx" : "standard_moe", expert_count: numberValue("expert-count"), experts_per_token: numberValue("experts-per-token"), expert_intermediate_size: numberValue("expert-intermediate-size"), shared_expert_intermediate_size: numberValue("shared-expert-intermediate-size"), shared_expert_count: numberValue("shared-expert-count"), routing: $("#moe-routing").value, activation: $("#ffn-type").value, situ_ops_per_element: numberValue("situ-ops"), routed_expert_weight_dtype: $("#routed-expert-dtype").value, shared_expert_weight_dtype: $("#shared-expert-dtype").value };
}

function compressedAttentionSpec(type) {
  const csa = type === "csa_attention";
  return {
    type, implementation: "compressed_kv",
    query_heads: numberValue("query-heads"), kv_heads: numberValue("kv-heads"), head_dim: numberValue("head-dim"),
    compress_ratio: numberValue(csa ? "csa-compress-ratio" : "hca-compress-ratio"),
    compress_overlap: csa ? numberValue("csa-compress-overlap") : 1,
    selected_entries: numberValue(csa ? "csa-selected-entries" : "hca-selected-entries"),
    sliding_window: csa ? numberValue("attention-sliding-window") : 0,
    selector: csa ? "indexer" : "uniform",
    indexer_heads: numberValue("csa-indexer-heads"), indexer_head_dim: numberValue("csa-indexer-head-dim"),
    qk_rope_head_dim: numberValue("mla-rope-dim"),
    q_lora_rank: numberValue("attention-q-lora-rank"), o_lora_rank: numberValue("attention-o-lora-rank"),
    weight_dtype: $("#attention-weight-dtype").value,
  };
}

function attentionSpec(type) {
  if (type === "unmodeled") return unmodeledSpec();
  if (type === "kda") return { type, implementation: "chunkwise", heads: numberValue("hybrid-kda-heads"), key_dim: numberValue("kda-key-dim"), value_dim: numberValue("kda-value-dim"), short_conv_kernel_size: numberValue("hybrid-kda-conv-size"), chunk_size: numberValue("kda-chunk-size") };
  if (type === "gated_mla") return { type, implementation: "latent_cache", query_heads: numberValue("query-heads"), q_lora_rank: numberValue("mla-q-rank"), kv_lora_rank: numberValue("mla-kv-rank"), qk_nope_head_dim: numberValue("mla-nope-dim"), qk_rope_head_dim: numberValue("mla-rope-dim"), v_head_dim: numberValue("mla-v-dim") };
  if (type === "csa_attention" || type === "hca_attention") return compressedAttentionSpec(type);
  return { type: "standard_attention", implementation: $("#flash-attention").checked ? "flash_attention" : "standard", query_heads: numberValue("query-heads"), kv_heads: numberValue("kv-heads"), head_dim: numberValue("head-dim"), query_width_equals_hidden: $("#query-width-mode").value === "hidden", sliding_window: numberValue("attention-sliding-window"), q_lora_rank: numberValue("attention-q-lora-rank"), o_lora_rank: numberValue("attention-o-lora-rank"), weight_dtype: $("#attention-weight-dtype").value };
}

function unmodeledSpec() {
  return {
    type: "unmodeled",
    name: $("#unmodeled-name").value.trim() || "待补充结构",
    parameter_count: Math.round(numberValue("unmodeled-operator-parameters-b") * 1e9),
    state_bytes: Math.round(numberValue("unmodeled-operator-state-gib") * 2 ** 30),
    note: "只计容量；延迟只对应已建模代理结构，不构成严格上下界。",
  };
}

function residualSpec(type) {
  if (type === "unmodeled") return unmodeledSpec();
  if (type === "attnres") return { type: "attnres", block_count: numberValue("attnres-blocks") };
  if (type === "mhc") return { type: "mhc", channels: numberValue("mhc-channels"), sinkhorn_iters: numberValue("mhc-sinkhorn-iters") };
  return { type: "standard_residual" };
}

function modelSpec() {
  const pattern = readPatternRows().map((row) => ({
    repeat: row.repeat,
    norm: row.norm === "unmodeled" ? unmodeledSpec() : { type: row.norm },
    attention: attentionSpec(row.attention),
    residual: residualSpec(row.residual),
    ffn: ffnSpec(row.ffn),
  }));
  return {
    id: modelIdentity.id, name: modelIdentity.name,
    dimensions: { layer_count: numberValue("layer-count"), hidden_size: numberValue("hidden-size"), intermediate_size: numberValue("intermediate-size"), vocab_size: numberValue("vocab-size"), padded_vocab_size: numberValue("padded-vocab-size") },
    embedding: { type: "token_embedding", tied_lm_head: $("#tied-lm-head").checked }, layer_pattern: pattern,
    ...(modelLayerPrefix.length ? { layer_prefix: JSON.parse(JSON.stringify(modelLayerPrefix)) } : {}),
    output: { norm: { type: "rms_norm" }, head: { type: "lm_head" }, sampling: { type: "sampling" } },
    dtype: { weight: $("#weight-dtype").value, activation: $("#activation-dtype").value, kv_cache: $("#kv-dtype").value, state: $("#kda-state-dtype").value, accumulation: $("#accumulation-dtype").value, logits: $("#logits-dtype").value },
    quantization: { block_size: numberValue("quant-block-size"), scale_bytes: numberValue("quant-scale-bytes") },
    inference: { prefill_logits_mode: $("#prefill-logits-mode").value },
    extra: { parameter_count: Math.round(numberValue("hybrid-unattributed-b") * 1e9), sharding: "tp_ep" },
    metadata: { ...modelMetadata, parameter_target: Math.round(numberValue("hybrid-parameter-target-b") * 1e9) },
  };
}

export function buildConfig() {
  const weight = $("#weight-dtype").value;
  const key = weight === "mxfp4" ? "mxfp4_mxfp8_ops_per_second" : `${weight}_dense_ops_per_second`;
  const measured = optionalNumber("measured-tflops");
  const measuredBw = optionalNumber("measured-bandwidth");
  const baseline = optionalNumber("baseline-unavailable-gib");
  const hybrid = patternUses("kda", "attention") || patternUses("gated_mla", "attention");
  const config = {
    schema_version: 2,
    hardware: { name: $("#hardware-name").value.trim(), device_count: numberValue("device-count"), compute: { throughput: { [key]: numberValue("peak-tflops") * 1e12 }, ...(measured == null ? {} : { measured_throughput: { [key]: measured * 1e12 } }), matrix_tile: { m: numberValue("tile-m"), n: numberValue("tile-n"), k: numberValue("tile-k") } }, device_memory: { type: "HBM", capacity_bytes: Math.round(numberValue("memory-gib") * 2 ** 30), peak_bandwidth_bytes_per_second: numberValue("peak-bandwidth") * 1e9, measured_read_bandwidth_bytes_per_second: measuredBw == null ? null : measuredBw * 1e9, ...(baseline == null ? {} : { baseline_unavailable_bytes: Math.round(baseline * 2 ** 30) }), reserved_capacity_bytes: Math.round(numberValue("reserved-gib") * 2 ** 30) }, runtime: { kernel_launch_latency_seconds: numberValue("launch-us") * 1e-6 }, interconnect: { topology: $("#interconnect-topology").value, effective_channel_bandwidth_bytes_per_second: numberValue("interconnect-bandwidth") * 1e9, collective_step_latency_seconds: numberValue("collective-step-latency") * 1e-6, pipeline_effective_bandwidth_bytes_per_second: numberValue("pipeline-bandwidth") * 1e9, pipeline_transfer_latency_seconds: numberValue("pipeline-transfer-latency") * 1e-6, mesh_rows: $("#interconnect-topology").value === "mesh" ? numberValue("mesh-rows") : null, mesh_columns: $("#interconnect-topology").value === "mesh" ? numberValue("mesh-columns") : null } },
    model: modelSpec(),
    serving: { batch_size: numberValue("batch-size"), prompt_length: { distribution: "fixed", value: numberValue("prompt-length") }, output_length: { distribution: "fixed", value: numberValue("output-length") }, max_sequence_length: numberValue("max-sequence-length"), prefix_cache: { kda_state_hit_rate: numberValue("kda-state-hit-rate"), kda_cached_prefix_tokens: numberValue("kda-cached-prefix-tokens"), mla_prefix_hit_rate: numberValue("mla-prefix-hit-rate"), mla_average_matched_tokens: numberValue("mla-matched-tokens"), block_tokens: numberValue("hybrid-cache-block-tokens"), metadata_bytes_per_block: 16 } },
    execution: { fusion: { flash_attention: $("#flash-attention").checked, rope_kv_write: $("#rope-kv-write").checked, gated_mlp: $("#gated-mlp").checked, rmsnorm_linear: $("#rmsnorm-linear").checked }, overlap: { mode: "bounded_interpolation", interpolation_rho: numberValue("overlap-rho"), prefill_tp_interpolation_rho: numberValue("prefill-tp-overlap-rho"), decode_tp_interpolation_rho: numberValue("decode-tp-overlap-rho"), ep_interpolation_rho: numberValue("ep-overlap-rho") }, efficiencies: { prefill_gemm: numberValue("prefill-gemm"), decode_gemm: numberValue("decode-gemm"), prefill_attention: numberValue("prefill-attention"), decode_attention: numberValue("decode-attention"), vector: numberValue("vector-efficiency"), hbm: numberValue("hbm-efficiency") }, memory: { kv_paged: $("#kv-paged").checked, kv_page_tokens: numberValue("kv-page-tokens") } },
    parallelism: { tensor_parallel: numberValue("tensor-parallel"), expert_parallel: numberValue("expert-parallel"), pipeline_parallel: numberValue("pipeline-parallel"), pipeline_microbatches: numberValue("pipeline-microbatches"), kv_head_policy: $("#kv-head-policy").value },
  };
  const stageBoundaries = $("#pipeline-stage-boundaries").value.split(",").map((item) => item.trim()).filter(Boolean).map(Number);
  if (stageBoundaries.length && stageBoundaries.every((value) => Number.isInteger(value) && value > 0)) config.parallelism.pipeline_stage_boundaries = stageBoundaries;
  if (hybrid) config.deployment = { mode: $("#deployment-mode").value, prefill_replicas: numberValue("prefill-replicas"), decode_replicas: numberValue("decode-replicas"), transfer: { effective_bandwidth_bytes_per_second: numberValue("pd-bandwidth") * 1e9, latency_seconds: numberValue("pd-latency") * 1e-6, overlap_rho: numberValue("pd-overlap-rho") } };
  return config;
}

export function activateTab(name) {
  $$(".tab").forEach((tab, index) => { const active = tab.dataset.tab === name; tab.classList.toggle("active", active); tab.setAttribute("aria-selected", String(active)); if (active) $("#step-indicator").textContent = `${index + 1} / ${$$(".tab").length}`; });
  $$(".form-section").forEach((section) => { const active = section.dataset.section === name; section.classList.toggle("active", active); section.hidden = !active; });
}

export function updateDecodePreview() {
  const output = Math.max(1, numberValue("output-length") || 1);
  $("#decode-step-preview").textContent = `${output} − 1 = ${output - 1} 步`;
}
