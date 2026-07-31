/** 固定Transformer骨架中的循环层Pattern编辑器。 */

import { $ } from "./dom.js";

const OPTIONS = {
  norm: [["rms_norm", "RMSNorm"], ["layer_norm", "LayerNorm"], ["unmodeled", "未建模（只计容量）"]],
  attention: [["standard_attention", "标准Attention"], ["kda", "KDA"], ["gated_mla", "Gated MLA"], ["csa_attention", "CSA压缩稀疏注意力"], ["hca_attention", "HCA重度压缩注意力"], ["unmodeled", "未建模（只计容量）"]],
  residual: [["standard_residual", "普通Residual"], ["attnres", "AttnRes"], ["mhc", "mHC超连接"], ["unmodeled", "未建模（只计容量）"]],
  ffn: [["dense_ffn", "Dense FFN"], ["gated_ffn", "Gated FFN / SwiGLU"], ["moe", "MoE / LatentMoE近似"], ["unmodeled", "未建模（只计容量）"]],
};

const optionHtml = (name, selected) => OPTIONS[name]
  .map(([value, label]) => `<option value="${value}"${value === selected ? " selected" : ""}>${label}</option>`)
  .join("");

function createRow(item = {}, index = 0) {
  const row = document.createElement("div");
  row.className = "pattern-row";
  row.dataset.patternRow = "";
  row.innerHTML = `
    <div class="pattern-row-heading"><strong>Pattern ${index + 1}</strong><button type="button" class="text-button pattern-remove">删除</button></div>
    <label class="field"><span>重复层数</span><input class="pattern-repeat" type="number" min="1" step="1" value="${Number(item.repeat || 1)}" required></label>
    <label class="field"><span>Norm算子</span><select class="pattern-norm">${optionHtml("norm", item.norm?.type || "rms_norm")}</select></label>
    <label class="field"><span>Attention算子</span><select class="pattern-attention">${optionHtml("attention", item.attention?.type || "standard_attention")}</select></label>
    <label class="field"><span>Residual算子</span><select class="pattern-residual">${optionHtml("residual", item.residual?.type || "standard_residual")}</select></label>
    <label class="field"><span>FFN算子</span><select class="pattern-ffn">${optionHtml("ffn", item.ffn?.type || "gated_ffn")}</select></label>`;
  return row;
}

function refreshRows() {
  const rows = [...document.querySelectorAll("[data-pattern-row]")];
  rows.forEach((row, index) => {
    row.querySelector("strong").textContent = `Pattern ${index + 1}`;
    row.querySelector(".pattern-remove").disabled = rows.length === 1;
  });
}

function notifyChanged() {
  $("#config-form").dispatchEvent(new Event("input", { bubbles: true }));
  document.dispatchEvent(new CustomEvent("patternchange"));
}

export function writePatternRows(pattern) {
  const editor = $("#layer-pattern-editor");
  editor.replaceChildren(...(pattern?.length ? pattern : [{}]).map(createRow));
  refreshRows();
}

export function readPatternRows() {
  return [...document.querySelectorAll("[data-pattern-row]")].map((row) => ({
    repeat: Number(row.querySelector(".pattern-repeat").value),
    norm: row.querySelector(".pattern-norm").value,
    attention: row.querySelector(".pattern-attention").value,
    residual: row.querySelector(".pattern-residual").value,
    ffn: row.querySelector(".pattern-ffn").value,
  }));
}

export function patternUses(type, slot = null) {
  return readPatternRows().some((row) => slot ? row[slot] === type : Object.values(row).includes(type));
}

export function initializePatternEditor() {
  $("#add-pattern-row").addEventListener("click", () => {
    $("#layer-pattern-editor").append(createRow({}, readPatternRows().length));
    refreshRows(); notifyChanged();
  });
  $("#layer-pattern-editor").addEventListener("click", (event) => {
    if (!event.target.classList.contains("pattern-remove")) return;
    if (document.querySelectorAll("[data-pattern-row]").length > 1) event.target.closest("[data-pattern-row]").remove();
    refreshRows(); notifyChanged();
  });
  $("#layer-pattern-editor").addEventListener("change", notifyChanged);
  writePatternRows([{}]);
}
