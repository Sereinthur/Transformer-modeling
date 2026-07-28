/** 可视化应用的请求流程和事件绑定。 */

import { $, $$ } from "./dom.js";
import {
  activateTab,
  buildConfig,
  fillForm,
  populateDtypes,
  updateAttentionArchitectureFields,
  updateDecodePreview,
  updateFfnFields,
} from "./form.js";
import { updateDeviceCount, updateTopologyFields } from "./topology.js";
import { downloadResult, renderOperatorTable, renderResult } from "./results.js";
import { initializeManualPresets } from "./manual-presets.js";
import { initializePatternEditor } from "./pattern-editor.js";

let exampleConfig = null;
let manualPresetController = null;

export function showError(message) {
  $("#empty-state").hidden = true;
  $("#loading-state").hidden = true;
  $("#result-content").hidden = true;
  const error = $("#error-state");
  error.textContent = `无法完成计算：${message}`;
  error.hidden = false;
}

export async function estimateConfig(config, button = null) {
  $("#empty-state").hidden = true;
  $("#error-state").hidden = true;
  $("#result-content").hidden = true;
  $("#loading-state").hidden = false;
  if (button) button.disabled = true;
  try {
    const response = await fetch("/api/estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "参数校验失败");
    renderResult(payload);
  } catch (error) {
    showError(error.message || String(error));
  } finally {
    if (button) button.disabled = false;
  }
}

async function calculate(event) {
  event.preventDefault();
  try {
    const presetError = manualPresetController?.validationError();
    if (presetError) {
      activateTab("model");
      showError(presetError);
      return;
    }
    const form = $("#config-form");
    if (!form.checkValidity()) {
      const invalid = form.querySelector(":invalid");
      const section = invalid?.closest(".form-section");
      if (section) activateTab(section.dataset.section);
      invalid?.reportValidity();
      return;
    }
    await estimateConfig(buildConfig(), $("#calculate"));
  } catch (error) {
    showError(error.message || String(error));
  }
}

export async function initialize() {
  populateDtypes();
  initializePatternEditor();
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
  $("#config-form").addEventListener("submit", calculate);
  $("#reset-example").addEventListener("click", () => {
    if (!exampleConfig) return;
    fillForm(exampleConfig);
    manualPresetController?.syncFromConfig(exampleConfig);
  });
  $("#output-length").addEventListener("input", updateDecodePreview);
  document.addEventListener("patternchange", () => {
    updateFfnFields();
    updateAttentionArchitectureFields();
  });
  $("#hybrid-moe-variant").addEventListener("change", updateAttentionArchitectureFields);
  $("#tensor-parallel").addEventListener("input", updateDeviceCount);
  $("#expert-parallel").addEventListener("input", updateDeviceCount);
  $("#pipeline-parallel").addEventListener("input", updateDeviceCount);
  $("#interconnect-topology").addEventListener("change", updateTopologyFields);
  $("#operator-phase").addEventListener("change", renderOperatorTable);
  $("#download-result").addEventListener("click", downloadResult);
  try {
    const response = await fetch("/api/example");
    if (!response.ok) throw new Error("示例配置加载失败");
    exampleConfig = await response.json();
    fillForm(exampleConfig);
    manualPresetController = await initializeManualPresets({
      getConfig: buildConfig,
      applyConfig: fillForm,
    });
    manualPresetController.syncFromConfig(exampleConfig);
  } catch (error) {
    showError(error.message || String(error));
  }
}
