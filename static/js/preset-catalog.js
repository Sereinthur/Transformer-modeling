/** 手动模式和本机模式共用的真实模型预设目录。 */

let catalogPromise = null;

export async function loadModelPresets() {
  if (!catalogPromise) {
    catalogPromise = fetch("/api/model-presets")
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "真实模型预设加载失败");
        return payload.presets || [];
      });
  }
  return catalogPromise;
}

export function replacePresetOptions(select, presets, customLabel, customFirst = true) {
  select.replaceChildren();
  const custom = new Option(customLabel, "custom");
  if (customFirst) select.add(custom);
  presets.forEach((preset) => select.add(new Option(preset.name, preset.id)));
  if (!customFirst) select.add(custom);
}

export function presetDescription(preset) {
  const quality = preset.mapping_quality === "parameterized_draft"
    ? "参数化草案，官方字段与自动假设分开标注"
    : (preset.mapping_quality === "approximate" ? "近似映射" : "支持字段精确映射");
  const unsupported = preset.unsupported_features?.length
    ? `；未显式建模：${preset.unsupported_features.join("、")}`
    : "";
  return `${preset.family} · ${quality}${unsupported}`;
}

export function clonePresetValue(value) {
  return JSON.parse(JSON.stringify(value));
}
