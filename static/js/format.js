/** 延迟、容量、速率和参数量的显示格式。 */

export function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return "不适用";
  if (seconds < 1e-6) return `${(seconds * 1e9).toFixed(2)} ns`;
  if (seconds < 1e-3) return `${(seconds * 1e6).toFixed(2)} µs`;
  if (seconds < 1) return `${(seconds * 1e3).toFixed(2)} ms`;
  return `${seconds.toFixed(3)} s`;
}

export function formatBytes(bytes) {
  if (bytes == null || !Number.isFinite(bytes)) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = Math.abs(bytes);
  let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return `${bytes < 0 ? "−" : ""}${value.toFixed(index < 2 ? 0 : 2)} ${units[index]}`;
}

export function formatRate(value) {
  if (value == null || !Number.isFinite(value)) return "不适用";
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: value >= 1000 ? 0 : 2 }).format(value);
}

export function formatParameters(value) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} M`;
  return formatRate(value);
}
