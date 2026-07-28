/** 无依赖的轻量条形图渲染。 */

export function renderBarChart(container, rows) {
  const max = Math.max(...rows.map((row) => row.value || 0), 1e-15);
  container.innerHTML = "";
  rows.forEach((row) => {
    const element = document.createElement("div");
    element.className = "bar-row";
    const width = Math.max(0, Math.min(100, ((row.value || 0) / max) * 100));
    element.innerHTML = `
      <span class="bar-label">${row.label}</span>
      <span class="bar-track"><span class="bar-fill" style="display:block;width:${width}%"></span></span>
      <span class="bar-value">${row.display}</span>`;
    container.appendChild(element);
  });
}
