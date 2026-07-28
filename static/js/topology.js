/** 并行规模与互联拓扑表单联动。 */

import { $, numberValue, setValue } from "./dom.js";

const TOPOLOGY_NOTES = {
  ring: ["Ring 模型", "Reduce-Scatter + All-Gather；每层 2×All-Reduce"],
  bus: ["Bus 模型", "共享总线串行上传并广播；带宽为整条总线共享带宽"],
  crossbar: ["Crossbar 模型", "非阻塞端口 + Recursive Doubling；2次幂TP最准确"],
  mesh: ["Mesh 模型", "二维维度有序路由；All-Reduce采用保守生成树模型"],
};
let latestResult = null;
let exampleConfig = null;

export function nearestMeshDimensions(ranks) {
  let rows = Math.max(1, Math.floor(Math.sqrt(ranks)));
  while (ranks % rows !== 0) rows -= 1;
  return [rows, ranks / rows];
}

export function updateTopologyFields() {
  const topology = $("#interconnect-topology").value;
  const mesh = topology === "mesh";
  $("#mesh-fields").hidden = !mesh;
  $("#mesh-rows").required = mesh;
  $("#mesh-columns").required = mesh;
  if (mesh) {
    const tp = Math.max(1, numberValue("tensor-parallel") || 1);
    const currentRows = numberValue("mesh-rows");
    const currentColumns = numberValue("mesh-columns");
    if (!currentRows || !currentColumns || currentRows * currentColumns !== tp) {
      const [rows, columns] = nearestMeshDimensions(tp);
      setValue("mesh-rows", rows);
      setValue("mesh-columns", columns);
    }
  }
  const [title, note] = TOPOLOGY_NOTES[topology];
  $("#topology-note-title").textContent = title;
  $("#topology-note").textContent = note;
}

export function updateDeviceCount() {
  const tp = Math.max(1, numberValue("tensor-parallel") || 1);
  const ep = Math.max(1, numberValue("expert-parallel") || 1);
  const pp = Math.max(1, numberValue("pipeline-parallel") || 1);
  setValue("device-count", tp * ep * pp);
  updateTopologyFields();
}
