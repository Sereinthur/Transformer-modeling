"""使用同一份 TP 配置比较四种芯片互联拓扑。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from transformer_modeling import Config, estimate


EXAMPLE_DIR = Path(__file__).resolve().parent


def main() -> None:
    # 保持芯片、模型和请求不变，只替换互联拓扑，便于做同口径比较。
    base = json.loads((EXAMPLE_DIR / "tp4_gqa.json").read_text(encoding="utf-8"))
    print(f"{'拓扑':<10} {'TTFT(ms)':>12} {'TPOT(ms)':>12} {'加速比':>10} {'并行效率':>12}")
    for topology in ("ring", "bus", "crossbar", "mesh"):
        data = copy.deepcopy(base)
        interconnect = data["hardware"]["interconnect"]
        interconnect["topology"] = topology
        if topology == "mesh":
            # 4 个 rank 显式放入 2×2 Mesh；也可以省略，让模型自动推导。
            interconnect["mesh_rows"] = 2
            interconnect["mesh_columns"] = 2
        result = estimate(Config.from_dict(data), details=False)
        performance = result["performance"]
        scaling = performance["scaling"]
        print(
            f"{topology:<10} "
            f"{performance['first_token']['ttft_seconds'] * 1e3:>12.3f} "
            f"{performance['decode']['device_inter_token_interval']['mean_seconds'] * 1e3:>12.3f} "
            f"{scaling['speedup']:>10.3f} "
            f"{scaling['parallel_efficiency']:>12.3f}"
        )


if __name__ == "__main__":
    main()
