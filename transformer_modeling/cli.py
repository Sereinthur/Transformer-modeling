"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .estimators import estimate

#搭建命令行参数解析器
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transformer-model",
        description="按可替换算子估算Transformer容量、延迟、带宽和吞吐。",
    )
    parser.add_argument("config", type=Path, help="Schema v3 JSON 配置路径")
    parser.add_argument("-o", "--output", type=Path, help="Write result JSON to this path")
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Omit operator-level breakdowns from the result",
    )
    parser.add_argument("--compact", action="store_true", help="Print compact JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with args.config.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        config = Config.from_dict(raw_config)
        result = estimate(config, details=not args.no_details)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    indent = None if args.compact else 2
    rendered = json.dumps(result, ensure_ascii=False, indent=indent)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    # 即使容量不足也已经写出完整理论性能；退出码3只用于脚本识别部署不可行。
    return 0 if result["capacity"]["capacity_feasible"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
