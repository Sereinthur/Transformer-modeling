"""流程图编辑器HTTP服务：只做数据搬运，计算全部交给 transformer_modeling。"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from transformer_modeling import Config, estimate  # noqa: E402 - 需先补全sys.path
from transformer_modeling.models import (  # noqa: E402
    preset_catalog,
    resolve_model_definition,
)
from transformer_modeling.operators import get_operator_catalog  # noqa: E402

from .flowchart_schema import config_to_flowchart, flowchart_to_config
from .operator_schemas import schema_payload

PACKAGE_ROOT = Path(__file__).resolve().parent
# PyInstaller extracts bundled resources into _MEIPASS rather than the source tree.
PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", PACKAGE_ROOT.parents[1]))
STATIC_ROOT = PACKAGE_ROOT / "static"
EXAMPLE_CONFIG = PROJECT_ROOT / "examples" / "single_chip_gqa.json"
MAX_REQUEST_BYTES = 4_000_000

# 相对fp16 dense峰值的占位倍率（按元素字节折算），只用于让预设开箱可算；
# 真实评估时用户应在硬件面板填入datasheet或实测值。
_THROUGHPUT_SCALE = {
    "fp32": 0.5, "tf32": 0.5, "fp16": 1.0, "bf16": 1.0,
    "fp8": 2.0, "mxfp8": 2.0, "int8": 2.0, "int4": 4.0, "mxfp4": 4.0,
}


class ClientError(ValueError):
    """请求内容不合法：统一映射为400。"""


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ClientError(f"{name} 必须是对象")
    return value


class FlowchartHandler(BaseHTTPRequestHandler):
    """静态页面 + 流程图/配置转换 + 性能评估接口。"""

    server_version = "VisualModelingUI/0.1"

    # ---------------- 基础响应 ----------------

    def _send_bytes(
        self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        # 本地开发时前端可能跑在另一个端口（如Vite），放开跨域读取。
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(self, data: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8", status)

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ClientError("请求长度无效") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ClientError("请求内容为空或过大")
        try:
            raw = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClientError(f"请求JSON无法解析：{exc}") from exc
        return _require_dict(raw, "请求体")

    # ---------------- 路由 ----------------

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._send_bytes(b"", "text/plain; charset=utf-8", HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = unquote(urlparse(self.path).path)
        handlers: dict[str, Callable[[], object]] = {
            "/api/operator-catalog": get_operator_catalog,
            "/api/operator-schemas": schema_payload,
            "/api/model-presets": preset_catalog,
            "/api/example-config": self._example_config,
        }
        handler = handlers.get(route)
        if handler is not None:
            self._dispatch(handler)
            return
        if route == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if route.startswith("/api/"):
            self._send_json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
            return
        self._serve_static(route)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = unquote(urlparse(self.path).path)
        handlers: dict[str, Callable[[dict[str, Any]], object]] = {
            "/api/resolve-preset": self._resolve_preset,
            "/api/config-to-flowchart": self._config_to_flowchart,
            "/api/flowchart-to-config": self._flowchart_to_config,
            "/api/estimate": self._estimate,
        }
        handler = handlers.get(route)
        if handler is None:
            self._send_json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
            return
        try:
            body = self._read_json_body()
        except ClientError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._dispatch(lambda: handler(body))

    def _dispatch(self, action: Callable[[], object]) -> None:
        """统一异常出口：配置类错误→400，其余→500。"""

        try:
            self._send_json(action())
        except (ClientError, TypeError, ValueError, KeyError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001 - 兜底避免整条连接静默断开
            self._send_json(
                {"error": f"服务内部错误：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

    # ---------------- 接口实现 ----------------

    def _example_config(self) -> dict[str, Any]:
        try:
            return json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"无法读取示例配置：{exc}") from exc

    def _resolve_preset(self, body: dict[str, Any]) -> dict[str, Any]:
        preset_id = body.get("preset_id")
        hf_config = body.get("hf_config")
        if preset_id is not None and not isinstance(preset_id, str):
            raise ClientError("preset_id 必须是字符串")
        if hf_config is not None and not isinstance(hf_config, dict):
            raise ClientError("hf_config 必须是对象")
        resolved = resolve_model_definition(
            preset_id=preset_id,
            hf_config=hf_config,
            scenario=str(body.get("scenario", "base")),
        )
        # 附上可直接送回 /api/config-to-flowchart 或 /api/estimate 的完整config。
        return {**resolved, "config": self._assemble_config(resolved)}

    def _assemble_config(self, resolved: dict[str, Any]) -> dict[str, Any]:
        """用示例配置作为硬件/负载模板，套上解析出的model段。"""

        config = self._example_config()
        config.pop("_说明", None)
        model = resolved.get("resolved_model", {})
        config["model"] = model
        weight = str(model.get("dtype", {}).get("weight", "fp16")).lower()
        key = (
            "mxfp4_mxfp8_ops_per_second" if weight == "mxfp4"
            else f"{weight}_dense_ops_per_second"
        )
        throughput = config["hardware"]["compute"]["throughput"]
        reference = float(throughput.get("fp16_dense_ops_per_second", 1e14))
        throughput.setdefault(key, reference * _THROUGHPUT_SCALE.get(weight, 1.0))
        throughput["_说明"] = "非fp16精度的峰值为按元素字节折算的占位值，请改为实测或datasheet数值。"
        context = int(resolved.get("default_max_sequence_length") or 0)
        serving = config.get("serving", {})
        required = 2048 + 128 - 1
        if context >= required:
            serving["max_sequence_length"] = context
        return config

    def _config_to_flowchart(self, body: dict[str, Any]) -> dict[str, Any]:
        # 允许直接提交完整config，或包在 {"config": {...}} 里。
        config = body.get("config", body)
        return config_to_flowchart(_require_dict(config, "config"))

    def _flowchart_to_config(self, body: dict[str, Any]) -> dict[str, Any]:
        flowchart = _require_dict(body.get("flowchart"), "flowchart")
        # base_config 与 config 都接受：后者是前端左侧栏提交的基础配置字段名。
        raw_base = body.get("base_config", body.get("config", {}))
        base = _require_dict(raw_base, "base_config")
        config = flowchart_to_config(flowchart, base)
        # 顶层直接就是完整config，另附 config 别名，兼容两种前端取值写法。
        return {**config, "config": config}

    def _estimate(self, body: dict[str, Any]) -> dict[str, Any]:
        config = body.get("config", body)
        return estimate(Config.from_dict(_require_dict(config, "config")), details=True)

    # ---------------- 静态资源 ----------------

    def _serve_static(self, route: str) -> None:
        relative = "index.html" if route in {"", "/"} else route.lstrip("/")
        if relative.startswith("static/"):
            relative = relative[len("static/"):]
        static_root = STATIC_ROOT.resolve()
        candidate = (static_root / relative).resolve()
        try:
            candidate.relative_to(static_root)
        except ValueError:
            self._send_json({"error": "无效资源路径"}, HTTPStatus.BAD_REQUEST)
            return
        if not candidate.is_file():
            self._send_json({"error": "页面资源不存在"}, HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(candidate.name)
        base = content_type or "application/octet-stream"
        suffix = "" if base.startswith("image/") or base.startswith("font/") else "; charset=utf-8"
        self._send_bytes(candidate.read_bytes(), f"{base}{suffix}")

    def log_message(self, format: str, *args: object) -> None:
        # 只打印失败请求，保持终端安静。
        if len(args) > 1 and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动Transformer结构流程图编辑器")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机可访问")
    parser.add_argument("--port", type=int, default=8001, help="监听端口，默认 8001")
    parser.add_argument("--no-browser", action="store_true", help="启动时不自动打开浏览器")
    return parser


def start_local_server(host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start the visual API in a daemon thread and return its local URL."""

    server = ThreadingHTTPServer((host, port), FlowchartHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return server, f"http://{display_host}:{server.server_port}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("端口必须在 1 到 65535 之间")
    server = ThreadingHTTPServer((args.host, args.port), FlowchartHandler)
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{display_host}:{server.server_port}"
    print(f"Transformer 结构流程图编辑器已启动：{url}")
    print("按 Ctrl+C 关闭。")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n窗口已关闭。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
