"""本地可视化入口：只负责网页与原有计算引擎之间的数据传递。"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from transformer_modeling import Config, estimate
from transformer_modeling.models import (
    preset_catalog as model_preset_catalog,
    resolve_model_definition,
)
from transformer_modeling.operators import get_operator_catalog


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PROJECT_ROOT / "static"
MAX_REQUEST_BYTES = 1_000_000


class VisualizationHandler(BaseHTTPRequestHandler):
    """提供静态页面，并将表单参数交给既有 estimate 函数。"""

    server_version = "TransformerModelingUI/0.1"

    def _send_bytes(
        self,
        payload: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, data: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = unquote(urlparse(self.path).path)
        if route == "/api/example":
            example_path = PROJECT_ROOT / "examples" / "single_chip_gqa.json"
            try:
                example = json.loads(example_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._send_json({"error": f"无法读取示例配置：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(example)
            return
        if route == "/api/model-presets":
            self._send_json(model_preset_catalog())
            return
        if route == "/api/operator-catalog":
            self._send_json(get_operator_catalog())
            return
        if route == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        relative = "index.html" if route == "/" else route.lstrip("/")
        candidate = (STATIC_ROOT / relative).resolve()
        try:
            candidate.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self._send_json({"error": "无效资源路径"}, HTTPStatus.BAD_REQUEST)
            return
        if not candidate.is_file():
            self._send_json({"error": "页面资源不存在"}, HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(candidate.name)
        self._send_bytes(
            candidate.read_bytes(),
            f"{content_type or 'application/octet-stream'}; charset=utf-8",
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = urlparse(self.path).path
        if route not in {"/api/estimate", "/api/model-definitions/resolve"}:
            self._send_json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "请求长度无效"}, HTTPStatus.BAD_REQUEST)
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json({"error": "请求内容为空或过大"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            raw = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(raw, dict):
                raise TypeError("请求JSON必须是对象")
            if route == "/api/model-definitions/resolve":
                preset_id = raw.get("preset_id")
                hf_config = raw.get("hf_config")
                if preset_id is not None and not isinstance(preset_id, str):
                    raise TypeError("preset_id必须是字符串")
                if hf_config is not None and not isinstance(hf_config, dict):
                    raise TypeError("hf_config必须是对象")
                self._send_json(resolve_model_definition(
                    preset_id=preset_id,
                    hf_config=hf_config,
                    scenario=str(raw.get("scenario", "base")),
                ))
                return
            config = Config.from_dict(raw)
            result = estimate(config, details=True)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(result)

    def log_message(self, format: str, *args: object) -> None:
        # 保持终端输出简洁，只记录失败请求。
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 Transformer 芯片性能评估可视化窗口")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机可访问")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    parser.add_argument("--no-browser", action="store_true", help="启动时不自动打开浏览器")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("端口必须在 1 到 65535 之间")
    server = ThreadingHTTPServer((args.host, args.port), VisualizationHandler)
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{display_host}:{server.server_port}"
    print(f"Transformer 性能评估窗口已启动：{url}")
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
