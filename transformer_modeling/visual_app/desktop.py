"""Native desktop shell for the local visual modeling application."""

from __future__ import annotations

import argparse

from .server import start_local_server


WINDOW_TITLE = "Transformer Modeling"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 Transformer Modeling 桌面应用")
    parser.add_argument("--debug", action="store_true", help="启用 WebView 开发工具")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import webview
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "桌面界面依赖未安装。请运行: pip install -e .[desktop]"
        ) from exc

    server, url = start_local_server()
    try:
        webview.create_window(
            WINDOW_TITLE,
            url,
            width=1440,
            height=940,
            min_size=(1100, 720),
        )
        webview.start(debug=args.debug)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
