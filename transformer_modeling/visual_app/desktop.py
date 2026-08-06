"""Native desktop shell for the local visual modeling application."""

from __future__ import annotations

import argparse
from urllib.request import urlopen

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
        # Do not let the native window navigate before the local HTTP listener
        # has accepted a request; some WebView2 versions otherwise keep a blank
        # first navigation instead of retrying the connection.
        for path in ("", "/styles.css", "/js/app.js"):
            with urlopen(f"{url}{path}", timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"local visual server returned HTTP {response.status} for {path or '/'}")
        webview.create_window(
            WINDOW_TITLE,
            url,
            width=1600,
            height=940,
            min_size=(1500, 720),
        )
        # The application is designed and tested with the Windows WebView2
        # backend. Selecting it explicitly avoids pywebview falling back to a
        # legacy renderer that can show an empty document for ES modules.
        webview.start(gui="edgechromium", debug=args.debug)
    except Exception as exc:
        raise SystemExit(
            "Unable to open the desktop interface. Install Microsoft Edge WebView2 Runtime "
            f"and retry. Details: {exc}"
        ) from exc
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
