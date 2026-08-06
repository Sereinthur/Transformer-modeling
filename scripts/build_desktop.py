"""Build a Windows desktop executable with the bundled visual UI resources."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("当前打包脚本只支持 Windows。")
    separator = ";"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        "TransformerModeling",
        "--collect-all",
        "webview",
        "--add-data",
        f"{ROOT / 'transformer_modeling' / 'visual_app' / 'static'}{separator}transformer_modeling/visual_app/static",
        "--add-data",
        f"{ROOT / 'examples'}{separator}examples",
        "--distpath",
        str(DIST),
        "--workpath",
        str(ROOT / "build" / "pyinstaller"),
        "--specpath",
        str(ROOT / "build" / "pyinstaller"),
        str(ROOT / "scripts" / "desktop_entry.py"),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    print(f"桌面程序已生成：{DIST / 'TransformerModeling' / 'TransformerModeling.exe'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
