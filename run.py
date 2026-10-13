"""Start the dashboard with its complete Python environment: python run.py."""

from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CANDIDATES = [
    ROOT / ".venv/bin/python",
    ROOT.parents[1] / "work/venv/bin/python",
    Path(sys.executable),
]
CHECK = "import streamlit, plotly, pandas, numpy, sklearn, yaml, reportlab, matplotlib, jinja2"


def main():
    for interpreter in CANDIDATES:
        if not interpreter.exists():
            continue
        result = subprocess.run(
            [str(interpreter), "-c", CHECK],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            if "--check" in sys.argv:
                print(f"Ready: {interpreter}")
                return
            os.chdir(ROOT)
            os.execv(
                str(interpreter),
                [
                    str(interpreter),
                    "-m",
                    "streamlit",
                    "run",
                    "app/streamlit_app.py",
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    "8509",
                    *sys.argv[1:],
                ],
            )
    raise SystemExit(
        "Install the project environment first: python -m venv .venv && .venv/bin/python -m pip install -r requirements-app.txt"
    )


if __name__ == "__main__":
    main()
