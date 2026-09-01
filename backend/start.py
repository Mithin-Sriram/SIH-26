"""One-command demo bootstrap for SIH26162.

Ensures the synthetic dataset and model artifacts exist, then starts the
FastAPI server. Safe to re-run: existing dataset/artifacts are reused.

Usage (from backend/):
    .venv\\Scripts\\python.exe start.py    (Windows PowerShell)
    .venv/bin/python start.py             (macOS/Linux)
    python start.py                       (with the venv activated)
"""

from __future__ import annotations

import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE, "app", "data", "training_data.csv")
ARTIFACTS_DIR = os.path.join(BASE, "app", "ml", "artifacts")
ARTIFACT_FILES = ("model.pkl", "calibrator.pkl", "features.json")


def main() -> None:
    py = sys.executable

    if not os.path.exists(DATA_CSV):
        print("[1/3] training_data.csv missing — generating synthetic "
              "dataset ...")
        subprocess.run(
            [py, "-m", "app.data.generate_synthetic_data"],
            cwd=BASE, check=True,
        )
    else:
        print("[1/3] training_data.csv found")

    missing = [
        f for f in ARTIFACT_FILES
        if not os.path.exists(os.path.join(ARTIFACTS_DIR, f))
    ]
    if missing:
        print(f"[2/3] model artifacts missing ({', '.join(missing)}) — "
              "training (≈2 min) ...")
        subprocess.run([py, "-m", "app.ml.train_model"], cwd=BASE, check=True)
    else:
        print("[2/3] model artifacts found")

    static_dir = os.path.join(BASE, "static")
    dashboard = " (dashboard + API)" if os.path.isdir(static_dir) else ""
    print(f"[3/3] starting FastAPI on http://localhost:8000{dashboard} ...")
    subprocess.run(
        [py, "-m", "uvicorn", "app.main:app", "--port", "8000"],
        cwd=BASE,
    )


if __name__ == "__main__":
    main()
