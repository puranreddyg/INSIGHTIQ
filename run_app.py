#!/usr/bin/env python
"""
InsightIQ — single-command startup.

Installs dependencies (if missing), ingests all 6 CSV datasets into a fresh
SQLite database, and launches the FastAPI web app on localhost.

Usage:
    python run_app.py [--port 8000] [--rebuild-db] [--no-reload]
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"


def _missing_packages() -> list[str]:
    required = ["fastapi", "uvicorn", "pandas", "numpy", "pydantic"]
    return [pkg for pkg in required if importlib.util.find_spec(pkg) is None]


def ensure_dependencies() -> None:
    missing = _missing_packages()
    if not missing:
        print("[setup] Dependencies already satisfied.")
        return
    print(f"[setup] Installing dependencies from {REQUIREMENTS.name} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch InsightIQ")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rebuild-db", action="store_true", help="Force re-ingest CSVs even if the DB already exists")
    parser.add_argument("--no-reload", action="store_true", help="Disable uvicorn auto-reload")
    args = parser.parse_args()

    ensure_dependencies()

    sys.path.insert(0, str(ROOT))
    from backend.database import build_database as build_db
    build_db(force=args.rebuild_db)

    import uvicorn

    print(f"\n[InsightIQ] Launching at http://{args.host}:{args.port}\n")
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        reload_dirs=[str(ROOT / "backend")],
    )


if __name__ == "__main__":
    main()
