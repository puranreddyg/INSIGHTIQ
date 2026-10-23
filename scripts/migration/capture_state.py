"""Snapshot raw datasets + SQLite database into a JSON file.

    python scripts/migration/capture_state.py <label> [--db PATH]

Writes docs/migration/state_<label>.json. Used to record the real before/after
state that the migration report is generated from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "DataSets"
OUT_DIR = ROOT / "docs" / "migration"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_files() -> list[Path]:
    return sorted(p for p in DATASET_DIR.rglob("*.csv"))


def describe_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
    }


def describe_db(db_path: Path) -> dict:
    if not db_path.exists():
        return {"path": str(db_path), "exists": False}
    conn = sqlite3.connect(db_path)
    try:
        objects = conn.execute("SELECT name, type, tbl_name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name").fetchall()
        tables = {}
        for name, kind, _ in objects:
            if kind != "table":
                continue
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")').fetchall()]
            rows = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            tables[name] = {"rows": int(rows), "columns": len(cols), "column_names": cols}
        indexes = {name: tbl for name, kind, tbl in objects if kind == "index"}
        extra = {}
        if "supplier_profiles" in tables:
            extra["supplier_profiles_snapshot"] = pd.read_sql_query("SELECT * FROM supplier_profiles", conn).to_dict(orient="records")
        if "products" in tables and "urgent" in tables["products"]["column_names"]:
            extra["products_urgent"] = int(conn.execute("SELECT COUNT(*) FROM products WHERE urgent = 1").fetchone()[0])
            extra["products_urgent_discontinued"] = int(conn.execute("SELECT COUNT(*) FROM products WHERE urgent = 1 AND discontinued = 1").fetchone()[0])
        return {
            "path": db_path.relative_to(ROOT).as_posix() if db_path.is_relative_to(ROOT) else str(db_path),
            "exists": True,
            "sha256": sha256(db_path),
            "user_version": conn.execute("PRAGMA user_version").fetchone()[0],
            "tables": tables,
            "indexes": indexes,
            **extra,
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--db", default=str(ROOT / "data" / "insightiq.db"))
    args = parser.parse_args()

    state = {
        "label": args.label,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "csv_files": [describe_csv(p) for p in csv_files()],
        "database": describe_db(Path(args.db)),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"state_{args.label}.json"
    out.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}: {len(state['csv_files'])} CSVs, "
          f"{len(state['database'].get('tables', {}))} DB tables")


if __name__ == "__main__":
    main()
