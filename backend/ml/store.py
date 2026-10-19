"""Load trained artefacts saved by the database build (table `ml_models`)."""
from __future__ import annotations

import json
import sqlite3

from backend.ml.demand_forecaster import DemandModel


def load_payload(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute("SELECT payload FROM ml_models WHERE name = ?", (name,)).fetchone()
    return row[0] if row else None


def load_demand_model(conn: sqlite3.Connection) -> DemandModel:
    payload = load_payload(conn, "demand_forecaster")
    if payload is None:
        raise RuntimeError("demand forecaster not trained — rebuild the database")
    return DemandModel.from_json(payload)


def load_supplier_meta(conn: sqlite3.Connection) -> dict:
    payload = load_payload(conn, "supplier_estimator")
    return json.loads(payload) if payload else {}
