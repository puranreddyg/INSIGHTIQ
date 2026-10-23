"""
Offline, reproducible training of the demand model.

    python -m backend.forecasting.train

Reads the REAL supply-chain CSV (not the database), runs the chronological
evaluation, refits on all days and writes backend/forecasting/artifacts/demand_model.json.
The result is deterministic: same CSV -> same artifact.
"""
from __future__ import annotations

import json

import pandas as pd

from backend import config
from backend.forecasting import model as demand_model
from backend.forecasting.demand import aggregate_daily


def load_wide() -> pd.DataFrame:
    raw = pd.read_csv(config.CSV_PATHS["supply_chain"], usecols=["Date", "SKU_ID", "Warehouse_ID", "Supplier_ID", "Units_Sold"])
    raw.columns = ["date", "sku_id", "warehouse_id", "supplier_id", "units_sold"]
    return aggregate_daily(raw)


def main() -> None:
    artifact = demand_model.build_artifact(load_wide())
    path = demand_model.save_artifact(artifact)
    print(f"[train] wrote {path}")
    print(json.dumps({"selection": artifact["selection"], "validation": artifact["validation"]}, indent=2))


if __name__ == "__main__":
    main()
