"""
Generator for the SIMULATED supplier-SKU profile (DataSets/simulated/).

The source data has no supplier-side availability and no supplier quality that
can be attached to SUP_1..SUP_10, so those three fields are simulated from real
distributions. Relationships are never invented: every (sku_id, supplier_id) row
is a pair that exists in supply_chain_dataset1.csv.

  available_qty    one draw from that pair's own historical Order_Quantity values
  defect_rate      supplier-level; Beta fitted (method of moments) to the 5 real
                   KPI supplier defect rates, capped at the highest real rate
  fulfillment_rate supplier-level; Binomial(n, p) / n with p = real pooled
                   fulfillment rate, n = median finished POs per KPI supplier

Each field has its own random stream spawned from one seed, so changing one
field's method never shifts another field's values.

    python -m backend.simulation.supplier_sku_profile          # (re)write files
    python -m backend.simulation.supplier_sku_profile --check  # verify committed CSV
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from backend import config

COLUMNS = ["sku_id", "supplier_id", "available_qty", "defect_rate", "fulfillment_rate"]
DECIMALS = 4
PERCENTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

OUT_CSV = config.CSV_PATHS["supplier_sku_profile"]
OUT_META = config.SIMULATED_DATASET_DIR / "supplier_sku_profile.meta.json"
OUT_PAIR_STATS = config.SIMULATED_DATASET_DIR / "evidence" / "order_quantity_by_pair.csv"


def _num(identifier: str) -> int:
    return int(identifier.split("_")[1])


def _stats(s: pd.Series) -> dict:
    d = {"n": int(len(s)), "min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()),
         "median": float(s.median()), "std": float(s.std(ddof=1))}
    d.update({f"p{int(p * 100)}": float(s.quantile(p)) for p in PERCENTILES})
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in d.items()}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(config.CSV_PATHS["supply_chain"]),
        pd.read_csv(config.CSV_PATHS["procurement_kpi"]),
    )


def valid_pairs(supply_chain: pd.DataFrame) -> pd.DataFrame:
    pairs = supply_chain[["SKU_ID", "Supplier_ID"]].drop_duplicates()
    pairs = pairs.assign(_s=pairs.SKU_ID.map(_num), _p=pairs.Supplier_ID.map(_num))
    return pairs.sort_values(["_s", "_p"]).drop(columns=["_s", "_p"]).reset_index(drop=True)


def order_quantity_history(supply_chain: pd.DataFrame) -> pd.Series:
    """Historical order sizes per (SKU, supplier), in source-file order."""
    orders = supply_chain[supply_chain.Order_Quantity > 0]
    return orders.groupby(["SKU_ID", "Supplier_ID"]).Order_Quantity.apply(np.array)


def kpi_defect_rates(kpi: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(per-PO defect rate, quantity-weighted rate per KPI supplier). POs with a
    blank Defective_Units are excluded — a blank is unknown, not zero."""
    k = kpi.dropna(subset=["Defective_Units"])
    per_po = k.Defective_Units / k.Quantity
    per_supplier = k.groupby("Supplier").apply(
        lambda g: g.Defective_Units.sum() / g.Quantity.sum(), include_groups=False
    )
    return per_po, per_supplier


def kpi_fulfillment(kpi: pd.DataFrame) -> tuple[float, pd.Series, pd.Series]:
    """Fulfillment = Delivered / (Delivered + Partially Delivered + Cancelled).
    Pending POs are excluded because their outcome is not yet known."""
    finished = kpi[kpi.Order_Status != "Pending"]
    delivered = finished.Order_Status == "Delivered"
    per_supplier = delivered.groupby(finished.Supplier).mean()
    return float(delivered.mean()), per_supplier, finished.groupby("Supplier").size()


def beta_method_of_moments(values: pd.Series) -> tuple[float, float, float, float]:
    m, v = float(values.mean()), float(values.var(ddof=1))
    common = m * (1 - m) / v - 1
    return m * common, (1 - m) * common, m, math.sqrt(v)


def generate(supply_chain: pd.DataFrame, kpi: pd.DataFrame, seed: int = config.SIMULATION_SEED) -> tuple[pd.DataFrame, dict]:
    pairs = valid_pairs(supply_chain)
    history = order_quantity_history(supply_chain)

    missing = [p for p in pairs.itertuples(index=False) if (p.SKU_ID, p.Supplier_ID) not in history.index]
    sparse = {f"{s}|{p}": len(h) for (s, p), h in history.items() if len(h) < config.MIN_ORDER_OBSERVATIONS_PER_PAIR}
    if missing or sparse:
        raise ValueError(f"insufficient order history: missing={missing} sparse={sparse}")

    per_po_defect, supplier_defect = kpi_defect_rates(kpi)
    alpha, beta, fit_mean, fit_sd = beta_method_of_moments(supplier_defect)
    cap_exact = float(supplier_defect.max())
    cap = math.floor(cap_exact * 10**DECIMALS) / 10**DECIMALS  # rounding up would exceed the real max

    p_pool, supplier_fulfillment, finished_counts = kpi_fulfillment(kpi)
    n_po = int(finished_counts.median())

    stream_avail, stream_defect, stream_fulfil = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )

    available = [int(stream_avail.choice(history[(p.SKU_ID, p.Supplier_ID)])) for p in pairs.itertuples(index=False)]

    suppliers = sorted(pairs.Supplier_ID.unique(), key=_num)
    defect_raw = np.round(stream_defect.beta(alpha, beta, len(suppliers)), DECIMALS)
    defect = np.minimum(defect_raw, cap)
    fulfillment = np.round(stream_fulfil.binomial(n_po, p_pool, len(suppliers)) / n_po, DECIMALS)

    quality = pd.DataFrame({"supplier_id": suppliers, "defect_rate": defect, "fulfillment_rate": fulfillment})
    profile = (
        pairs.rename(columns={"SKU_ID": "sku_id", "Supplier_ID": "supplier_id"})
        .assign(available_qty=available)
        .merge(quality, on="supplier_id", how="left")[COLUMNS]
    )

    meta = {
        "table": "supplier_sku_profile",
        "seed": seed,
        "random_streams": "numpy.random.SeedSequence(seed).spawn(3) -> [available_qty, defect_rate, fulfillment_rate]",
        "row_order": "sku number ascending, then supplier number ascending",
        "rows": int(len(profile)),
        "granularity": {
            "available_qty": "SKU-supplier",
            "defect_rate": "supplier (same value on every SKU row of a supplier)",
            "fulfillment_rate": "supplier (same value on every SKU row of a supplier)",
        },
        "relationships": {
            "valid_pairs_in_source": int(len(pairs)),
            "skus": int(pairs.SKU_ID.nunique()),
            "suppliers": int(pairs.Supplier_ID.nunique()),
            "suppliers_per_sku": {str(k): int(v) for k, v in pairs.groupby("SKU_ID").size().value_counts().sort_index().items()},
        },
        "available_qty": {
            "status": "SIMULATED",
            "source": "supply_chain_dataset1.csv Order_Quantity where Order_Quantity > 0",
            "method": "one value drawn uniformly at random (with replacement) from the pair's own historical Order_Quantity values",
            "historical_orders_total": int(sum(len(h) for h in history)),
            "min_observations_required": config.MIN_ORDER_OBSERVATIONS_PER_PAIR,
            "observations_per_pair": _stats(pd.Series([len(h) for h in history])),
            "order_quantity_all_pairs": _stats(pd.Series(np.concatenate(history.values))),
            "generated": _stats(profile.available_qty),
        },
        "defect_rate": {
            "status": "SIMULATED",
            "source": "Procurement KPI Analysis Dataset.csv: Defective_Units / Quantity; POs with blank Defective_Units excluded",
            "usable_pos": int(len(per_po_defect)),
            "excluded_pos_blank_defective_units": int(kpi.Defective_Units.isna().sum()),
            "per_po_defect_rate": _stats(per_po_defect),
            "real_supplier_rates_weighted": {k: round(float(v), 6) for k, v in supplier_defect.items()},
            "distribution": "Beta(alpha, beta) fitted by method of moments to the 5 real supplier-level weighted rates",
            "alpha": round(alpha, 6), "beta": round(beta, 6),
            "fit_mean": round(fit_mean, 6), "fit_sd": round(fit_sd, 6),
            "cap_exact_real_max": round(cap_exact, 8),
            "cap_applied": cap,
            "cap_rule": "min(value, floor(real max supplier rate to 4 dp))",
            "values_before_cap": dict(zip(suppliers, map(float, defect_raw))),
            "capped_suppliers": [s for s, raw in zip(suppliers, defect_raw) if raw > cap],
        },
        "fulfillment_rate": {
            "status": "SIMULATED",
            "not_on_time_rate": "the KPI dataset has no promised/expected delivery date, so on-time performance cannot be computed",
            "definition": "Delivered / (Delivered + Partially Delivered + Cancelled); Pending excluded",
            "source": "Procurement KPI Analysis Dataset.csv Order_Status",
            "real_pooled_rate": round(p_pool, 6),
            "real_supplier_rates": {k: round(float(v), 6) for k, v in supplier_fulfillment.items()},
            "finished_pos_per_real_supplier": {k: int(v) for k, v in finished_counts.items()},
            "method": "Binomial(n, p) / n per simulated supplier",
            "n": n_po, "p": round(p_pool, 6),
        },
    }
    return profile, meta


def pair_order_stats(supply_chain: pd.DataFrame) -> pd.DataFrame:
    history = order_quantity_history(supply_chain)
    rows = [{"sku_id": s, "supplier_id": p, **_stats(pd.Series(h))} for (s, p), h in history.items()]
    df = pd.DataFrame(rows)
    return df.assign(_s=df.sku_id.map(_num), _p=df.supplier_id.map(_num)).sort_values(["_s", "_p"]).drop(columns=["_s", "_p"])


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write() -> None:
    supply_chain, kpi = load_raw()
    profile, meta = generate(supply_chain, kpi)
    config.SIMULATED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PAIR_STATS.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(OUT_CSV, index=False, lineterminator="\n")
    pair_order_stats(supply_chain).to_csv(OUT_PAIR_STATS, index=False, lineterminator="\n")
    meta.update({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "backend/simulation/supplier_sku_profile.py",
        "command": "python -m backend.simulation.supplier_sku_profile",
        "inputs_sha256": {
            config.CSV_PATHS[k].relative_to(config.ROOT_DIR).as_posix(): _sha256(config.CSV_PATHS[k])
            for k in ("supply_chain", "procurement_kpi")
        },
        "output_sha256": _sha256(OUT_CSV),
    })
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(profile)} rows -> {OUT_CSV.relative_to(config.ROOT_DIR)}")


def check() -> bool:
    profile, _ = generate(*load_raw())
    committed = pd.read_csv(OUT_CSV)
    same = committed.equals(profile.reset_index(drop=True))
    print("committed CSV matches regeneration" if same else "MISMATCH between committed CSV and regeneration")
    return same


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    write()
