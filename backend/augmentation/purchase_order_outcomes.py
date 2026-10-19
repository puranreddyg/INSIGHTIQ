"""
Purchase-order outcomes — the one dataset change InsightIQ needs.

The supply-chain ledger records 5,027 replenishments (rows with Order_Quantity > 0)
but every one arrives on time, in full and defect-free, so supplier performance
cannot be learned from it. This script adds ONE outcome row per existing
replenishment, without changing any original row:

  ledger row (date D, Order_Quantity Q)   = goods ACCEPTED into stock on day D
  promised lead time L                    = the lane's real Supplier_Lead_Time_Days
  actual lead time                        = L + delay
  order_date                              = D - actual lead time
  received_qty                            = Q + rejected_qty   (inspection rejects never reach stock)
  ordered_qty                             = received_qty + short_qty

So the ledger stays exactly consistent: stock rises by Q on day D, as it always did.

Each supplier has a HIDDEN performance profile (defect rate, on-time probability,
typical delay, short-shipment probability), with a small per-SKU variation. The
profiles are spread over ranges taken from real procurement data:
  defect rate     2.4% - 14.6%  real per-vendor defect rates in the KPI file
  on-time         60% - 97%     KPI file: 80.5% of finished POs fully delivered
  short shipment  1% - 10%      AdventureWorks: 4.3% of PO lines received short
The answer key is written to supplier_profiles_HIDDEN.json. The application never
reads it (a test enforces this); tests use it to check the model learns the truth.

    python -m backend.augmentation.purchase_order_outcomes          # (re)write files
    python -m backend.augmentation.purchase_order_outcomes --check  # verify committed CSV
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from backend import config

SEED = 20261020
OUT_DIR = config.DATASET_DIR / "augmented"
OUT_CSV = OUT_DIR / "purchase_order_outcomes.csv"
OUT_KEY = OUT_DIR / "supplier_profiles_HIDDEN.json"

COLUMNS = [
    "po_id", "sku_id", "warehouse_id", "supplier_id", "order_date", "promised_date", "received_date",
    "promised_lead_time_days", "actual_lead_time_days", "delay_days",
    "ordered_qty", "received_qty", "accepted_qty", "rejected_qty", "short_qty", "order_status",
]

RANGES = {
    "defect_rate": (0.024, 0.146),
    "on_time_prob": (0.60, 0.97),
    "mean_late_days": (1.5, 6.0),
    "short_prob": (0.01, 0.10),
}


def _num(identifier: str) -> int:
    return int(identifier.split("_")[1])


def supplier_profiles(suppliers: list[str], rng: np.random.Generator) -> dict[str, dict]:
    """Evenly spread each trait over its real range, then shuffle independently per
    trait, so every supplier differs and good/bad suppliers both exist."""
    n = len(suppliers)
    grid = (np.arange(n) + 0.5) / n
    traits = {}
    for name, (lo, hi) in RANGES.items():
        traits[name] = lo + (hi - lo) * rng.permutation(grid)
    return {s: {k: round(float(v[i]), 4) for k, v in traits.items()} for i, s in enumerate(suppliers)}


def generate(supply_chain: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    suppliers = sorted(supply_chain.Supplier_ID.unique(), key=_num)
    profiles = supplier_profiles(suppliers, rng)

    pairs = supply_chain[["SKU_ID", "Supplier_ID"]].drop_duplicates()
    pairs = pairs.assign(_s=pairs.SKU_ID.map(_num), _p=pairs.Supplier_ID.map(_num)).sort_values(["_s", "_p"])
    # per SKU-supplier variation around the supplier's own defect rate (log-normal, ~25%)
    pair_defect = {
        (r.SKU_ID, r.Supplier_ID): float(np.clip(profiles[r.Supplier_ID]["defect_rate"] * np.exp(rng.normal(0, 0.25)), 0.005, 0.25))
        for r in pairs.itertuples()
    }

    receipts = supply_chain[supply_chain.Order_Quantity > 0].copy()
    receipts["Date"] = pd.to_datetime(receipts["Date"])
    receipts = receipts.assign(_s=receipts.SKU_ID.map(_num), _w=receipts.Warehouse_ID.map(_num)) \
        .sort_values(["Date", "_s", "_w"]).reset_index(drop=True)

    rows = []
    for i, r in enumerate(receipts.itertuples()):
        prof = profiles[r.Supplier_ID]
        accepted = int(r.Order_Quantity)
        promised = int(r.Supplier_Lead_Time_Days)
        late = rng.random() > prof["on_time_prob"]
        delay = int(1 + rng.poisson(prof["mean_late_days"] - 1)) if late else 0
        p_def = pair_defect[(r.SKU_ID, r.Supplier_ID)]
        rejected = int(rng.binomial(int(round(accepted / (1 - p_def))), p_def))
        received = accepted + rejected
        short = int(np.ceil(received * rng.uniform(0.05, 0.25))) if rng.random() < prof["short_prob"] else 0
        received_date = r.Date
        order_date = received_date - pd.Timedelta(days=promised + delay)
        rows.append({
            "po_id": f"PO_{i + 1:05d}",
            "sku_id": r.SKU_ID, "warehouse_id": r.Warehouse_ID, "supplier_id": r.Supplier_ID,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "promised_date": (order_date + pd.Timedelta(days=promised)).strftime("%Y-%m-%d"),
            "received_date": received_date.strftime("%Y-%m-%d"),
            "promised_lead_time_days": promised, "actual_lead_time_days": promised + delay, "delay_days": delay,
            "ordered_qty": received + short, "received_qty": received, "accepted_qty": accepted,
            "rejected_qty": rejected, "short_qty": short,
            "order_status": "Partially Delivered" if short else "Delivered",
        })
    out = pd.DataFrame(rows, columns=COLUMNS)
    key = {
        "seed": SEED,
        "note": "ANSWER KEY. Never read by the application; tests only.",
        "ranges": RANGES,
        "suppliers": profiles,
        "pair_defect_rate": {f"{s}|{p}": round(v, 5) for (s, p), v in pair_defect.items()},
    }
    return out, key


def _digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    sc = pd.read_csv(config.CSV_PATHS["supply_chain"])
    out, key = generate(sc)
    if args.check:
        committed = pd.read_csv(OUT_CSV, dtype=str)
        same = committed.equals(out.astype(str))
        print("purchase_order_outcomes.csv matches generator:", same)
        return 0 if same else 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_KEY.write_text(json.dumps(key, indent=1))
    print(f"wrote {len(out)} rows -> {OUT_CSV} (sha256 {_digest(out)[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
