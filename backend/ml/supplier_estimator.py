"""
Supplier estimator — replaces the old randomly generated supplier profile.

Nothing here draws a random number. Each supplier-SKU field is an estimate with
a stated evidence base:

available_qty      DEMONSTRATED CAPACITY: the largest single order this supplier has
                   actually delivered for this SKU in the ledger. It is a floor on
                   what the supplier can ship (the buyer never ordered more), not a
                   live stock level; pairs with fewer than
                   MIN_ORDER_OBSERVATIONS_PER_PAIR orders get no estimate and are
                   left out of scoring rather than guessed.

defect_rate        Empirical-Bayes (Buhlmann credibility) estimate per supplier:
                       estimate = Z * (supplier's own PO defect rate) + (1 - Z) * prior
                   prior = mean PO defect rate of the real KPI vendors, Z = n/(n + k)
                   with k = within-vendor variance / between-vendor variance, both
                   measured on the real KPI purchase orders. With no evidence for a
                   supplier Z = 0 and the supplier sits exactly on the prior.

fulfillment_rate   Same shape with a Beta-Binomial update: prior = pooled share of
                   finished KPI POs that were fully delivered, prior strength = the
                   median number of finished POs per real vendor.

Where does supplier-specific evidence come from? The KPI file names five vendors
(Alpha_Inc ...) and shares no key with SUP_1..SUP_10, so by default no supplier has
any and every supplier gets the prior. Real evidence can be supplied in
DataSets/supplier_evidence/ (see its README):
    vendor_links.json     {"SUP_3": "Delta_Logistics"}  — only if you KNOW they are the same company
    supplier_outcomes.csv real delivery outcomes recorded per supplier
Either one switches that supplier from "prior" to "evidence" with no code change.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backend import config

EVIDENCE_DIR = config.DATASET_DIR / "supplier_evidence"
VENDOR_LINKS_PATH = EVIDENCE_DIR / "vendor_links.json"
OUTCOMES_PATH = EVIDENCE_DIR / "supplier_outcomes.csv"
OUTCOME_COLUMNS = ["supplier_id", "quantity", "defective_units", "order_status"]
FINISHED_STATUSES = {"Delivered", "Partially Delivered", "Cancelled"}

PROFILE_COLUMNS = [
    "sku_id", "supplier_id", "available_qty", "defect_rate", "fulfillment_rate",
    "fill_rate", "on_time_prob", "expected_delay_days", "delay_sd",
    "evidence_pos", "capacity_orders", "evidence_source",
]


def _num(identifier: str) -> int:
    return int(identifier.split("_")[1])


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def load_evidence(kpi: pd.DataFrame, supplier_ids: set[str], links_path: Path = VENDOR_LINKS_PATH,
                  outcomes_path: Path = OUTCOMES_PATH) -> pd.DataFrame:
    """PO outcomes attributable to a specific SUP_x: columns supplier_id, quantity,
    defective_units (NaN = unknown), order_status, source. Empty unless the user
    supplied a link or an outcomes file. Invalid input fails loudly."""
    parts = []
    if links_path.exists():
        links = json.loads(links_path.read_text(encoding="utf-8") or "{}")
        vendors = set(kpi["supplier"].unique())
        for sup, vendor in links.items():
            if sup not in supplier_ids:
                raise ValueError(f"vendor_links.json: {sup!r} is not a supplier in the supply-chain data")
            if vendor not in vendors:
                raise ValueError(f"vendor_links.json: {vendor!r} is not a vendor in the KPI file ({sorted(vendors)})")
        if len(set(links.values())) != len(links):
            raise ValueError("vendor_links.json: one KPI vendor cannot be linked to two suppliers")
        for sup, vendor in links.items():
            rows = kpi[kpi.supplier == vendor]
            parts.append(pd.DataFrame({
                "supplier_id": sup, "quantity": rows.quantity.to_numpy(float),
                "defective_units": rows.defective_units.to_numpy(float),
                "order_status": rows.order_status.to_numpy(), "source": f"KPI vendor {vendor}",
            }))
    if outcomes_path.exists():
        out = pd.read_csv(outcomes_path)
        if out.empty:
            out = pd.DataFrame(columns=OUTCOME_COLUMNS)
        missing = [c for c in OUTCOME_COLUMNS if c not in out.columns]
        if missing:
            raise ValueError(f"supplier_outcomes.csv is missing columns {missing}")
        unknown = sorted(set(out.supplier_id) - supplier_ids)
        if unknown:
            raise ValueError(f"supplier_outcomes.csv has unknown suppliers {unknown}")
        if (out.quantity <= 0).any() or (out.defective_units > out.quantity).any() or (out.defective_units < 0).any():
            raise ValueError("supplier_outcomes.csv has impossible quantities")
        bad = sorted(set(out.order_status.dropna()) - FINISHED_STATUSES - {"Pending"})
        if bad:
            raise ValueError(f"supplier_outcomes.csv has unknown order_status values {bad}")
        parts.append(out.assign(source="supplier_outcomes.csv")[OUTCOME_COLUMNS + ["source"]])
    cols = OUTCOME_COLUMNS + ["source"]
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Priors measured on the real KPI purchase orders
# ---------------------------------------------------------------------------

def fit_priors(kpi: pd.DataFrame) -> dict:
    k = kpi.dropna(subset=["defective_units"]).copy()
    k["rate"] = (k.defective_units / k.quantity).clip(0, 1)
    vendor = k.groupby("supplier").rate.agg(["mean", "count", "var"])
    mu0 = float(k.rate.mean())
    within = float((vendor["var"] * (vendor["count"] - 1)).sum() / (vendor["count"].sum() - len(vendor)))
    between = float(max(vendor["mean"].var(ddof=1) - (within / vendor["count"]).mean(), 1e-9))
    kappa_defect = within / between  # PO-equivalents of prior information

    finished = kpi[kpi.order_status.isin(FINISHED_STATUSES)]
    delivered = (finished.order_status == "Delivered")
    p0 = float(delivered.mean())
    n0 = float(finished.groupby("supplier").size().median())
    return {
        "defect_prior_mean": mu0, "defect_within_var": within, "defect_between_var": between,
        "defect_kappa_pos": kappa_defect, "defect_prior_sd": math.sqrt(between),
        "defect_vendor_rates": {s: round(float(r), 6) for s, r in vendor["mean"].items()},
        "fulfillment_prior_mean": p0, "fulfillment_kappa_pos": n0,
        "kpi_pos_with_defect_data": int(len(k)), "kpi_finished_pos": int(len(finished)), "kpi_vendors": int(len(vendor)),
    }


def supplier_estimates(priors: dict, evidence: pd.DataFrame, supplier_ids: list[str]) -> pd.DataFrame:
    rows = []
    for s in supplier_ids:
        e = evidence[evidence.supplier_id == s]
        known = e.dropna(subset=["defective_units"])
        n_def = len(known)
        z_def = n_def / (n_def + priors["defect_kappa_pos"]) if n_def else 0.0
        own = float((known.defective_units / known.quantity).clip(0, 1).mean()) if n_def else priors["defect_prior_mean"]
        defect = z_def * own + (1 - z_def) * priors["defect_prior_mean"]
        defect_sd = math.sqrt((1 - z_def) * priors["defect_between_var"])

        fin = e[e.order_status.isin(FINISHED_STATUSES)]
        n_fin = len(fin)
        delivered = int((fin.order_status == "Delivered").sum())
        k0 = priors["fulfillment_kappa_pos"]
        fulfil = (priors["fulfillment_prior_mean"] * k0 + delivered) / (k0 + n_fin)
        rows.append({
            "supplier_id": s, "defect_rate": defect, "defect_sd": defect_sd, "fulfillment_rate": fulfil,
            "evidence_pos": int(max(n_def, n_fin)),
            "evidence_source": ", ".join(sorted(e.source.unique())) if len(e) else "prior (no supplier-specific evidence)",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

def demonstrated_capacity(supply_chain: pd.DataFrame) -> pd.DataFrame:
    orders = supply_chain[supply_chain.order_quantity > 0]
    cap = orders.groupby(["sku_id", "supplier_id"]).order_quantity.agg(["max", "size"]).reset_index()
    cap.columns = ["sku_id", "supplier_id", "available_qty", "capacity_orders"]
    pairs = supply_chain[["sku_id", "supplier_id"]].drop_duplicates()
    cap = pairs.merge(cap, how="left", on=["sku_id", "supplier_id"])
    cap["capacity_orders"] = cap.capacity_orders.fillna(0).astype(int)
    thin = cap.capacity_orders < config.MIN_ORDER_OBSERVATIONS_PER_PAIR
    cap.loc[thin, "available_qty"] = np.nan  # no estimate rather than a guess
    return cap


def learned_profile(supply_chain: pd.DataFrame, po_outcomes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Profile from the supplier performance model trained on purchase-order outcomes.
    Capacity = largest quantity the supplier actually delivered for the SKU."""
    from backend.ml import supplier_model

    model = supplier_model.fit(po_outcomes)
    pairs = supply_chain[["sku_id", "supplier_id"]].drop_duplicates()
    cap = po_outcomes.groupby(["sku_id", "supplier_id"]).agg(
        available_qty=("received_qty", "max"), capacity_orders=("po_id", "size")).reset_index()
    prof = pairs.merge(cap, how="left").merge(model["pairs"], how="left", on=["sku_id", "supplier_id"])
    prof["capacity_orders"] = prof.capacity_orders.fillna(0).astype(int)
    prof.loc[prof.capacity_orders < config.MIN_ORDER_OBSERVATIONS_PER_PAIR, "available_qty"] = np.nan
    prof = prof.rename(columns={"expected_delay": "expected_delay_days"})
    prof["fulfillment_rate"] = prof.otif_prob
    prof["evidence_pos"] = prof.orders.fillna(0).astype(int)
    prof["evidence_source"] = "purchase_order_outcomes (learned)"
    prof = prof.assign(_s=prof.sku_id.map(_num), _p=prof.supplier_id.map(_num)) \
        .sort_values(["_s", "_p"]).drop(columns=["_s", "_p"]).reset_index(drop=True)
    prof["available_qty"] = prof.available_qty.astype("Int64")
    meta = {
        "source": "purchase_order_outcomes",
        "params": model["params"],
        "backtest": supplier_model.backtest(po_outcomes),
        "suppliers": model["suppliers"].set_index("supplier_id")[
            ["orders", "defect_rate", "on_time_prob", "short_prob", "fill_rate", "expected_delay", "delay_sd"]
        ].round(4).to_dict(orient="index"),
        "capacity_basis": "largest quantity delivered per SKU-supplier pair (purchase_order_outcomes)",
        "min_orders_per_pair": config.MIN_ORDER_OBSERVATIONS_PER_PAIR,
    }
    return prof[PROFILE_COLUMNS], meta


def build_profile(supply_chain: pd.DataFrame, kpi: pd.DataFrame,
                  links_path: Path = VENDOR_LINKS_PATH, outcomes_path: Path = OUTCOMES_PATH,
                  po_outcomes: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Learned profile when purchase-order outcomes exist; otherwise the prior-based
    estimate (real KPI baseline, updated by any evidence in DataSets/supplier_evidence/)."""
    if po_outcomes is not None and not po_outcomes.empty:
        return learned_profile(supply_chain, po_outcomes)
    suppliers = sorted(supply_chain.supplier_id.unique(), key=_num)
    priors = fit_priors(kpi)
    evidence = load_evidence(kpi, set(suppliers), links_path, outcomes_path)
    per_supplier = supplier_estimates(priors, evidence, suppliers)
    profile = demonstrated_capacity(supply_chain).merge(per_supplier, on="supplier_id", how="left")
    profile = profile.assign(_s=profile.sku_id.map(_num), _p=profile.supplier_id.map(_num)) \
        .sort_values(["_s", "_p"]).drop(columns=["_s", "_p"]).reset_index(drop=True)
    profile["available_qty"] = profile.available_qty.astype("Int64")
    profile["fill_rate"] = 1.0
    profile["on_time_prob"] = profile.fulfillment_rate
    profile["expected_delay_days"] = 0.0
    profile["delay_sd"] = 0.0
    meta = {
        "priors": priors,
        "suppliers": per_supplier.set_index("supplier_id").round(6).to_dict(orient="index"),
        "evidence_rows": int(len(evidence)),
        "capacity_basis": "largest order delivered per SKU-supplier pair in the supply_chain ledger",
        "min_orders_per_pair": config.MIN_ORDER_OBSERVATIONS_PER_PAIR,
    }
    return profile[PROFILE_COLUMNS], meta
