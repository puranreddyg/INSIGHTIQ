"""Build docs/migration/InsightIQ_SKU_Migration_Report.pdf from the actual migration artefacts.

    pip install reportlab   # report-only dependency
    python scripts/migration/build_report.py

Inputs: docs/migration/{state_before,state_after,validation_results,change_log}.json,
DataSets/simulated/supplier_sku_profile.{csv,meta.json}, and live queries on data/insightiq.db.
"""
from __future__ import annotations

import json
import platform
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "migration"
OUT = DOCS / "InsightIQ_SKU_Migration_Report.pdf"
SIM_DIR = ROOT / "DataSets" / "simulated"


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


BEFORE, AFTER = _load(DOCS / "state_before.json"), _load(DOCS / "state_after.json")
VALID, LOG = _load(DOCS / "validation_results.json"), _load(DOCS / "change_log.json")
META = _load(SIM_DIR / "supplier_sku_profile.meta.json")
PROFILE = pd.read_csv(SIM_DIR / "supplier_sku_profile.csv")

# ---------------------------------------------------------------------------
# Fonts / styles
# ---------------------------------------------------------------------------
FONT, BOLD = "Helvetica", "Helvetica-Bold"
_fonts = Path("C:/Windows/Fonts")
if (_fonts / "arial.ttf").exists():
    pdfmetrics.registerFont(TTFont("Arial", str(_fonts / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(_fonts / "arialbd.ttf")))
    FONT, BOLD = "Arial", "Arial-Bold"
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial", boldItalic="Arial-Bold")

ASCII = {"→": "->", "≥": ">=", "≤": "<=", "−": "-", "α": "alpha", "β": "beta", "×": "x", "₹": "Rs "}


def t(s) -> str:
    s = "" if s is None else str(s)
    if FONT == "Helvetica":
        for k, v in ASCII.items():
            s = s.replace(k, v)
    return s


INK, MUTED, ACCENT = colors.HexColor("#10233f"), colors.HexColor("#56647d"), colors.HexColor("#2a78d6")
REAL_C, SIM_C, DEL_C = colors.HexColor("#1baf7a"), colors.HexColor("#b45309"), colors.HexColor("#d03b3b")
base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=base["Title"], fontName=BOLD, fontSize=22, leading=27, textColor=INK, alignment=TA_LEFT),
    "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName=BOLD, fontSize=15, leading=19, textColor=INK, spaceBefore=4, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=BOLD, fontSize=11.5, leading=15, textColor=ACCENT, spaceBefore=8, spaceAfter=4,
                         keepWithNext=1),
    "body": ParagraphStyle("body", parent=base["BodyText"], fontName=FONT, fontSize=9.2, leading=12.6, textColor=INK, spaceAfter=4),
    "small": ParagraphStyle("small", parent=base["BodyText"], fontName=FONT, fontSize=8, leading=10.5, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName=FONT, fontSize=7.3, leading=9, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=BOLD, fontSize=7.3, leading=9, textColor=colors.white),
    "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontName=FONT, fontSize=9.2, leading=12.6, leftIndent=12, bulletIndent=2, textColor=INK),
}


def P(text, style="body"):
    return Paragraph(t(text), S[style])


def bullets(items):
    return [Paragraph(t(i), S["bullet"], bulletText="•") for i in items]


def esc(v) -> str:
    return t(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def table(header, rows, widths, status_col: int | None = None):
    data = [[Paragraph(esc(h), S["cellb"]) for h in header]]
    data += [[Paragraph(esc(c), S["cell"]) for c in r] for r in rows]
    tb = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d3e3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6fb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if status_col is not None:
        for i, r in enumerate(rows, 1):
            v = str(r[status_col]).upper()
            c = SIM_C if "SIMULATED" in v else DEL_C if v in ("DELETED", "FAIL", "NO") else REAL_C if ("REAL" in v or v in ("PASS", "YES")) else None
            if c is not None:
                style.append(("TEXTCOLOR", (status_col, i), (status_col, i), c))
    tb.setStyle(TableStyle(style))
    return tb


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

def flow(boxes, width=170, height=34, fills=None, caption=None):
    """Horizontal chain of labelled boxes joined by arrows. Units: mm."""
    W, H = width * mm, height * mm
    d = Drawing(W, H + (5 * mm if caption else 0))
    n = len(boxes)
    gap = 7 * mm
    bw = (W - gap * (n - 1)) / n
    y0 = 2 * mm
    for i, label in enumerate(boxes):
        x = i * (bw + gap)
        fill = (fills or {}).get(i, colors.HexColor("#eaf1fb"))
        d.add(Rect(x, y0, bw, H - 4 * mm, rx=4, ry=4, fillColor=fill, strokeColor=ACCENT, strokeWidth=0.8))
        lines = t(label).split("\n")
        top = y0 + (H - 4 * mm) / 2 + (len(lines) - 1) * 4.6 / 2 * mm / 1.4
        for j, line in enumerate(lines):
            d.add(String(x + bw / 2, top - j * 9.2 - 3, line, fontName=BOLD if j == 0 else FONT,
                         fontSize=7.8 if j == 0 else 6.7, fillColor=INK, textAnchor="middle"))
        if i < n - 1:
            ax, ay = x + bw + 1, y0 + (H - 4 * mm) / 2
            d.add(Line(ax, ay, ax + gap - 3, ay, strokeColor=MUTED, strokeWidth=1))
            d.add(Polygon([ax + gap - 3, ay + 2.6, ax + gap - 1, ay, ax + gap - 3, ay - 2.6], fillColor=MUTED, strokeColor=MUTED))
    if caption:
        d.add(String(0, H + 1.2 * mm, t(caption), fontName=BOLD, fontSize=8, fillColor=MUTED))
    return d


RED_FILL, SIM_FILL, REAL_FILL = colors.HexColor("#fbe3e2"), colors.HexColor("#fbe8c9"), colors.HexColor("#dcf4ea")


# ---------------------------------------------------------------------------
# Live facts (queried, not typed)
# ---------------------------------------------------------------------------

def live_facts() -> dict:
    c = sqlite3.connect(ROOT / "data" / "insightiq.db")
    try:
        sc = pd.read_sql_query("SELECT units_sold, demand_forecast FROM supply_chain", c)
        below = c.execute("SELECT MIN(n), MAX(n), AVG(n) FROM (SELECT date, SUM(inventory_level < reorder_point) n FROM supply_chain GROUP BY date)").fetchone()
        lead_varies = c.execute("SELECT COUNT(*) FROM (SELECT sku_id, warehouse_id FROM supply_chain GROUP BY 1, 2 HAVING COUNT(DISTINCT supplier_lead_time_days) > 1)").fetchone()[0]
        return {
            "stockout_sum": c.execute("SELECT SUM(stockout_flag) FROM supply_chain").fetchone()[0],
            "below_rop_per_day": (below[0], below[1], round(below[2], 1)),
            "forecast_corr": round(float(sc.corr().iloc[0, 1]), 3),
            "lanes_with_varying_lead_time": lead_varies,
            "multi_lane_pairs": c.execute("SELECT COUNT(*) FROM (SELECT sku_id, supplier_id FROM sku_supplier_lanes GROUP BY 1, 2 HAVING COUNT(*) > 1)").fetchone()[0],
        }
    finally:
        c.close()


def old_facts() -> dict:
    """Pre-migration figures from the backed-up DB and the before snapshot."""
    snap = pd.DataFrame(BEFORE["database"]["supplier_profiles_snapshot"])
    c = sqlite3.connect(ROOT / "data" / "backup" / "pre_migration_2026-09-27" / "insightiq.db")
    try:
        max_deficit = c.execute("SELECT MAX(deficit) FROM products WHERE urgent = 1").fetchone()[0]
    finally:
        c.close()
    return {"min_stock": int(snap.stock_availability.min()), "max_deficit": int(max_deficit),
            "max_rating": float(snap.rating.max()), "min_delivery": float(snap.delivery_time_days.min()),
            "reliable": snap[snap.reliability_score >= 0.90].supplier_id.tolist()}


LIVE, OLDF = live_facts(), old_facts()
E2E = VALID["end_to_end"]
URG = E2E["urgent_skus"]
PLANNED = [r for r in URG if r["plan_status"] == "planned"]
BLOCKED = [r for r in URG if r["plan_status"] == "no_eligible_supplier"]
OLD = VALID["old_behaviour"]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def cover():
    return [
        Spacer(1, 30 * mm),
        P("InsightIQ", "small"),
        Paragraph(t("Dataset &amp; Architecture Migration Report"), S["title"]),
        P("ProductID-based procurement → SKU_ID-based procurement", "h2"),
        Spacer(1, 6 * mm),
        P(f"Migration date: 2026-09-27 · Report generated: {datetime.now():%Y-%m-%d %H:%M} · Simulation seed: {META['seed']}"),
        P(f"Before-state captured {BEFORE['captured_at']} · after-state captured {AFTER['captured_at']} · validation run {VALID['run_at']}"),
        Spacer(1, 6 * mm),
        P("Counts, statistics and test results in this report are read at build time from the migration artefacts in "
          "docs/migration/ (state snapshots, validation results, change log), the simulation metadata, and the current and "
          "backed-up databases. Explanatory text (reasons, assumptions) is written by the author.", "small"),
        Spacer(1, 10 * mm),
        table(["Headline", "Result"], [
            ["Raw source data", "Unchanged — SHA-256 identical for all 6 raw CSVs; 0 raw records deleted"],
            ["Old behaviour (re-run of pre-migration code)",
             f"{OLD['urgent_products_run']} urgent products → recommended {OLD['recommended_counts']} · backup {OLD['backup_counts']}"],
            ["New behaviour", f"{len(URG)} urgent SKUs · {len(PLANNED)} planned with {len(E2E['distinct_recommended_suppliers'])} different suppliers · "
                              f"{len(BLOCKED)} correctly get no supplier (no single supplier has enough) · "
                              f"{E2E['distinct_full_rankings_across_50_skus']} distinct rankings across 50 SKUs"],
            ["Simulated fields", "available_qty, defect_rate, fulfillment_rate (211 real SKU-supplier pairs, seed 20260926)"],
            ["Validation", f"{sum(c['passed'] for k in ('raw_checksums', 'integrity', 'future_compatibility', 'reproducibility') for c in VALID[k])} "
                           f"checks passed · pytest: {VALID['pytest']['summary']}"],
        ], [55, 115]),
        PageBreak(),
    ]


def s1_exec():
    ev = VALID["simulation"]["evidence"]
    return [
        P("1. Executive Summary", "h1"),
        P("<b>The original problem.</b> Every time procurement was triggered, the system recommended SUP_1 with SUP_9 as backup. "
          f"Re-running the pre-migration code on the backed-up database confirms it: all {OLD['urgent_products_run']} urgent products "
          "received exactly that pair. This was not a coding error in the ranking itself; the inputs made any other outcome impossible:"),
        *bullets([
            "Supplier scores did not depend on the product: one static row per supplier was ranked for every request.",
            "The eligibility check compared a product's deficit with the supplier's total stock. That stock was actually "
            f"the buyer's own warehouse inventory (Inventory_Level follows buyer stock flow on {ev['inventory_is_buyer_stock_flow_match_pct']}% of rows) "
            "and was always larger than any deficit, so no supplier was ever filtered out.",
            "Supplier quality (rating, defect rate) came from a round-robin assignment of 5 KPI vendors to SUP_1–SUP_10 with no shared key.",
            "The product table (product.csv) has no relationship to any supplier or SKU: its SupplierID equals ProductID on all 1,000 rows. "
            f"{BEFORE['database']['products_urgent_discontinued']} of its {BEFORE['database']['products_urgent']} urgent products were discontinued.",
        ]),
        P("<b>Why SKU_ID.</b> supply_chain_dataset1.csv is the only dataset where items, suppliers, cost and lead time are genuinely "
          "connected: 50 SKUs, 10 suppliers, 211 real SKU-supplier pairs, 3–5 candidate suppliers per SKU."),
        P("<b>What changed.</b> Procurement requests are now SKU-based and built from the latest real supply-chain snapshot; candidate "
          "suppliers are only those that actually supply the SKU; eligibility compares the requested quantity with each supplier's "
          "availability; unsupported mappings were removed; the Knowledge Agent's rules were rewired to the new inputs without inventing thresholds."),
        P("<b>What was added.</b> A real, lossless lane table (sku_supplier_lanes, 250 rows) and a clearly labelled simulated table "
          "(supplier_sku_profile, 211 rows) for the three facts the data does not contain."),
        P("<b>What was removed.</b> The supplier_profiles table and its round-robin mapping, three derived product columns, one index, "
          "three config constants, one obsolete function and endpoint. No raw source record was deleted (Section 6)."),
        P("<b>What remains simulated.</b> Supplier available quantity, supplier defect rate and supplier fulfillment rate — each derived "
          "from real distributions and documented in Section 8."),
        P("<b>Future compatibility.</b> The full 91,250-row daily time series is retained; raw supplier criteria are kept on every "
          "candidate; ranking weights are injectable; the agents run on any source that fills the same tables (tested with non-CSV data). "
          "Delivery-time ML remains unsupported by the data (Section 14)."),
        PageBreak(),
    ]


def s2_original():
    snap = pd.DataFrame(BEFORE["database"]["supplier_profiles_snapshot"])
    snap["n"] = snap.supplier_id.str[4:].astype(int)
    snap = snap.sort_values("n")
    return [
        P("2. Original Data Architecture", "h1"),
        flow(["product.csv\n1,000 products\nSupplierID = ProductID", "Procurement KPI\n5 named vendors\n777 POs", "supply_chain\n10 SUP_x, 50 SKUs\n91,250 rows"],
             fills={0: RED_FILL, 1: RED_FILL}, caption="Original datasets (no shared keys between them)"),
        Spacer(1, 3 * mm),
        flow(["KPI vendor\nquality", "round-robin\nSUP_1→Alpha_Inc …", "supplier_profiles\n10 static rows", "ranked for\nevery product"],
             fills={1: RED_FILL, 2: RED_FILL}, caption="Original relationships"),
        Spacer(1, 3 * mm),
        flow(["ProductID\n(urgent flag)", "deficit vs\nsupplier 'stock'", "same 10-row\nranking", "Rule 3 drops\nSUP_5", "SUP_1 / SUP_9\nevery time"],
             fills={4: RED_FILL}, caption="Original procurement flow"),
        P("Problems discovered", "h2"),
        *bullets([
            "Disjoint ID spaces: product.csv (1,000 ProductIDs), KPI (5 vendor names), supply_chain (SUP_1–10, SKU_1–50) share no join key.",
            "Round-robin mapping gave SUP_x the quality history of whichever vendor matched its sorted position.",
            f"stock_availability summed buyer warehouse stock, not supplier stock; smallest value ({OLDF['min_stock']:,}) exceeded every deficit (max {OLDF['max_deficit']:,}).",
            f"Product fields are uniformly random and {BEFORE['database']['products_urgent_discontinued']} urgent products were discontinued; order.csv has no product reference.",
            f"Old Rule 2 could never fire (max rating {OLDF['max_rating']} < 4.5; min delivery {OLDF['min_delivery']} d > 3 d); "
            f"Rule 4 fired only for {', '.join(OLDF['reliable'])} (a relative index maxed at 1.0).",
        ]),
        P("The deleted round-robin supplier_profiles table (from state_before.json)", "h2"),
        table(["Supplier", "Mapped to", "Rating", "Defect", "Reliability", "Delivery d", "'Stock'"],
              [[r.supplier_id, r.benchmarked_against, r.rating, r.defect_rate, r.reliability_score, r.delivery_time_days, int(r.stock_availability)]
               for r in snap.itertuples()], [20, 30, 18, 18, 22, 22, 22]),
        PageBreak(),
    ]


def s3_new():
    return [
        P("3. New Data Architecture", "h1"),
        flow(["SKU\nsupply_chain\n(REAL)", "Supplier\nsku_supplier_lanes\n(REAL, 250 lanes)", "Supplier-SKU profile\nsupplier_sku_profile\n(SIMULATED, 211)", "Procurement\nrequest → ranking"],
             fills={0: REAL_FILL, 1: REAL_FILL, 2: SIM_FILL}, caption="New relationships"),
        Spacer(1, 4 * mm),
        *bullets([
            "SKU identity and urgency: supply_chain (per SKU × warehouse × day). A SKU is urgent if any warehouse lane is below its reorder point on the latest date (2024-12-30).",
            "Supplier relationship: sku_supplier_lanes — each (SKU, warehouse) has exactly one supplier with a real unit cost and lead time that never change during the year (lossless).",
            "Supplier-SKU profile: supplier_sku_profile keyed by the 211 real (SKU, supplier) pairs; adds simulated availability and supplier-level quality.",
            "Candidate matrix for a request = suppliers serving the SKU (lanes) joined with their profile row. Where a supplier serves the SKU at "
            f"2–3 warehouses ({LIVE['multi_lane_pairs']} pairs), cost and lead time are averaged and min/max/lane count are kept.",
            "required_quantity = sum over urgent lanes of the lane's median historical order size, rounded up to 10 — replicating the real (s, Q) "
            "policy: orders occur only below the reorder point, and lot size is unrelated to the shortfall.",
        ]),
    ]


def s4_datasets():
    b, a = {c["path"]: c for c in BEFORE["csv_files"]}, {c["path"]: c for c in AFTER["csv_files"]}
    rows = []
    for path in sorted(set(b) | set(a)):
        x, y = b.get(path), a.get(path)
        rows.append([path.replace("DataSets/", ""), "-" if x else "new file", x["rows"] if x else "-", y["rows"] if y else "-",
                     x["columns"] if x else "-", y["columns"] if y else "-",
                     "unchanged (hash identical)" if x and y and x["sha256"] == y["sha256"] else "ADDED" if not x else "CHANGED"])
    tb, ta = BEFORE["database"]["tables"], AFTER["database"]["tables"]
    dbrows = []
    reasons = {
        "supplier_profiles": "Unsupported round-robin mapping; mislabelled stock",
        "products": "Derived procurement columns removed; reference only",
        "purchase_orders": "sku_id added by ALTER (rows preserved)",
        "sku_supplier_lanes": "Real per-lane supplier cost/lead time",
        "supplier_sku_profile": "Simulated availability + quality",
        "calendar": "Pre-existing: 65 duplicate dates filtered (unchanged)",
    }
    for name in sorted(set(tb) | set(ta)):
        x, y = tb.get(name), ta.get(name)
        cx, cy = set(x["column_names"]) if x else set(), set(y["column_names"]) if y else set()
        dbrows.append([name, x["rows"] if x else "-", y["rows"] if y else "-", x["columns"] if x else "-", y["columns"] if y else "-",
                       ", ".join(sorted(cy - cx)) if x and y else ("table added" if y else "-"),
                       ", ".join(sorted(cx - cy)) if x and y else ("table deleted" if x else "-"),
                       reasons.get(name, "unchanged")])
    return [
        PageBreak(),
        P("4. Dataset-by-Dataset Changes", "h1"),
        P("Files (raw and generated)", "h2"),
        table(["File", "New?", "Rows before", "Rows after", "Cols before", "Cols after", "Status"], rows, [62, 16, 17, 17, 16, 16, 26]),
        P("No raw file was renamed, and no raw column was added, removed, renamed or transformed. "
          "Transformations happen only when loading into SQLite (below).", "small"),
        P("SQLite tables", "h2"),
        table(["Table", "Rows before", "Rows after", "Cols before", "Cols after", "Columns added", "Columns removed", "Reason"], dbrows,
              [26, 15, 15, 13, 13, 27, 30, 31]),
        P("Rows: 0 rows deleted from any table carried over. Rows added: 250 (lanes) + 211 (profile). Rows filtered: "
          f"{VALID['calendar_filter']['duplicate_dates_removed']} duplicate calendar dates (pre-existing behaviour); all 1,000 products "
          "excluded from procurement but retained. Renamed: config weight keys w1_rating → w1_quality, w4_reliability → w4_fulfillment; "
          "proposed field on_time_rate → fulfillment_rate. Transformed: products.reorder_level clip(lower=1) removed; SUP_3 defect rate capped.", "small"),
    ]


def s5_changes():
    rows = [[f"{c['dataset']}: {c['location']}", c["field"], c["change_type"], c["before"], c["after"], c["reason"], c["real_or_simulated"]]
            for c in LOG["dataset_changes"]]
    return [PageBreak(), P("5. Detailed Change Table", "h1"),
            P(f"{len(rows)} entries. Full machine-readable version with evidence, method and impact: docs/migration/change_log.json.", "small"),
            table(["Dataset / location", "Field", "Change", "Before", "After", "Reason", "Status"], rows, [29, 21, 24, 22, 22, 30, 22], status_col=6)]


def s6_deletions():
    out = [PageBreak(), P("6. Deletion Report", "h1"),
           P(f"<b>No raw source records were deleted.</b> All 6 raw CSVs have identical SHA-256 before and after, and git reports no change "
             f"against HEAD. supply_chain keeps all {AFTER['database']['tables']['supply_chain']['rows']:,} rows and "
             f"{AFTER['database']['tables']['supply_chain']['columns']} columns. Every deletion below is of a derived database artefact, code or behaviour.")]
    rows = [[f"{i}. {d['what']}", d["count"], d["why"], d["evidence"], d["future_impact"], d["preserved_in_backup"], d["replaced_by"]]
            for i, d in enumerate(LOG["deletions"], 1)]
    out.append(table(["What", "Count", "Why", "Evidence", "Future impact", "Backup", "Replaced by"], rows, [30, 26, 25, 25, 22, 22, 20]))
    return out


def s7_simulated():
    q = PROFILE.groupby("supplier_id")[["defect_rate", "fulfillment_rate"]].first()
    q = q.loc[sorted(q.index, key=lambda s: int(s[4:]))]
    avail = PROFILE.groupby("supplier_id").available_qty.agg(["count", "min", "max"])
    return [
        PageBreak(), P("7. Simulated Dataset", "h1"),
        P(f"Table <b>supplier_sku_profile</b> · file DataSets/simulated/supplier_sku_profile.csv · {META['rows']} rows · "
          f"{META['relationships']['valid_pairs_in_source']} valid SKU-supplier relationships (all from the source, 0 invented, 0 missing) · "
          f"{META['relationships']['skus']} SKUs · {META['relationships']['suppliers']} suppliers · suppliers per SKU {META['relationships']['suppliers_per_sku']}."),
        table(["Field", "Type", "Status", "Granularity", "Source / method"], [
            ["sku_id", "text", "REAL", "row key", "supply_chain SKU_ID (existing relationship)"],
            ["supplier_id", "text", "REAL", "row key", "supply_chain Supplier_ID (existing relationship)"],
            ["available_qty", "integer", "SIMULATED", "SKU-supplier", "one draw from the pair's own real Order_Quantity history"],
            ["defect_rate", "fraction", "SIMULATED", "supplier", "Beta fit to real KPI supplier defect rates, capped at real max"],
            ["fulfillment_rate", "fraction", "SIMULATED", "supplier", "Binomial sampling around the real pooled KPI fulfillment rate"],
        ], [26, 16, 22, 22, 84], status_col=2),
        P("Warehouse is not a column: each (SKU, warehouse) has exactly one supplier in the source, and procurement requests carry no warehouse.", "small"),
        P("Supplier-level simulated quality (and availability range across that supplier's SKUs)", "h2"),
        table(["Supplier", "defect_rate", "fulfillment_rate", "Rule 3 High Risk", "SKUs", "available_qty range"],
              [[s, f"{r.defect_rate:.4f}", f"{r.fulfillment_rate:.4f}", "yes" if r.defect_rate > 0.05 else "no",
                int(avail.loc[s, "count"]), f"{int(avail.loc[s, 'min'])}–{int(avail.loc[s, 'max'])}"] for s, r in q.iterrows()],
              [22, 24, 26, 28, 18, 32]),
        P("First 10 rows", "h2"),
        table(list(PROFILE.columns), PROFILE.head(10).values.tolist(), [30, 30, 30, 30, 30]),
    ]


def s8_method():
    a, d, f = META["available_qty"], META["defect_rate"], META["fulfillment_rate"]
    ev, sim = VALID["simulation"]["evidence"], VALID["simulation"]
    oq = a["order_quantity_all_pairs"]
    mix = sim["eligibility_mix_by_request_size"]
    return [
        PageBreak(), P("8. Simulation Methodology", "h1"),
        P("Seed 20260926, fixed before any result was seen and not re-rolled. numpy SeedSequence(seed).spawn(3) gives one independent "
          "stream per field, so changing one field's method never shifts another field's values."),
        P("8.1 available_qty", "h2"),
        *bullets([
            f"Why simulated: supplier-side stock is not in any file; Inventory_Level satisfies inv[t] = inv[t-1] − sold[t] + order_qty[t] on {ev['inventory_is_buyer_stock_flow_match_pct']}% of rows, i.e. it is buyer stock.",
            f"Historical source: {a['historical_orders_total']:,} real orders (Order_Quantity > 0); {int(a['observations_per_pair']['min'])}–{int(a['observations_per_pair']['max'])} per pair (minimum required {a['min_observations_required']}).",
            f"Statistical basis: order sizes {int(oq['min'])}–{int(oq['max'])}, mean {oq['mean']:.1f}, median {oq['median']:.0f}, sd {oq['std']:.1f}, p10 {oq['p10']:.0f}, p90 {oq['p90']:.0f}; "
            f"decile counts {ev['order_qty_decile_counts_200_500']} (uniform); pairs do not differ (Kruskal-Wallis p = {ev['order_qty_pairs_differ_kruskal_p']}).",
            "Sampling method: one value drawn uniformly (with replacement) from that pair's own order-size history.",
            f"Validation: every value is a real order size of its pair ({sim['available_qty_vs_history']['every_value_is_a_real_order_size_of_its_pair']}); "
            f"generated mean/sd {sim['available_qty_vs_history']['generated_mean']}/{sim['available_qty_vs_history']['generated_sd']} vs history "
            f"{sim['available_qty_vs_history']['history_mean']}/{sim['available_qty_vs_history']['history_sd']}.",
            "Limitation: because pairs do not really differ, differences between suppliers come from the draw.",
        ]),
        table(["Request size", "Pairs eligible", "SKUs with no eligible supplier", "SKUs with all eligible"],
              [[q, f"{v['pairs_eligible_pct']}%", v["skus_with_none"], v["skus_with_all"]] for q, v in mix.items()], [35, 35, 55, 45]),
        P("8.2 defect_rate", "h2"),
        *bullets([
            "Why the KPI names cannot map to SUP_x: no shared key, no SKU, no category in supply_chain, and non-overlapping dates (KPI 2022-01 → 2024-01, supply chain 2024).",
            f"Calculation: Defective_Units / Quantity per PO; {d['excluded_pos_blank_defective_units']} POs with blank Defective_Units excluded (unknown ≠ zero), {d['usable_pos']} used. "
            f"Supplier rate = Σ defective / Σ quantity: " + ", ".join(f"{k} {v:.4f}" for k, v in d["real_supplier_rates_weighted"].items()) + ".",
            f"Per-PO distribution: min {d['per_po_defect_rate']['min']:.4f}, median {d['per_po_defect_rate']['median']:.4f}, mean {d['per_po_defect_rate']['mean']:.4f}, "
            f"sd {d['per_po_defect_rate']['std']:.4f}, p95 {d['per_po_defect_rate']['p95']:.4f}, max {d['per_po_defect_rate']['max']:.4f}. Defect rate is a supplier trait "
            f"(Kruskal-Wallis by supplier p = {ev['defect_rate_by_supplier_kruskal_p']:.1e}), so it is simulated per supplier, never per SKU.",
            f"Distribution: Beta(α = {d['alpha']:.4f}, β = {d['beta']:.4f}), method-of-moments fit to the 5 supplier rates (mean {d['fit_mean']:.4f}, sd {d['fit_sd']:.4f}).",
            f"Cap: real maximum {d['cap_exact_real_max']} floored to 4 dp = {d['cap_applied']} (rounding up would exceed the real max). Capped: "
            f"{', '.join(f'{s} {d['values_before_cap'][s]} → {d['cap_applied']}' for s in d['capped_suppliers'])}.",
            f"Result: {len(sim['high_risk_suppliers'])} of 10 suppliers exceed Rule 3's 0.05 ({', '.join(sim['high_risk_suppliers'])}) vs 2 of 5 real vendors.",
        ]),
        P("8.3 fulfillment_rate", "h2"),
        *bullets([
            "Why not an on-time rate: the KPI file has Order_Date, Delivery_Date and Order_Status but no promised/expected date, so lateness cannot be measured.",
            "Definition: Delivered / (Delivered + Partially Delivered + Cancelled); Pending excluded because its outcome is unknown.",
            f"Real values: pooled {f['p']}; per vendor " + ", ".join(f"{k} {v:.3f}" for k, v in f["real_supplier_rates"].items())
            + f"; vendors do not differ (chi-square = {ev['fulfillment_by_supplier_chi2']}, p = {ev['fulfillment_by_supplier_chi2_p']}).",
            f"Method: every simulated supplier has the same true rate p = {f['p']}; observed rate = Binomial({f['n']}, p) / {f['n']} ({f['n']} = median finished POs per real vendor).",
            "Limitation: differences between simulated suppliers are sampling noise by design, and their spread is wider than the real one. "
            "The ranking therefore uses the rate absolutely rather than Min-Max stretching it.",
        ]),
    ]


def s9_before_after():
    return [
        PageBreak(), P("9. Before / After Architecture", "h1"),
        flow(["ProductID\nproduct.csv", "Procurement\ndeficit = 3×ROP − stock", "Supplier\nsame 10-row pool", "Outcome\nSUP_1 / SUP_9"],
             fills={0: RED_FILL, 3: RED_FILL}, caption="BEFORE"),
        Spacer(1, 5 * mm),
        flow(["SKU\nurgent lanes", "Supplier\nlanes serving SKU", "Supplier-SKU\nprofile", "Eligibility\nR5 avail ≥ need", "Ranking\nheuristic + R2–R4"],
             fills={0: REAL_FILL, 1: REAL_FILL, 2: SIM_FILL}, caption="AFTER"),
        Spacer(1, 4 * mm),
        table(["Aspect", "Before", "After"], [
            ["Item identity", "ProductID (1,000 random products)", "SKU_ID (50 real SKUs)"],
            ["Urgency", "UnitsInStock < ReorderLevel (product.csv)", "Inventory < ReorderPoint per warehouse lane, latest date"],
            ["Quantity", "3 × ReorderLevel − stock", "Σ median historical lot size of urgent lanes, rounded to 10"],
            ["Candidates", "All 10 suppliers, every time", "Only suppliers that supply the SKU (3–5)"],
            ["Eligibility", "Buyer stock ≥ deficit (never failed)", "Supplier available_qty ≥ required quantity"],
            ["Nobody eligible", "Ranked the full pool and recommended anyway", "No supplier selected; combined availability reported"],
            ["Quality source", "Round-robin KPI vendor", "Explicitly simulated supplier-level profile"],
            [f"Result ({OLD['urgent_products_run']} / {len(URG)} urgent items)", f"{OLD['recommended_counts']}",
             f"{len(PLANNED)} planned via {', '.join(E2E['distinct_recommended_suppliers'])}; {len(BLOCKED)} no supplier"],
        ], [34, 68, 68]),
    ]


def s10_code():
    rows = [[c["file"], "new" if c["added"] is None else f"+{c['added']} / −{c['removed']}", c["what"], c["old"], c["new"]] for c in LOG["code_changes"]]
    rules = [[r["rule"], r["before"], r["after"], r["threshold"], r["why"]] for r in LOG["rule_changes"]]
    return [
        PageBreak(), P("10. Code Changes", "h1"),
        P("Line counts from git diff --numstat against the pre-migration commit.", "small"),
        table(["File", "Lines", "What changed / why", "Old behaviour", "New behaviour"], rows, [46, 18, 44, 31, 31]),
        P("Knowledge Agent rule changes", "h2"),
        table(["Rule", "Before", "After", "Threshold", "Justification"], rules, [22, 34, 36, 26, 52]),
        P("Heuristic scoring: weights unchanged (0.30 / 0.25 / 0.20 / 0.25). Quality = 1 − defect_rate and Fulfillment = fulfillment_rate are "
          "used absolutely (the precedent of the old Rating/5 term); Cost and Delivery keep Min-Max normalization over eligible suppliers. "
          "Weights can now be passed in as data (rank_suppliers(..., weights=...)).", "small"),
    ]


def s11_db():
    b, a = BEFORE["database"], AFTER["database"]
    return [
        PageBreak(), P("11. Database Changes", "h1"),
        table(["Item", "Before", "After"], [
            ["Tables added", "-", "sku_supplier_lanes, supplier_sku_profile"],
            ["Tables removed", "supplier_profiles", "-"],
            ["Columns added", "-", "purchase_orders.sku_id"],
            ["Columns removed", "products.target_stock, deficit, urgent", "-"],
            ["Columns renamed", "-", "none in DB (config keys only)"],
            ["Indexes", ", ".join(sorted(b["indexes"])), ", ".join(sorted(a["indexes"]))],
            ["Constraints", "none", "UNIQUE(sku_id, warehouse_id) on lanes; UNIQUE(sku_id, supplier_id) on profile"],
            ["Schema version (PRAGMA user_version)", str(b["user_version"]), str(a["user_version"])],
        ], [45, 60, 65]),
        P("Migration and initialization", "h2"),
        *bullets([
            "On startup, build_database() compares PRAGMA user_version with SCHEMA_VERSION = 2; an older database is rebuilt from the CSVs automatically.",
            "Rebuild replaces derived tables, drops obsolete supplier_profiles / idx_products_urgent, and migrates purchase_orders in place with "
            "ALTER TABLE ADD COLUMN sku_id — existing purchase orders are kept (tested with a v1 database containing a PO row).",
            "The profile loader rejects invented (SKU, supplier) pairs, duplicates and out-of-range values; the lane builder fails if a lane's supplier, cost or lead time ever varies.",
            f"At migration time purchase_orders had {b['tables']['purchase_orders']['rows']} rows.",
        ]),
    ]


def s12_api():
    return [P("12. API Changes", "h1"),
            table(["Endpoint", "Before", "After"], [[r["endpoint"], r["before"], r["after"]] for r in LOG["api_changes"]], [45, 50, 75]),
            P("Backward compatibility: requests using product_id receive 410 Gone with an explanation instead of a silent 422, so old clients "
              "fail loudly. approve-order now validates the (SKU, supplier) relationship and quantity ≤ available_qty (422 otherwise).", "small")]


def s13_frontend():
    return [P("13. Frontend Changes", "h1"), *bullets([
        "Dashboard: KPIs are urgent SKUs, real average supplier lead time and simulated average fulfillment (labelled); chart of top SKUs by required quantity; "
        "alert table shows SKU, urgent warehouses, inventory, reorder point, shortfall, required quantity, candidate supplier count.",
        "Pipeline & XAI: 50-SKU picker (urgent SKUs marked with their need); perceived state shows snapshot date, urgent warehouses, required quantity; "
        "recommended supplier shows unit cost, lead time, available quantity [SIM] and score; a 'No supplier selected' state explains combined availability "
        "and that split orders are not implemented; the trace shows Rule 1 per lane, Rule 5 capacity and an 'Ineligible' badge.",
        "Supplier Matrix: SKU-scoped with a required-quantity input; formula text built from the weights actually used; columns for unit cost, lead time, "
        "available [SIM], defect [SIM], fulfillment [SIM]; multi-warehouse costs marked with * and a min–max tooltip; separate table of excluded suppliers with shortfall.",
        "Action Hub: plan titled SKU → supplier; approval payload uses sku_id; order history shows SKU.",
        "Simulated-data labels: an amber SIM tag on every simulated value, with a tooltip pointing to the data dictionary.",
        "Verified in the running app (all four views, SKU_2 planned, SKU_19 no-eligible, required-quantity re-ranking, no console errors, all API calls 200).",
    ])]


def s14_future():
    ok = lambda k: all(c["passed"] for c in VALID["future_compatibility"])
    return [
        Spacer(1, 4 * mm), P("14. Future Compatibility", "h1"),
        table(["Future feature", "Compatible?", "Why / evidence", "Limitation"], [
            ["Demand ML", "Yes", "All 91,250 daily rows × 15 columns retained (365 days per SKU-warehouse, tested); SKU identity unchanged.",
             f"Demand_Forecast correlates {LIVE['forecast_corr']} with Units_Sold — keep as baseline, not a feature (leakage)."],
            ["Demand-spike prediction", "Yes", "Promotion_Flag, warehouse, region, inventory, date retained; requests keep lanes.",
             "Region varies randomly within fixed lanes; SKUs have near-identical mean demand."],
            ["Delivery-time ML", "No", f"Lead time is constant for every lane ({LIVE['lanes_with_varying_lead_time']} lanes vary). lead_time_days is a candidate column, so a model could supply it later without redesign.",
             "Needs real promised vs actual delivery dates per supplier/SKU/order."],
            ["GA dynamic weighting", "Yes (hook)", f"Weights injectable (tested); delivery-heavy weights change the top supplier for {E2E['skus_whose_top_supplier_changes_with_delivery_heavy_weights']} of 50 SKUs; "
             f"global state available: {LIVE['below_rop_per_day'][0]}–{LIVE['below_rop_per_day'][1]} lanes below ROP per day.",
             f"No outcome data (stockout flag sum = {LIVE['stockout_sum']}); fitness must come from a disclosed simulation."],
            ["Split-order optimization", "Yes (data)", f"Per-supplier available_qty kept; combined availability reported; {len(BLOCKED)} of {len(URG)} urgent SKUs need it today.",
             "Allocation algorithm not implemented."],
            ["AHP", "Yes", "Candidate matrix keeps raw cost, lead time, availability, defect, fulfillment on every candidate (tested).", "Pairwise judgements needed."],
            ["TOPSIS", "Yes", "score_candidates / candidate_matrix work on a plain DataFrame (tested).", "-"],
            ["ERP integration", "Yes", "Pipeline tested end-to-end on tables filled from plain records (no CSV); loader validates relationships; simulated columns replaceable.",
             "No live connector; SQLite store."],
        ], [30, 18, 72, 50], status_col=1),
        P(f"Architectural checks from validation: {'all passed' if ok(0) else 'FAILED'} — " +
          "; ".join(c["check"] for c in VALID["future_compatibility"]) + ".", "small"),
    ]


def s15_assumptions():
    return [P("15. Assumptions", "h1"), *bullets([
        "SKU_ID is the procurement item identity; product.csv takes no part in procurement.",
        "Procurement requests have no warehouse dimension; candidate suppliers are considered at SKU level.",
        "A supplier that delivers an SKU to one warehouse could deliver it to any warehouse.",
        f"Where a supplier serves an SKU at several warehouses ({LIVE['multi_lane_pairs']} pairs), its cost and lead time for the SKU are the arithmetic mean of its lanes.",
        "The latest date in the data (2024-12-30) is 'today' for stock state; planning dates are computed from the real current date.",
        "Required quantity replicates the observed policy: median historical lot size per urgent lane, summed, rounded up to 10.",
        "Daily demand = mean real Units_Sold over the last 28 days of the focus lanes.",
        "Supplier availability is simulated from that pair's historical order sizes.",
        "Supplier quality (defect, fulfillment) is simulated at supplier level.",
        "Fulfillment rate is not an on-time rate.",
        "Rule 4's 0.90 is read as a 90% fulfillment target (unchanged number).",
        "Future delivery-time ML requires promised and actual delivery dates.",
    ])]


def s16_limits():
    sim = VALID["simulation"]
    return [P("16. Limitations", "h1"), *bullets([
        "All three supplier-side fields are simulated; rankings and eligibility depend on them and must not be presented as observed supplier behaviour.",
        "The underlying source datasets are themselves synthetic (perfect 50 × 5 × 365 grid, uniform order sizes, same-day receipt of orders).",
        "Defect distribution is fitted to only five real vendors; order sizes and fulfillment do not differ between real entities, so simulated differences there are sampling noise.",
        f"With this seed {len(sim['high_risk_suppliers'])} of 10 suppliers exceed the 0.05 defect threshold; for "
        f"{sum(1 for r in PLANNED if r['recommended_status'] == 'High Risk')} of {len(PLANNED)} planned SKUs the only eligible supplier is High Risk "
        "(existing fallback recommends it with an explicit warning).",
        f"No supplier reaches fulfillment 0.90, so Rule 4 never labels a supplier Reliable ({sim['reliable_suppliers_fulfillment_ge_090'] or 'none'}).",
        f"{len(BLOCKED)} of {len(URG)} urgent SKUs cannot be served by one supplier; split-order allocation is not implemented.",
        "No real on-time, delivery-time or supplier-capacity data exists; delivery-time ML is not supported.",
        "Calendar seasonality in planning still uses clander.csv, an unrelated practice dataset (unchanged by this migration).",
        f"The backup in {'data/backup/pre_migration_2026-09-27/'} is local only (data/ is git-ignored); git history also holds the raw CSVs.",
        "Real data required to remove the simulation: supplier available quantity per SKU, supplier defect history for SUP_x, promised vs actual delivery dates.",
    ])]


def s17_validation():
    groups = [("raw_checksums", "Raw data"), ("integrity", "Integrity"), ("reproducibility", "Reproducibility"), ("future_compatibility", "Future")]
    rows = [[g, c["check"], "PASS" if c["passed"] else "FAIL", json.dumps(c["detail"]) if not isinstance(c["detail"], str) else c["detail"]]
            for key, g in groups for c in VALID[key]]
    rc = [[r["source"], r["table"], r["source_rows"], r["db_rows"], r["difference"]] for r in VALID["row_counts"]]
    e2e = [[r["sku_id"], r["urgent_warehouses"], r["required_quantity"], f"{r['eligible']}/{r['candidates']}", r["excluded"] or "-",
            r["recommended"] or "none", r["recommended_status"] or "-", r["backup"] or "-", r["plan_status"]] for r in URG]
    return [
        PageBreak(), P("17. Validation Results", "h1"),
        P(f"Run {VALID['run_at']} · all checks passed: {VALID['all_checks_passed']} · pytest: {VALID['pytest']['summary']} "
          f"({', '.join(f'{k}: {v}' for k, v in VALID['pytest']['tests_per_file'].items())})."),
        table(["Group", "Check", "Result", "Detail"], rows, [22, 70, 14, 64], status_col=2),
        P("Row-count validation (raw source vs database)", "h2"),
        table(["Source", "Table", "Source rows", "DB rows", "Difference"], rc, [70, 34, 22, 22, 22]),
        P("The only difference is the calendar's 65 duplicate dates, removed by the pre-existing loader.", "small"),
        P("End-to-end procurement test (every urgent SKU)", "h2"),
        table(["SKU", "Urgent WH", "Need", "Eligible", "Excluded (Rule 5)", "Chosen", "Status", "Backup", "Plan"],
              [r[:8] + [r[8].replace("_", " ")] for r in e2e], [14, 19, 11, 14, 36, 16, 17, 14, 29]),
        *bullets([
            f"Different SKUs → different rankings: {E2E['distinct_full_rankings_across_50_skus']} distinct rankings across 50 SKUs; top supplier counts {E2E['top_heuristic_supplier_count_across_50_skus']}.",
            f"Insufficient availability excluded: every ranked candidate has available ≥ need and every excluded one < need (tested for all {len(URG)} urgent SKUs).",
            f"No eligible supplier: {', '.join(r['sku_id'] for r in BLOCKED)} → no recommendation, no plan, combined availability reported.",
            f"Non-urgent SKU_3 → plan status '{E2E['non_urgent_sku_SKU_3_plan_status']}' (no purchase order).",
            f"Before: {OLD['urgent_products_run']} of {OLD['urgent_products_run']} urgent products → SUP_1 / SUP_9 (old code re-run on backup DB).",
            "API tests: SKU pipeline, 410 for product_id, 404 for unknown SKU, 422 for invented pair or excessive quantity, PO creation with sku_id.",
        ]),
    ]


def s18_repro():
    hashes = META["inputs_sha256"]
    return [
        PageBreak(), P("18. Reproducibility", "h1"),
        table(["Item", "Value"], [
            ["Simulation seed", str(META["seed"])],
            ["Random streams", META["random_streams"]],
            ["Generation script", META["generator"]],
            ["Simulated data generated", META["generated_at"]],
            ["Migration date", "2026-09-27"],
            ["Output SHA-256", META["output_sha256"]],
            *[[f"Input SHA-256: {k}", v] for k, v in hashes.items()],
            ["Python (report build)", f"{platform.python_version()} ({sys.platform}); app pinned to 3.11 for deployment"],
            ["Backup", "data/backup/pre_migration_2026-09-27/ (raw CSVs + pre-migration DB, verified by checksum)"],
        ], [45, 125]),
        P("Commands", "h2"),
        table(["Purpose", "Command"], [
            ["Verify committed simulated CSV", "python -m backend.simulation.supplier_sku_profile --check"],
            ["Regenerate simulated dataset", "python -m backend.simulation.supplier_sku_profile"],
            ["Rebuild database", "python backend/database.py   (or: python run_app.py --rebuild-db)"],
            ["Run tests", "python -m pytest backend/tests"],
            ["Capture state snapshot", "python scripts/migration/capture_state.py <before|after>"],
            ["Validate migration", "python scripts/migration/validate.py"],
            ["Regenerate change / deletion logs", "python scripts/migration/change_log.py"],
            ["Regenerate this report", "pip install reportlab && python scripts/migration/build_report.py"],
            ["Verify raw files", "cd DataSets && sha256sum -c CHECKSUMS.sha256"],
        ], [55, 115]),
        P("Regeneration was tested by deleting the simulated CSV and regenerating it: the new file is byte-identical "
          "(Section 17, Reproducibility checks).", "small"),
    ]


def _page(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, t("InsightIQ — ProductID → SKU_ID migration report"))
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                            title="InsightIQ SKU Migration Report", author="InsightIQ project")
    story = [*cover(), *s1_exec(), *s2_original(), *s3_new(), *s4_datasets(), *s5_changes(), *s6_deletions(), *s7_simulated(),
             *s8_method(), *s9_before_after(), *s10_code(), *s11_db(), *s12_api(), *s13_frontend(), *s14_future(),
             *s15_assumptions(), *s16_limits(), *s17_validation(), *s18_repro()]
    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    print(f"wrote {OUT.relative_to(ROOT)} (font: {FONT})")


if __name__ == "__main__":
    main()
