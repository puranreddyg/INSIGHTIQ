"""Central configuration for InsightIQ: paths, weights and tunable heuristics."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "DataSets"
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "insightiq.db"
FRONTEND_DIR = ROOT_DIR / "frontend"

# Bumped whenever the derived schema changes; an older on-disk DB is rebuilt on startup.
SCHEMA_VERSION = 3

CSV_PATHS = {
    "product": DATASET_DIR / "archive (1)" / "product.csv",
    "order": DATASET_DIR / "archive (1)" / "order.csv",
    "customer": DATASET_DIR / "archive (1)" / "customer.csv",
    "calendar": DATASET_DIR / "archive (1)" / "clander.csv",
    "procurement_kpi": DATASET_DIR / "Procurement KPI Analysis Dataset.csv",
    "supply_chain": DATASET_DIR / "supply_chain_dataset1.csv",
}

# Provenance of every criterion the procurement agents read.
#   REAL       observed in the supply-chain ledger
#   ESTIMATED  computed from real history by backend/ml/supplier_estimator.py (no random numbers);
#              quality fields start at the real KPI average and move with real supplier evidence
#   MODEL      predicted by the demand forecaster (backend/ml/demand_forecaster.py)
FIELD_PROVENANCE = {
    "unit_cost": "REAL",
    "lead_time_days": "REAL",
    "available_qty": "ESTIMATED",
    "defect_rate": "ESTIMATED",
    "fulfillment_rate": "ESTIMATED",
}

MIN_ORDER_OBSERVATIONS_PER_PAIR = 10

# Heuristic Rank Score weights (must sum to 1.0). Values are unchanged from the
# product-based version; w1 was "rating" and w4 was "reliability".
HEURISTIC_WEIGHTS = {
    "w1_quality": 0.30,
    "w2_cost": 0.25,
    "w3_delivery": 0.20,
    "w4_fulfillment": 0.25,
}

# Knowledge Agent rule thresholds
RULE_PREFERRED_DELIVERY_DAYS = 3
RULE_HIGH_RISK_DEFECT_RATE = 0.05
RULE_RELIABLE_THRESHOLD = 0.90
# A supplier's quality status (Rules 2-4) is only asserted once it has at least this many
# purchase orders of its own evidence; before that it sits on the prior and is "Unrated".
RULE_MIN_EVIDENCE_POS = 5

# Perception / Planning
ORDER_QUANTITY_ROUNDING = 10  # round order qty up to nearest N units
DEMAND_LOOKBACK_DAYS = 28  # trailing window kept only as the previous method's comparison figure
SERVICE_LEVEL_Z = 1.645  # 95% service level for safety stock / forecast reorder point
ORDER_CYCLE_FALLBACK_DAYS = 14  # used when a lane has too few historical orders to measure its cycle

TOP_N_SUPPLIER_CANDIDATES = 5
