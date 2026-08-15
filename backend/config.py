"""Central configuration for InsightIQ: paths, weights and tunable heuristics."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "DataSets"
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "insightiq.db"
FRONTEND_DIR = ROOT_DIR / "frontend"

CSV_PATHS = {
    "product": DATASET_DIR / "archive (1)" / "product.csv",
    "order": DATASET_DIR / "archive (1)" / "order.csv",
    "customer": DATASET_DIR / "archive (1)" / "customer.csv",
    "calendar": DATASET_DIR / "archive (1)" / "clander.csv",
    "procurement_kpi": DATASET_DIR / "Procurement KPI Analysis Dataset.csv",
    "supply_chain": DATASET_DIR / "supply_chain_dataset1.csv",
}

# Target stock level = ReorderPoint * this multiplier. Deficit = Target - Inventory.
# Represents the buffer needed to cover lead-time demand plus safety stock.
TARGET_STOCK_MULTIPLIER = 3

# Heuristic Rank Score weights (must sum to 1.0): rating, cost, delivery time, reliability
HEURISTIC_WEIGHTS = {
    "w1_rating": 0.30,
    "w2_cost": 0.25,
    "w3_delivery": 0.20,
    "w4_reliability": 0.25,
}

# Knowledge Agent rule thresholds
RULE_PREFERRED_RATING = 4.5
RULE_PREFERRED_DELIVERY_DAYS = 3
RULE_HIGH_RISK_DEFECT_RATE = 0.05
RULE_RELIABLE_THRESHOLD = 0.90

# Planning Agent
DEFAULT_DAILY_DEMAND_FALLBACK = 15  # units/day, used when no SKU-level history matches
ORDER_QUANTITY_ROUNDING = 10  # round order qty up to nearest N units

TOP_N_SUPPLIER_CANDIDATES = 5
