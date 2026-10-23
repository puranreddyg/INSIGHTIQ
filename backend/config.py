"""Central configuration for InsightIQ: paths, weights and tunable heuristics."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "DataSets"
SIMULATED_DATASET_DIR = DATASET_DIR / "simulated"
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
    "supplier_sku_profile": SIMULATED_DATASET_DIR / "supplier_sku_profile.csv",
}

# Provenance of every supplier criterion the procurement agents read. SIMULATED
# fields fill gaps the source data genuinely lacks (DataSets/simulated/README.md)
# and are meant to be replaced by real ERP / supplier-performance data.
FIELD_PROVENANCE = {
    "unit_cost": "REAL",
    "lead_time_days": "REAL",
    "available_qty": "SIMULATED",
    "defect_rate": "SIMULATED",
    "fulfillment_rate": "SIMULATED",
}

SIMULATION_SEED = 20260926
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

# Perception / Planning
ORDER_QUANTITY_ROUNDING = 10  # round order qty up to nearest N units
DEMAND_LOOKBACK_DAYS = 28  # trailing window of real Units_Sold used for daily demand

# RequiredQuantity method (backend/agents/perception_agent.py):
#   "forecast" — order-up-to: forecast demand over (lead time + review period)
#                + safety stock − current inventory, per urgent lane.
#   "median"   — legacy: sum of the urgent lanes' median historical Order_Quantity.
#                Kept as a fallback until the forecast method is validated in use.
REQUIRED_QUANTITY_METHOD = "forecast"

# Demand forecast used by the "forecast" method (backend/forecasting):
#   "auto"           — the regression model if its artifact is valid AND its held-out
#                      horizon WAPE beat the moving average; otherwise moving average.
#   "moving_average" — mean of the last DEMAND_LOOKBACK_DAYS days, flat over the horizon.
#   "seasonal_naive" — repeats the last 7 days (weekly seasonal naive).
#   "model"          — the regression model whenever its artifact is valid.
FORECAST_METHOD = "auto"

# Review period R added to the lead time L (protection interval L + R). The source
# data holds no review-period setting, so it is derived per lane as the median
# number of days between that lane's real historical orders; this default is used
# when a lane has fewer than two orders. Set FORECAST_REVIEW_PERIOD_SOURCE to
# "fixed" to always use the default (0 = pure lead-time horizon).
FORECAST_REVIEW_PERIOD_SOURCE = "order_cycle"
FORECAST_REVIEW_PERIOD_DAYS = 14

# Safety stock: "service_level" = z(SERVICE_LEVEL) * sigma_daily * sqrt(L + R),
# assuming i.i.d. normal daily demand and a deterministic lead time (lead times are
# constant per lane in the source). "days_of_cover" = SAFETY_STOCK_DAYS * mean demand.
SAFETY_STOCK_METHOD = "service_level"
SERVICE_LEVEL = 0.95
SAFETY_STOCK_DAYS = 3

TOP_N_SUPPLIER_CANDIDATES = 5
