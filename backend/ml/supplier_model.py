"""
Supplier performance model — learns how each supplier actually performs from its
purchase-order outcomes (DataSets/augmented/purchase_order_outcomes.csv).

Predicted per SKU-supplier pair (hierarchical empirical Bayes, i.e. partial pooling):

  defect_rate        rejected / received units          Beta-Binomial, pair -> supplier -> all
  fill_rate          received / ordered units           pooled ratio, pair -> supplier
  on_time_prob       P(delay = 0)                       Beta-Binomial, pair -> supplier -> all
  expected_delay     mean delay days (incl. on-time 0)  shrunk mean, pair -> supplier
  delay_sd           spread of the delay                supplier level
  otif_prob          on_time_prob * P(no short ship)    "on time in full"

Partial pooling: a pair with few orders borrows strength from its supplier, and a
supplier with few orders from all suppliers. The strength of borrowing (prior
"pseudo-orders") is estimated from the data itself (method of moments), not set by
hand. Everything is deterministic.

`backtest()` trains on orders received before a cutoff and scores predictions on
the orders received after it, against the naive baseline "every supplier performs
like the average supplier".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_KAPPA, MAX_KAPPA = 1.0, 1e5


def _beta_kappa(successes: pd.Series, trials: pd.Series) -> tuple[float, float]:
    """(prior mean, prior strength in pseudo-trials) for group rates, by method of moments."""
    trials = trials.astype(float)
    rate = successes / trials
    mu = float(successes.sum() / trials.sum())
    if len(rate) < 2 or mu <= 0 or mu >= 1:
        return mu, MAX_KAPPA
    w = trials / trials.sum()
    var_obs = float((w * (rate - mu) ** 2).sum())
    var_sampling = float((w * mu * (1 - mu) / trials).sum())
    between = var_obs - var_sampling
    if between <= 0:
        return mu, MAX_KAPPA
    kappa = mu * (1 - mu) / between - 1
    return mu, float(np.clip(kappa, MIN_KAPPA, MAX_KAPPA))


def _shrink(successes, trials, prior_mean, kappa):
    return (successes + kappa * prior_mean) / (trials + kappa)


def fit(outcomes: pd.DataFrame) -> dict:
    """Learn supplier- and pair-level performance. Returns {"pairs": DataFrame, "suppliers": DataFrame, "params": dict}."""
    o = outcomes.copy()
    o["on_time"] = (o.delay_days <= 0).astype(int)
    o["shorted"] = (o.short_qty > 0).astype(int)

    sup = o.groupby("supplier_id").agg(
        orders=("po_id", "size"), rej=("rejected_qty", "sum"), rec=("received_qty", "sum"),
        ordq=("ordered_qty", "sum"), on_time=("on_time", "sum"), shorted=("shorted", "sum"),
        delay_sum=("delay_days", "sum"), delay_sd=("delay_days", "std"),
        promised=("promised_lead_time_days", "mean"),
    )
    pair = o.groupby(["sku_id", "supplier_id"]).agg(
        orders=("po_id", "size"), rej=("rejected_qty", "sum"), rec=("received_qty", "sum"),
        ordq=("ordered_qty", "sum"), on_time=("on_time", "sum"), shorted=("shorted", "sum"),
        delay_sum=("delay_days", "sum"),
    ).reset_index()

    # defect: units are not independent trials (one bad batch spoils many), so the
    # pooling strength is measured in ORDERS: per-order rates, weight = orders.
    def order_level(rate_num, rate_den, groups):
        per_order = (o[rate_num] / o[rate_den]).rename("r")
        g = pd.concat([o[groups], per_order], axis=1).groupby(groups).r.agg(["mean", "count", "var"])
        return g

    params: dict = {}
    # --- supplier level (shrunk to all suppliers) ---
    mu_def = float(o.rejected_qty.sum() / o.received_qty.sum())
    g = order_level("rejected_qty", "received_qty", ["supplier_id"])
    within = float(np.nanmean(g["var"]))
    between = max(float(g["mean"].var(ddof=1)) - within / float(g["count"].mean()), 1e-9)
    k_def_sup = within / between
    sup["defect_rate"] = (sup.orders * (sup.rej / sup.rec) + k_def_sup * mu_def) / (sup.orders + k_def_sup)

    mu_ot, k_ot_sup = _beta_kappa(sup.on_time, sup.orders)
    sup["on_time_prob"] = _shrink(sup.on_time, sup.orders, mu_ot, k_ot_sup)
    mu_sh, k_sh_sup = _beta_kappa(sup.shorted, sup.orders)
    sup["short_prob"] = _shrink(sup.shorted, sup.orders, mu_sh, k_sh_sup)
    sup["fill_rate"] = sup.rec / sup.ordq
    sup["expected_delay"] = sup.delay_sum / sup.orders
    sup["delay_sd"] = sup.delay_sd.fillna(0.0)

    # --- pair level (shrunk to its supplier) ---
    gp = order_level("rejected_qty", "received_qty", ["sku_id", "supplier_id"])
    within_p = float(np.nanmean(gp["var"]))
    gp = gp.reset_index().merge(sup[["defect_rate"]].rename(columns={"defect_rate": "sup_def"}), left_on="supplier_id", right_index=True)
    between_p = max(float(((gp["mean"] - gp.sup_def) ** 2).mean()) - within_p / float(gp["count"].mean()), 1e-9)
    k_def_pair = within_p / between_p
    pair = pair.merge(sup[["defect_rate", "on_time_prob", "short_prob", "fill_rate", "expected_delay", "delay_sd"]]
                      .add_prefix("sup_"), left_on="supplier_id", right_index=True)
    pair["defect_rate"] = (pair.orders * (pair.rej / pair.rec) + k_def_pair * pair.sup_defect_rate) / (pair.orders + k_def_pair)
    # on-time / short / delay are supplier behaviours; pairs get a light pull to their supplier
    k_pair = 20.0
    pair["on_time_prob"] = _shrink(pair.on_time, pair.orders, pair.sup_on_time_prob, k_pair)
    pair["short_prob"] = _shrink(pair.shorted, pair.orders, pair.sup_short_prob, k_pair)
    pair["fill_rate"] = (pair.rec + k_pair * pair.sup_fill_rate * pair.ordq / pair.orders) / (pair.ordq + k_pair * pair.ordq / pair.orders)
    pair["expected_delay"] = (pair.delay_sum + k_pair * pair.sup_expected_delay) / (pair.orders + k_pair)
    pair["delay_sd"] = pair.sup_delay_sd
    pair["otif_prob"] = pair.on_time_prob * (1 - pair.short_prob)

    params.update({
        "defect_prior_mean": mu_def, "defect_kappa_supplier_orders": k_def_sup, "defect_kappa_pair_orders": k_def_pair,
        "on_time_prior_mean": mu_ot, "on_time_kappa_supplier": k_ot_sup,
        "short_prior_mean": mu_sh, "short_kappa_supplier": k_sh_sup, "pair_kappa_behaviour": k_pair,
        "orders": int(len(o)), "trained_from": str(o.received_date.min())[:10], "trained_through": str(o.received_date.max())[:10],
    })
    keep = ["sku_id", "supplier_id", "orders", "defect_rate", "fill_rate", "on_time_prob", "short_prob",
            "expected_delay", "delay_sd", "otif_prob"]
    return {"pairs": pair[keep], "suppliers": sup.reset_index(), "params": params}


def backtest(outcomes: pd.DataFrame, cutoff: str = "2024-10-01") -> dict:
    """Train on orders received before `cutoff`, score on the rest (per order)."""
    o = outcomes.copy()
    rd = pd.to_datetime(o.received_date)
    train, test = o[rd < cutoff], o[rd >= cutoff]
    model = fit(train)
    p = test.merge(model["pairs"], on=["sku_id", "supplier_id"], how="left", suffixes=("", "_pred"))
    base_def = train.rejected_qty.sum() / train.received_qty.sum()
    base_ot = float((train.delay_days <= 0).mean())
    base_delay = float(train.delay_days.mean())

    actual_def = p.rejected_qty / p.received_qty
    on_time = (p.delay_days <= 0).astype(float)

    def logloss(prob, y):
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(prob) + (1 - y) * np.log(1 - prob)).mean())

    w = p.received_qty
    res = {
        "cutoff": cutoff, "train_orders": int(len(train)), "test_orders": int(len(test)),
        "defect_mae_pp": {  # unit-weighted mean absolute error, percentage points
            "model": round(float((w * (actual_def - p.defect_rate).abs()).sum() / w.sum()) * 100, 3),
            "baseline_average_supplier": round(float((w * (actual_def - base_def).abs()).sum() / w.sum()) * 100, 3),
        },
        "on_time_logloss": {"model": round(logloss(p.on_time_prob, on_time), 4),
                            "baseline_average_supplier": round(logloss(np.full(len(p), base_ot), on_time), 4)},
        "delay_mae_days": {"model": round(float((p.delay_days - p.expected_delay).abs().mean()), 3),
                           "baseline_average_supplier": round(float((p.delay_days - base_delay).abs().mean()), 3)},
    }
    # can the model tell good from bad suppliers? rank agreement of supplier defect rates, train vs test
    sup_test = test.groupby("supplier_id").apply(lambda g: g.rejected_qty.sum() / g.received_qty.sum(), include_groups=False)
    sup_pred = model["suppliers"].set_index("supplier_id").defect_rate.reindex(sup_test.index)
    res["supplier_defect_rank_corr"] = round(float(sup_pred.rank().corr(sup_test.rank())), 3)
    return res
