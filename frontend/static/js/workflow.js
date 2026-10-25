// ===========================================================================
// Supplier-selection workflow views — rendered ONLY from what the backend recorded:
//   supplier.scored_ranking / eligible_single / non_scorable   (Supplier Intelligence Agent)
//   knowledge.single_supplier_evaluation                        (Knowledge Agent, per-supplier trace)
//   knowledge.search_trace                                      (procurement_search, recorded as it ran)
// Nothing here recomputes a score, rule or search step. Helpers (esc, fmtNum, …) live in app.js.
// ===========================================================================

function wfBadge(text, kind) {
  return `<span class="wf-badge wf-${kind}">${esc(text)}</span>`;
}

function wfResultKind(r) {
  if (r === "PASS" || r === "MATCHED") return "pass";
  if (r === "FAIL") return "fail";
  return "neutral"; // NOT MATCHED, NOT APPLICABLE, NOT YET EVALUATED
}

function wfNum(v, d = 3) {
  return v === null || v === undefined || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d);
}

// ---- Supplier Matrix: Section A / Section B / non-scorable ------------------

function eligibilityBadge(s) {
  return s.can_fulfil_full_order ? wfBadge("Eligible for full order", "pass") : wfBadge("Insufficient capacity for full order", "fail");
}

function rankingSectionAHtml(data) {
  const rows = data.scored_ranking || [];
  if (!rows.length) {
    return `<div class="card p-5 text-sm"><h2 class="font-semibold text-heading">Section A — Complete heuristic ranking</h2>
      <p class="text-body mt-1">No supplier for this SKU has all the inputs needed to be scored.</p></div>`;
  }
  const req = data.required_quantity;
  return `
    <div class="card p-0 overflow-hidden" id="matrix-section-a">
      <div class="px-5 py-4 border-b border-soft">
        <h2 class="font-semibold text-heading">Section A — Complete heuristic ranking (${rows.length} scorable supplier${rows.length === 1 ? "" : "s"})</h2>
        <p class="text-xs text-body">Every scorable supplier is scored together and sorted by heuristic score, highest first — capacity does not change the order.
          A supplier that cannot cover the full ${fmtNum(req)} units stays in the ranking, labelled; it is just not a single-supplier candidate.</p>
      </div>
      <div class="overflow-x-auto">
        <table class="table-clean w-full">
          <thead><tr><th>Rank</th><th>Supplier</th><th>Score</th><th>Unit Cost</th><th>Predicted Lead (d)</th><th>Available</th><th>Required</th>
            <th>Covers</th><th>Defect</th><th>Evidence POs</th><th>Eligibility</th></tr></thead>
          <tbody>${rows.map((s) => `
            <tr class="${s.can_fulfil_full_order ? "" : "wf-row-muted"}">
              <td class="font-semibold text-heading">#${s.rank}</td>
              <td class="font-medium text-heading">${esc(s.supplier_id)}</td>
              <td class="tabular">${wfNum(s.heuristic_score, 1)}</td>
              <td class="tabular">${fmtCurrency(s.unit_cost)}</td>
              <td class="tabular">${fmtDays(s.lead_time_days)}</td>
              <td class="tabular">${fmtNum(s.available_qty)}</td>
              <td class="tabular">${fmtNum(req)}</td>
              <td class="tabular">${s.fulfilment_ratio == null ? "—" : Math.round(s.fulfilment_ratio * 100) + "%"}</td>
              <td class="tabular">${(s.defect_rate * 100).toFixed(2)}%</td>
              <td class="tabular">${fmtNum(s.evidence_pos)}</td>
              <td>${eligibilityBadge(s)}<div class="text-xs text-muted mt-1">${esc(s.eligibility_reason)}</div></td>
            </tr>`).join("")}</tbody>
        </table>
      </div>
    </div>`;
}

function nonScorableHtml(data) {
  const rows = data.non_scorable || [];
  if (!rows.length) return "";
  return `
    <div class="card p-0 overflow-hidden">
      <div class="px-5 py-4 border-b border-soft">
        <h2 class="font-semibold text-heading">Excluded — not scorable (${rows.length})</h2>
        <p class="text-xs text-body">These suppliers are missing an input the heuristic needs. No score is invented for them and they are not part of the ranking, the eligible list or the multi-supplier search.</p>
      </div>
      <div class="overflow-x-auto"><table class="table-clean w-full">
        <thead><tr><th>Supplier</th><th>Status</th><th>Missing inputs</th><th>Reason</th></tr></thead>
        <tbody>${rows.map((r) => `<tr><td class="font-medium text-heading">${esc(r.supplier_id)}</td><td>${wfBadge("Not scorable", "neutral")}</td>
          <td>${esc((r.missing_fields || []).join(", "))}</td><td class="text-body">${esc(r.reason)}</td></tr>`).join("")}</tbody>
      </table></div>
    </div>`;
}

// ---- Single-supplier evaluation (Knowledge Agent) ---------------------------

function ruleStepsTable(rules) {
  if (!rules.length) return "";
  return `<table class="table-clean w-full wf-inner"><thead><tr><th>Rule</th><th>Inputs vs thresholds</th><th>Result</th><th>Resulting status</th></tr></thead><tbody>
    ${rules.map((r) => `<tr>
      <td><span class="font-medium text-heading">${esc(r.rule_id)}</span> ${esc(r.rule_name)}<div class="text-xs text-muted">${esc(r.if || "")}</div></td>
      <td>${r.conditions.map((c) => `<div class="tabular text-xs">${esc(prettyKey(c.label))}: ${esc(c.expression)} ${c.result ? "✓" : "✗"}</div>`).join("")}${r.logic ? `<div class="text-xs text-muted">combined with ${esc(r.logic)}</div>` : ""}</td>
      <td>${wfBadge(r.result, wfResultKind(r.result))}</td>
      <td>${r.outcome ? esc(r.outcome) : "—"}</td></tr>`).join("")}
  </tbody></table>`;
}

function singleEvaluationHtml(ev) {
  if (!ev) return "";
  const outcomeKind = (o) => (o === "SELECTED" ? "pass" : o === "REJECTED" ? "fail" : "neutral");
  const banner = ev.case === "A" || ev.case === "B"
    ? `<div class="wf-banner wf-banner-warn"><div class="font-semibold">${esc(ev.headline)}</div>
        ${ev.reasons.length ? `<ul class="wf-reasons">${ev.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : ""}
        ${ev.transition_message ? `<div class="mt-2 font-medium">${esc(ev.transition_message)}</div>` : ""}</div>`
    : ev.case === "SELECTED" ? `<div class="wf-banner wf-banner-ok"><div class="font-semibold">${esc(ev.headline)}</div>
        <div class="text-xs mt-1">Multi-supplier allocation was not needed.</div></div>` : "";
  const eligibleBlock = ev.eligible_list.length
    ? `<div class="text-xs text-body mb-2">Eligible single-supplier list (heuristic order, original ranks kept): ${ev.eligible_list.map((e) => `<span class="wf-chip">#${e.rank} ${esc(e.supplier_id)} · ${wfNum(e.heuristic_score, 1)} · ${fmtNum(e.available_qty)} avail.</span>`).join(" ")}</div>`
    : `<div class="wf-banner wf-banner-neutral mb-2">${esc(ev.eligible_empty_message || "")}</div>`;
  const evals = ev.evaluations.map((e) => `
    <details class="wf-eval ${e.phase === "backup_evaluation" ? "wf-eval-backup" : ""}" ${e.outcome === "SELECTED" || e.step === 1 ? "open" : ""}>
      <summary>
        <span class="wf-step">Step ${e.step}</span>
        <span class="font-medium text-heading">${esc(e.supplier_id)}</span>
        <span class="text-xs text-muted">rank #${e.rank} · score ${wfNum(e.heuristic_score, 1)}</span>
        ${wfBadge(e.outcome, outcomeKind(e.outcome))}
        <span class="text-xs text-body wf-flow">${esc(e.flow)}</span>
      </summary>
      <div class="wf-eval-body">
        <div class="text-xs text-body mb-2">${esc(e.decision_reason)}</div>
        ${ruleStepsTable(e.rules)}
        <div class="wf-gates">${e.gates.map((g) => `<div class="wf-gate">${wfBadge(g.result, wfResultKind(g.result))} <span class="font-medium">${esc(g.gate)}</span> <span class="text-xs text-muted">${esc(g.detail)}</span></div>`).join("")}</div>
        <div class="text-xs text-muted mt-1">Resulting status: <b>${esc(e.status)}</b> · order cost ${fmtCurrency(e.order_cost)}${e.phase === "backup_evaluation" ? " · backup evaluation only — the selection had already stopped" : ""}</div>
      </div>
    </details>`).join("");
  return `
    <div class="card p-5 wf-single-evaluation">
      <h2 class="font-semibold text-heading mb-1">Single-Supplier Evaluation — one supplier at a time, in heuristic order</h2>
      <p class="text-xs text-body mb-3">Only suppliers able to supply all ${fmtNum(ev.required_quantity)} units are evaluated. The first one that is not High Risk and fits the budget is selected and the selection stops;
        suppliers after it are evaluated only to list backups. ${esc(ev.result_legend)}</p>
      ${eligibleBlock}
      ${banner}
      <div class="mt-3 space-y-2">${evals}</div>
    </div>`;
}

// ---- Multi-Supplier Search Trace --------------------------------------------

const TRACE_FILTERS = [
  ["all", "All events"], ["expanded", "Expanded"], ["rejected", "Rejected"], ["pruned", "Pruned"], ["feasible", "Feasible / best"], ["final", "Final"],
];
const TRACE_TYPE_LABEL = {
  search_started: ["Search started", "neutral"], node_created: ["Generated", "neutral"], node_queued: ["Queued", "neutral"], node_expanded: ["Expanded", "info"],
  combination_evaluated: ["Evaluated", "info"], solver_invoked: ["Solver invoked", "info"], combination_rejected: ["Rejected", "fail"],
  branch_pruned: ["Pruned", "warn"], feasible_allocation: ["Feasible allocation", "pass"], new_best: ["New best", "pass"], search_completed: ["Search completed", "neutral"],
};
const TRACE_KIND = {
  node_expanded: "expanded", combination_rejected: "rejected", branch_pruned: "pruned", feasible_allocation: "feasible", new_best: "feasible", search_completed: "final",
};

function traceConstraint(name, c) {
  if (!c) return `<div class="wf-cons">${wfBadge("NOT YET EVALUATED", "neutral")} <b>${name}</b></div>`;
  const bits = [];
  if (c.combined !== undefined) bits.push(`combined capacity ${fmtNum(c.combined)} vs required ${fmtNum(c.required)}`);
  if (c.weighted_defect_rate !== undefined && c.weighted_defect_rate !== null) bits.push(`blended defect ${(c.weighted_defect_rate * 100).toFixed(2)}% (limit ${(c.limit * 100).toFixed(0)}%)`);
  if (c.lowest_reachable_blended_defect !== undefined && c.lowest_reachable_blended_defect !== null) bits.push(`lowest reachable ${(c.lowest_reachable_blended_defect * 100).toFixed(2)}% (limit ${(c.limit * 100).toFixed(0)}%)`);
  if (c.total_spend !== undefined) bits.push(`spend ${fmtCurrency(c.total_spend)}${c.budget != null ? ` vs budget ${fmtCurrency(c.budget)}` : " (no budget set)"}`);
  if (c.cheapest_possible_spend !== undefined && c.cheapest_possible_spend !== null) bits.push(`cheapest possible ${fmtCurrency(c.cheapest_possible_spend)} vs budget ${fmtCurrency(c.budget)}`);
  if (c.min_line_qty !== undefined) bits.push(`min ${c.min_line_qty} units per line, ≤ ${c.max_suppliers} suppliers`);
  if (c.note) bits.push(c.note);
  if (c.reason) bits.push(c.reason);
  return `<div class="wf-cons">${wfBadge(c.status, wfResultKind(c.status))} <b>${name}</b> <span class="text-xs text-body">${esc(bits.join(" · "))}</span></div>`;
}

function traceAllocationTable(a) {
  return `<table class="table-clean w-full wf-inner"><thead><tr><th>Supplier</th><th>Allocated</th><th>Available</th><th>Unit cost</th><th>Line cost</th><th>Defect</th><th>Rank / score</th></tr></thead><tbody>
    ${a.lines.map((l) => `<tr><td>${esc(l.supplier_id)}${l.high_risk ? " " + wfBadge("High Risk", "fail") : ""}</td><td class="tabular">${fmtNum(l.allocated_qty)}</td><td class="tabular">${fmtNum(l.available_qty)}</td>
      <td class="tabular">${fmtCurrency(l.unit_cost)}</td><td class="tabular">${fmtCurrency(l.line_cost)}</td><td class="tabular">${(l.defect_rate * 100).toFixed(2)}%</td>
      <td class="tabular">#${l.heuristic_rank} · ${wfNum(l.heuristic_score, 1)}</td></tr>`).join("")}
    <tr class="font-semibold"><td>Total</td><td class="tabular">${fmtNum(a.total_quantity)}</td><td></td><td></td><td class="tabular">${fmtCurrency(a.total_cost)}</td><td class="tabular">${(a.weighted_defect_rate * 100).toFixed(2)}%</td><td class="tabular">score ${wfNum(a.weighted_score, 4)}</td></tr>
  </tbody></table>`;
}

function traceEventHtml(e) {
  const [label, kind] = TRACE_TYPE_LABEL[e.type] || [e.type, "neutral"];
  const combo = e.combination ? e.combination.key : "";
  const hasBody = e.capacity || e.quality || e.budget || e.min_allocation || e.pruning || e.solver || e.allocation || e.priority || e.pool;
  let body = "";
  if (hasBody) {
    body += `<div class="wf-cons-grid">`;
    if (e.capacity || e.quality || e.budget || e.min_allocation) {
      body += traceConstraint("Capacity", e.capacity) + traceConstraint("Quality", e.quality) + traceConstraint("Budget", e.budget)
        + traceConstraint("Minimum allocation / supplier count", e.min_allocation);
    }
    body += `</div>`;
    if (e.priority) body += `<div class="text-xs text-body">Queue priority: set size ${e.priority.set_size}, optimistic score bound ${wfNum(e.priority.optimistic_bound, 4)}${e.queue_size_after_pop !== undefined ? ` · queue size after pop ${e.queue_size_after_pop}` : ""}${e.queue_size_after_push !== undefined ? ` · queue size ${e.queue_size_after_push}` : ""}</div>`;
    if (e.pruning) body += `<div class="wf-banner wf-banner-warn text-xs"><b>Pruned (${esc(e.pruning.stage)}):</b> ${esc(e.pruning.reason)}. Incumbent key [high-risk, suppliers, −score]: ${esc(JSON.stringify(e.pruning.incumbent_key.map((x, i) => (i === 2 ? Number(Number(x).toFixed(4)) : x))))}, branch bound ${wfNum(e.pruning.bound, 4)}.</div>`;
    if (e.solver) body += `<div class="text-xs text-body"><b>Solver:</b> ${esc(e.solver.kind)} → ${esc(e.solver.status)} — ${esc(e.solver.detail)}. Inputs: required ${fmtNum(e.solver.inputs.required)}, min line ${e.solver.inputs.min_line_qty}, max defect ${(e.solver.inputs.max_defect_rate * 100).toFixed(0)}%, budget ${e.solver.inputs.budget == null ? "none" : fmtCurrency(e.solver.inputs.budget)}.</div>`;
    if (e.allocation) body += traceAllocationTable(e.allocation);
    if (e.objective) body += `<div class="text-xs text-body">Objective (lexicographic): High Risk suppliers <b>${e.objective.priority_1_high_risk_suppliers}</b> → suppliers used <b>${e.objective.priority_2_suppliers_used}</b> → weighted score <b>${wfNum(e.objective.priority_3_weighted_score, 4)}</b>${e.replaced ? ` · replaced ${esc(e.replaced.combination_key)}` : ""}</div>`;
    if (e.pool) body += `<div class="text-xs text-body">Pool in heuristic order: ${e.pool.map((p) => `#${p.heuristic_rank} ${esc(p.supplier_id)} (${fmtNum(p.available_qty)})`).join(", ")}. ${esc(e.search_strategy || "")}</div>`;
  }
  const summaryBlock = e.summary ? `<div class="text-xs text-body">${Object.entries(e.summary).filter(([k]) => k !== "search_limits").map(([k, v]) => `<span class="wf-chip">${esc(prettyKey(k.replace(/_/g, " ")))}: ${esc(String(v))}</span>`).join(" ")}</div>` : "";
  return `
    <details class="wf-ev" data-trace-kind="${TRACE_KIND[e.type] || "other"}">
      <summary>
        <span class="wf-step">${e.step}</span>
        ${wfBadge(label, kind)}
        ${combo ? `<span class="font-medium text-heading">${esc(combo)}</span>` : ""}
        ${e.outcome ? `<span class="text-xs text-muted">${esc(e.outcome)}</span>` : ""}
        <span class="text-xs text-body wf-flow">${esc(e.reason || e.message)}</span>
      </summary>
      <div class="wf-eval-body"><div class="text-xs text-body mb-2">${esc(e.message)}</div>${body}${summaryBlock}</div>
    </details>`;
}

function traceFinalHtml(t) {
  const f = t.final;
  const c = t.summary;
  const counters = ["nodes_created", "nodes_expanded", "combinations_evaluated", "solver_invocations", "combinations_rejected", "branches_pruned", "feasible_allocations"]
    .map((k) => `<div class="wf-counter"><div class="text-xs text-muted">${esc(prettyKey(k.replace(/_/g, " ")))}</div><div class="font-semibold text-heading tabular">${fmtNum(c[k])}</div></div>`).join("");
  if (!f) {
    return `<div class="wf-final"><h3 class="font-semibold text-heading mb-1">Search result — no feasible allocation</h3>
      <p class="text-xs text-body">${esc(c.termination_reason)}</p><div class="wf-counters">${counters}</div></div>`;
  }
  const passed = Object.entries(f.constraints_passed).map(([k, v]) => `${wfBadge(v ? "PASS" : "FAIL", v ? "pass" : "fail")} ${esc(prettyKey(k.replace(/_/g, " ")))}`).join(" ");
  return `<div class="wf-final">
    <h3 class="font-semibold text-heading mb-1">Final allocation — ${esc(c.result_type)}</h3>
    ${traceAllocationTable(f.allocation)}
    <div class="text-xs text-body mt-2">Objective: High Risk suppliers <b>${f.objective.priority_1_high_risk_suppliers}</b> → suppliers used <b>${f.objective.priority_2_suppliers_used}</b> → quantity-weighted score <b>${wfNum(f.objective.priority_3_weighted_score, 4)}</b>${f.shortfall ? ` · shortfall ${fmtNum(f.shortfall)} units` : ""}</div>
    <div class="text-xs mt-1">${passed}</div>
    <div class="text-xs text-body mt-2"><b>Why this allocation:</b> ${f.alternatives_evaluated
      ? `${f.alternatives_evaluated} other feasible combination${f.alternatives_evaluated === 1 ? " was" : "s were"} found and ranked below it. ${f.preferred_over.map((p) => `${esc(p.alternative)} lost on ${esc(p.decided_by)} (${esc(String(p.alternative_value))} vs ${esc(String(p.selected))})`).join("; ")}.`
      : "It was the only feasible combination, or every other feasible combination was pruned because it could not beat it."} ${esc(c.termination_reason)}</div>
    <div class="wf-counters">${counters}</div></div>`;
}

function searchTraceHtml(t, transition) {
  if (!t) return "";
  return `
    <div class="card p-5 wf-search-trace">
      <h2 class="font-semibold text-heading mb-1">Multi-Supplier Search Trace</h2>
      <div class="wf-banner wf-banner-warn mb-2"><b>${esc(transition || "No single supplier can fulfil the full order. Multi-supplier allocation has started.")}</b></div>
      <p class="text-xs text-body mb-3">Recorded by the search itself, in the order it ran. A <b>combination</b> is a set of suppliers; an <b>allocation</b> is a quantity split over that set.
        <i>Generated</i> = a set was created, <i>expanded</i> = taken from the queue and extended, <i>evaluated</i> = its allocation was attempted, <i>rejected</i> = failed a constraint,
        <i>pruned</i> = skipped because it cannot beat the best known plan, <i>feasible</i> = a valid allocation exists, <i>new best</i> = best so far, <i>final</i> = the plan selected.
        Constraints a combination never reached are shown as NOT YET EVALUATED, not as passed.</p>
      ${traceFinalHtml(t)}
      <div class="wf-filters" role="group" aria-label="Filter search trace">${TRACE_FILTERS.map(([k, l], i) => `<button type="button" class="wf-filter ${i === 0 ? "is-active" : ""}" data-trace-filter="${k}">${l}</button>`).join("")}
        <span class="text-xs text-muted wf-filter-count"></span></div>
      <div class="wf-events">${t.events.map(traceEventHtml).join("")}</div>
    </div>`;
}

function bindWorkflowViews(root) {
  const scope = root || document;
  scope.querySelectorAll("[data-trace-filter]").forEach((btn) => btn.addEventListener("click", () => {
    const kind = btn.dataset.traceFilter;
    scope.querySelectorAll("[data-trace-filter]").forEach((b) => b.classList.toggle("is-active", b === btn));
    let shown = 0;
    scope.querySelectorAll(".wf-events .wf-ev").forEach((el) => {
      const match = kind === "all" || el.dataset.traceKind === kind;
      el.hidden = !match;
      if (match) shown += 1;
    });
    const n = scope.querySelector(".wf-filter-count");
    if (n) n.textContent = `${shown} event${shown === 1 ? "" : "s"} shown`;
  }));
  const total = scope.querySelectorAll(".wf-events .wf-ev").length;
  const n = scope.querySelector(".wf-filter-count");
  if (n) n.textContent = `${total} events shown`;
}
