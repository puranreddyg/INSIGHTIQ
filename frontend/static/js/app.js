// InsightIQ SPA — vanilla JS, no build step. Views are rendered on demand
// into the #view-* containers; Chart.js instances are tracked in `charts`
// and destroyed before every re-render to avoid canvas-reuse errors.
//
// Theming: a single `.dark` class on <html> (toggled by #theme-toggle,
// persisted in localStorage, defaulting to the OS preference on first
// visit) drives every color via CSS custom properties in styles.css.
// Chart.js can't read CSS vars, so chartPalette() mirrors the same two
// token sets in JS and every chart is re-rendered on theme change.

const LIGHT_PALETTE = {
  series1: "#2a78d6", series2: "#eb6834", series3: "#1baf7a", series4: "#eda100",
  status: { good: "#0ca30c", warning: "#fab219", serious: "#ec835a", critical: "#d03b3b" },
  ink: "#10233f", inkSecondary: "#56647d", inkMuted: "#8996ac", grid: "#e2e9f5",
};
const DARK_PALETTE = {
  series1: "#3987e5", series2: "#d95926", series3: "#199e70", series4: "#c98500",
  status: { good: "#0ca30c", warning: "#fab219", serious: "#ec835a", critical: "#d03b3b" },
  ink: "#eef2fa", inkSecondary: "#a7b4cc", inkMuted: "#6f7f9c", grid: "#223050",
};
function isDarkMode() { return document.documentElement.classList.contains("dark"); }
function chartPalette() { return isDarkMode() ? DARK_PALETTE : LIGHT_PALETTE; }
function prefersReducedMotion() { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; }

// ---------------------------------------------------------------------------
// Chart helpers — gradient fills + staggered entrance, shared by every chart
// ---------------------------------------------------------------------------

function barGradient(colorHex, horizontal) {
  return (context) => {
    const { chart } = context;
    const { ctx, chartArea } = chart;
    if (!chartArea) return colorHex; // first layout pass has no chartArea yet
    const gradient = horizontal
      ? ctx.createLinearGradient(chartArea.left, 0, chartArea.right, 0)
      : ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
    gradient.addColorStop(0, colorHex + "99");
    gradient.addColorStop(1, colorHex);
    return gradient;
  };
}

function areaGradient(colorHex) {
  return (context) => {
    const { chart } = context;
    const { ctx, chartArea } = chart;
    if (!chartArea) return colorHex + "22";
    const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
    gradient.addColorStop(0, colorHex + "4d");
    gradient.addColorStop(1, colorHex + "03");
    return gradient;
  };
}

function barEntranceAnimation() {
  if (prefersReducedMotion()) return { duration: 0 };
  return { duration: 750, easing: "easeOutQuart" };
}

function lineEntranceAnimation() {
  if (prefersReducedMotion()) return { duration: 0 };
  return { duration: 900, easing: "easeOutQuart" };
}

function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

function animateCountUps(container) {
  const reduced = prefersReducedMotion();
  container.querySelectorAll("[data-count-to]").forEach((el) => {
    const target = Number(el.dataset.countTo);
    const suffix = el.dataset.countSuffix || "";
    const decimals = (el.dataset.countTo.split(".")[1] || "").length;
    const render = (v) => v.toLocaleString("en-IN", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix;
    if (reduced) { el.textContent = render(target); return; }
    const duration = 900;
    const start = performance.now();
    function tick(now) {
      const p = Math.min(1, (now - start) / duration);
      el.textContent = render(target * easeOutCubic(p));
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}

function staggerIn(container, selector) {
  if (prefersReducedMotion() || !window.Motion) return;
  container.querySelectorAll(selector).forEach((el, i) => {
    window.Motion.animate(
      el,
      { opacity: [0, 1], transform: ["translateY(10px)", "translateY(0px)"] },
      { duration: 0.4, delay: i * 0.06, easing: [0.22, 1, 0.36, 1] }
    );
  });
}

const state = {
  view: "dashboard",
  dashboardData: null,
  skusCache: [],
  selectedSkuId: null,
  lastPipeline: null,
  lastSuppliers: null,
  approvedThisSession: null,
};

const charts = {};
function destroyChart(key) {
  if (charts[key]) { charts[key].destroy(); delete charts[key]; }
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

const INR_FORMATTER = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 2 });
function fmtCurrency(n) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  return INR_FORMATTER.format(Number(n));
}
function fmtNum(n) { return n === null || n === undefined ? "—" : Number(n).toLocaleString("en-IN"); }
function fmtDate(s) {
  if (!s) return "—";
  const d = new Date(s);
  if (isNaN(d)) return s;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function toast(message, type = "info") {
  const root = document.getElementById("toast-root");
  const colors = { info: "bg-[var(--accent)]", success: "bg-[var(--success)]", error: "bg-[var(--danger)]" };
  const div = document.createElement("div");
  div.className = `toast-item ${colors[type] || colors.info} text-white text-sm font-medium px-4 py-3 rounded-xl max-w-sm`;
  div.textContent = message;
  root.appendChild(div);
  setTimeout(() => { div.style.opacity = "0"; div.style.transition = "opacity .3s"; setTimeout(() => div.remove(), 300); }, 3500);
}

function statusBadgeClass(status) {
  const map = {
    "Preferred": "badge-preferred", "High Risk": "badge-risk",
    "Standard": "badge-standard", "Reliable": "badge-reliable",
    "Ineligible": "badge-ineligible",
  };
  return map[status] || "badge-standard";
}

// Marks a value that comes from the simulated supplier-SKU profile, not observed data.
function simBadge() {
  return `<span class="badge-sim" title="Simulated — DataSets/simulated/README.md">SIM</span>`;
}

async function ensureSkus() {
  if (!state.skusCache.length) {
    try { state.skusCache = (await Api.skus()).skus; } catch (_) {}
  }
  return state.skusCache;
}

function defaultSkuId() {
  return state.selectedSkuId ?? state.skusCache.find((s) => s.urgent)?.sku_id ?? state.skusCache[0]?.sku_id;
}

function skuOptionsHtml(selected) {
  return state.skusCache.map((s) =>
    `<option value="${esc(s.sku_id)}" ${s.sku_id === selected ? "selected" : ""}>${s.urgent ? "⚠ " : ""}${esc(s.sku_id)} · stock ${s.inventory}/${s.reorder_point}${s.urgent ? ` · need ${s.required_quantity}` : ""}</option>`
  ).join("");
}

// ---------------------------------------------------------------------------
// Theme toggle
// ---------------------------------------------------------------------------

function setTheme(dark, persist = true) {
  document.documentElement.classList.toggle("dark", dark);
  if (persist) {
    try { localStorage.setItem("insightiq-theme", dark ? "dark" : "light"); } catch (e) {}
  }
  // Chart.js bakes colors into the canvas at render time, so re-paint
  // whatever view is currently visible with the new palette.
  rerenderCurrentView();
}

document.getElementById("theme-toggle").addEventListener("click", () => setTheme(!isDarkMode()));

function rerenderCurrentView() {
  if (state.view === "dashboard" && state.dashboardData) renderDashboard(state.dashboardData);
  else if (state.view === "suppliers" && state.lastSuppliers) renderSuppliersView(state.lastSuppliers);
  else if (state.view === "pipeline" && state.lastPipeline) renderPipelineResult(state.lastPipeline);
  else if (state.view === "action") loadActionView();
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

function positionNavPill(immediate) {
  const nav = document.getElementById("main-nav");
  const pill = document.getElementById("nav-indicator");
  const activeBtn = nav.querySelector(".nav-btn.active");
  if (!nav || !pill || !activeBtn) return;
  const navRect = nav.getBoundingClientRect();
  const btnRect = activeBtn.getBoundingClientRect();
  const x = btnRect.left - navRect.left;
  const w = btnRect.width;

  if (!immediate && window.Motion && !prefersReducedMotion()) {
    window.Motion.animate(pill, { transform: `translateX(${x}px)`, width: `${w}px` }, { type: "spring", bounce: 0, duration: 0.4 });
  } else {
    pill.style.transition = "none";
    pill.style.transform = `translateX(${x}px)`;
    pill.style.width = `${w}px`;
  }
}

function animateViewIn(panel) {
  if (window.Motion && !prefersReducedMotion()) {
    window.Motion.animate(panel, { opacity: [0, 1], transform: ["translateY(6px)", "translateY(0px)"] }, { duration: 0.35, easing: [0.22, 1, 0.36, 1] });
  }
}

function switchView(view, opts = {}) {
  state.view = view;
  document.querySelectorAll(".view-panel").forEach((p) => p.classList.add("hidden"));
  const panel = document.getElementById(`view-${view}`);
  panel.classList.remove("hidden");
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  positionNavPill(!!opts.immediate);
  if (!opts.immediate) animateViewIn(panel);

  if (view === "dashboard") loadDashboard();
  if (view === "pipeline") loadPipelineView(opts.skuId);
  if (view === "suppliers") loadSuppliersView();
  if (view === "action") loadActionView();
}

document.getElementById("main-nav").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-view]");
  if (btn) switchView(btn.dataset.view);
});
document.querySelectorAll(".md\\:hidden [data-view]").forEach((btn) =>
  btn.addEventListener("click", () => switchView(btn.dataset.view))
);
document.getElementById("alerts-btn").addEventListener("click", () => switchView("dashboard"));
window.addEventListener("resize", () => positionNavPill(true));

// ---------------------------------------------------------------------------
// System health / nav strip
// ---------------------------------------------------------------------------

async function refreshSystemHealth() {
  try {
    const health = await Api.systemHealth();
    document.getElementById("alert-count").textContent = health.active_alerts;
    document.getElementById("health-dot").className = "w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse";
    renderAgentStrip(health.agents, "idle");
  } catch (e) {
    document.getElementById("health-dot").className = "w-2.5 h-2.5 rounded-full bg-red-400";
  }
}

function renderAgentStrip(agents, forcedStatus) {
  const strip = document.getElementById("agent-status-strip");
  const dotColor = { idle: "bg-slate-400", active: "bg-amber-400 animate-pulse", completed: "bg-emerald-400" };
  strip.innerHTML = agents.map((a) => {
    const st = forcedStatus || a.status;
    return `<span class="pulse-dot ${dotColor[st] || dotColor.idle}" title="${esc(a.name)}: ${st}"></span>`;
  }).join("");
}

// ===========================================================================
// VIEW 1 — Executive Enterprise Dashboard
// ===========================================================================

async function loadDashboard() {
  const root = document.getElementById("view-dashboard");
  root.innerHTML = dashboardSkeleton();
  try {
    const data = await Api.dashboard();
    state.dashboardData = data;
    renderDashboard(data);
  } catch (e) {
    root.innerHTML = `<div class="card p-8 text-center text-danger">Failed to load dashboard: ${esc(e.message)}</div>`;
  }
}

function dashboardSkeleton() {
  const kpi = `<div class="kpi-card"><div class="skeleton h-3 w-24 mb-3"></div><div class="skeleton h-8 w-16"></div></div>`;
  return `
    <div class="space-y-6">
      <div class="skeleton h-7 w-72"></div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">${kpi.repeat(4)}</div>
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div class="card p-5 lg:col-span-2"><div class="skeleton h-64 w-full"></div></div>
        <div class="card p-5"><div class="skeleton h-64 w-full"></div></div>
      </div>
      <div class="card p-5"><div class="skeleton h-80 w-full"></div></div>
    </div>`;
}

const KPI_ICONS = {
  urgent: `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
  info: `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3h18v13H3z"/><path d="M7 21h10M9 16v5M15 16v5"/></svg>`,
  neutral: `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>`,
  good: `<svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 3 6v6c0 5 4 8.5 9 10 5-1.5 9-5 9-10V6z"/></svg>`,
};

function renderDashboard(data) {
  const root = document.getElementById("view-dashboard");
  const k = data.kpis;

  root.innerHTML = `
    <div class="fade-in space-y-6">
      <div>
        <h1 class="page-title">Executive Enterprise Dashboard</h1>
        <p class="text-sm text-body mt-1">Supply-chain snapshot as of ${fmtDate(data.snapshot_date)} · generated ${new Date(data.generated_at).toLocaleTimeString()}</p>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        ${kpiCard("Critical Low Stock SKUs", fmtNum(k.critical_low_stock_skus), "urgent", "SKUs with a warehouse below reorder point")}
        ${kpiCard("Active Purchase Orders", fmtNum(k.active_purchase_orders), "info", "Approved & in flight")}
        ${kpiCard("Avg Supplier Lead Time", `${k.avg_supplier_lead_time_days ?? "—"}d`, "neutral", "Across all SKU-warehouse lanes")}
        ${kpiCard("Avg Supplier Fulfillment", `${k.avg_supplier_fulfillment_rate ?? "—"}%`, "good", "Simulated supplier-level rate")}
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div class="card hoverable p-5 lg:col-span-2">
          <h2 class="font-semibold text-heading mb-1">Top SKUs by Required Quantity</h2>
          <p class="text-xs text-body mb-3">Sum of each urgent warehouse's median historical order size</p>
          <div class="h-72"><canvas id="chart-deficit"></canvas></div>
        </div>
        <div class="card hoverable p-5">
          <h2 class="font-semibold text-heading mb-3">Recent Purchase Orders</h2>
          <div id="recent-po-list" class="space-y-3 max-h-72 overflow-y-auto pr-1"></div>
        </div>
      </div>

      <div class="card p-0 overflow-hidden">
        <div class="flex items-center justify-between px-5 py-4 border-b border-soft">
          <div>
            <h2 class="font-semibold text-heading">Live Alert Table</h2>
            <p class="text-xs text-body">SKUs with at least one warehouse where Inventory &lt; Reorder Point</p>
          </div>
          <span class="badge badge-urgent badge-live">${data.alerts.length} active alerts</span>
        </div>
        <div class="overflow-x-auto">
          <table class="table-clean w-full">
            <thead><tr>
              <th>SKU</th><th>Urgent Warehouses</th><th>Inventory</th><th>Reorder Pt</th><th>Shortfall</th><th>Required Qty</th><th>Candidate Suppliers</th><th></th>
            </tr></thead>
            <tbody>${data.alerts.map(alertRow).join("")}</tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  document.getElementById("recent-po-list").innerHTML = data.recent_purchase_orders.length
    ? data.recent_purchase_orders.map((po) => `
        <div class="border border-soft rounded-lg p-3 text-sm">
          <div class="flex justify-between font-medium text-heading"><span>${esc(po.sku_id || po.product_name)}</span><span class="tabular">${fmtCurrency(po.total_cost)}</span></div>
          <div class="text-xs text-body mt-1">${esc(po.supplier_name)} · ${po.quantity} units · ETA ${fmtDate(po.estimated_arrival)}</div>
        </div>`).join("")
    : `<div class="text-sm text-muted text-center py-8">No purchase orders approved yet.</div>`;

  root.querySelectorAll("[data-trigger-sku]").forEach((btn) =>
    btn.addEventListener("click", () => switchView("pipeline", { skuId: btn.dataset.triggerSku }))
  );

  renderDeficitChart(data.alerts);
  staggerIn(root, ".kpi-card");
  staggerIn(root, ".grid.lg\\:grid-cols-3 > .card");
  animateCountUps(root);
}

function kpiCard(label, value, tone, sub) {
  const toneStyle = {
    urgent: { text: "text-danger", tone: "var(--danger)" },
    info: { text: "text-info", tone: "var(--info)" },
    neutral: { text: "text-heading", tone: "var(--accent)" },
    good: { text: "text-success", tone: "var(--success)" },
  }[tone] || { text: "text-heading", tone: "var(--accent)" };
  const numericTarget = Number(String(value).replace(/[^0-9.-]/g, ""));
  const suffix = String(value).replace(/^[\s\S]*?[0-9.,]+/, "").trim();
  const isCountable = !isNaN(numericTarget) && /^-?[\d,]+(\.\d+)?/.test(String(value).trim());
  return `
    <div class="kpi-card" style="--tone-color:${toneStyle.tone}">
      <div class="flex items-start justify-between">
        <div class="text-xs font-semibold text-muted uppercase tracking-wide">${esc(label)}</div>
        <span class="kpi-icon">${KPI_ICONS[tone] || KPI_ICONS.neutral}</span>
      </div>
      <div class="text-3xl font-bold ${toneStyle.text} mt-2 tabular" ${isCountable ? `data-count-to="${numericTarget}" data-count-suffix="${esc(suffix)}"` : ""}>${isCountable ? "0" + suffix : value}</div>
      <div class="text-xs text-muted mt-1">${esc(sub)}</div>
    </div>`;
}

function alertRow(a) {
  return `
    <tr>
      <td class="font-mono text-xs font-semibold text-heading">${esc(a.sku_id)}</td>
      <td class="text-body">${esc(a.urgent_warehouses.join(", "))}</td>
      <td class="tabular">${fmtNum(a.inventory)}</td>
      <td class="tabular">${fmtNum(a.reorder_point)}</td>
      <td class="tabular">${fmtNum(a.shortfall)}</td>
      <td><span class="badge badge-urgent tabular">${fmtNum(a.required_quantity)}</span></td>
      <td class="tabular">${fmtNum(a.candidate_suppliers)}</td>
      <td><button data-trigger-sku="${esc(a.sku_id)}" class="btn-primary !py-1.5 !px-3 text-xs">Trigger AI Procurement</button></td>
    </tr>`;
}

function renderDeficitChart(alerts) {
  destroyChart("deficit");
  const pal = chartPalette();
  const top = alerts.slice(0, 8);
  const ctx = document.getElementById("chart-deficit");
  charts.deficit = new Chart(ctx, {
    type: "bar",
    data: {
      labels: top.map((a) => a.sku_id),
      datasets: [{
        label: "Required quantity (units)",
        data: top.map((a) => a.required_quantity),
        backgroundColor: barGradient(pal.series1, true),
        hoverBackgroundColor: pal.series1,
        borderRadius: 4,
        maxBarThickness: 28,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: barEntranceAnimation(),
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const a = top[ctx.dataIndex];
              return `${a.required_quantity} units needed (${a.urgent_warehouses.join(", ")} below reorder point)`;
            },
          },
        },
      },
      scales: {
        x: { grid: { color: pal.grid }, ticks: { color: pal.inkMuted } },
        y: { grid: { display: false }, ticks: { color: pal.inkSecondary } },
      },
    },
  });
}

// ===========================================================================
// VIEW 2 — Interactive Multi-Agent Pipeline & XAI Inspector
// ===========================================================================

async function loadPipelineView(prefillSkuId) {
  const root = document.getElementById("view-pipeline");
  await ensureSkus();
  const sid = prefillSkuId ?? defaultSkuId();
  state.selectedSkuId = sid;

  root.innerHTML = `
    <div class="fade-in space-y-6">
      <div class="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 class="page-title">Multi-Agent Pipeline &amp; XAI Inspector</h1>
          <p class="text-sm text-body mt-1">Perception → Supplier Intelligence → Knowledge → Planning</p>
        </div>
        <div class="flex items-center gap-2">
          <select id="sku-select" class="input-field min-w-[260px]"></select>
          <button id="run-pipeline-btn" class="btn-primary">Run Agent Pipeline</button>
        </div>
      </div>

      <div id="agent-flow" class="grid grid-cols-1 md:grid-cols-4 gap-4"></div>

      <div id="pipeline-result"></div>
    </div>
  `;

  const select = document.getElementById("sku-select");
  select.innerHTML = skuOptionsHtml(sid);

  const stages = ["Perception Agent", "Supplier Intelligence Agent", "Knowledge Agent", "Planning Agent"];
  document.getElementById("run-pipeline-btn").addEventListener("click", () => runPipeline(select.value));

  // Only auto-run when the user explicitly asked for it (clicked "Trigger AI
  // Procurement" on the dashboard, which passes prefillSkuId). Simply
  // opening this tab from the nav bar must never fire a procurement run on
  // its own — it should sit idle until "Run Agent Pipeline" is clicked.
  if (prefillSkuId) {
    renderAgentFlow(stages, {});
    runPipeline(prefillSkuId);
  } else if (state.lastPipeline && state.lastPipeline.sku_id === sid) {
    const completed = Object.fromEntries(stages.map((s) => [s, "completed"]));
    renderAgentFlow(stages, completed);
    stages.forEach((s, i) => {
      const el = document.querySelector(`[data-agent-card="${s}"] [data-agent-summary]`);
      if (el) el.textContent = state.lastPipeline.stage_log[i]?.summary || "";
    });
    renderPipelineResult(state.lastPipeline);
  } else {
    renderAgentFlow(stages, {});
    document.getElementById("pipeline-result").innerHTML = `
      <div class="card p-8 text-center mt-2">
        <p class="text-sm text-body">Pick a SKU above and click <span class="text-heading font-semibold">Run Agent Pipeline</span> to generate a procurement plan.</p>
      </div>`;
  }
}

function agentIcon(name) {
  const icons = {
    "Perception Agent": "👁", "Supplier Intelligence Agent": "🔍",
    "Knowledge Agent": "🧠", "Planning Agent": "📅",
  };
  return icons[name] || "●";
}

function renderAgentFlow(stages, statusMap) {
  const flow = document.getElementById("agent-flow");
  flow.innerHTML = stages.map((name, i) => {
    const status = statusMap[name] || "idle";
    return `
      <div class="relative">
        <div class="agent-card state-${status}" data-agent-card="${esc(name)}">
          <div class="flex items-center justify-between">
            <span class="agent-icon-wrap">${agentIcon(name)}</span>
            <span class="badge badge-${status}">${status}</span>
          </div>
          <div class="font-semibold text-heading mt-2 text-sm">${esc(name)}</div>
          <div class="text-xs text-body mt-1 min-h-[2.2em]" data-agent-summary></div>
        </div>
        ${i < stages.length - 1 ? '<div class="hidden md:block absolute top-1/2 -right-2 -translate-y-1/2 text-muted text-xl">→</div>' : ""}
      </div>`;
  }).join("");
}

async function runPipeline(skuId) {
  if (!skuId) return;
  state.selectedSkuId = skuId;
  const btn = document.getElementById("run-pipeline-btn");
  btn.disabled = true;
  btn.textContent = "Running…";

  const stages = ["Perception Agent", "Supplier Intelligence Agent", "Knowledge Agent", "Planning Agent"];
  renderAgentFlow(stages, {});
  renderAgentStrip(stages.map((n) => ({ name: n, status: "idle" })), null);

  try {
    const result = await Api.runPipeline(skuId);
    state.lastPipeline = result;

    // Replay stage completion sequentially for a legible pipeline animation —
    // the computation already finished server-side; this only paces the reveal.
    const statusMap = {};
    for (let i = 0; i < stages.length; i++) {
      statusMap[stages[i]] = "active";
      renderAgentFlow(stages, statusMap);
      await new Promise((r) => setTimeout(r, 350));
      statusMap[stages[i]] = "completed";
      renderAgentFlow(stages, statusMap);
      const card = document.querySelector(`[data-agent-card="${stages[i]}"] [data-agent-summary]`);
      if (card) card.textContent = result.stage_log[i]?.summary || "";
    }

    renderPipelineResult(result);
    toast("Pipeline completed successfully.", "success");
  } catch (e) {
    document.getElementById("pipeline-result").innerHTML = `<div class="card p-6 text-danger">Pipeline failed: ${esc(e.message)}</div>`;
    toast("Pipeline run failed.", "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Agent Pipeline";
  }
}

function splitAllocationsTable(lines) {
  return `
    <div class="overflow-x-auto mt-3">
      <table class="table-clean w-full">
        <thead><tr><th>Supplier</th><th>Quantity</th><th>Unit Cost</th><th>Cost</th><th>Arrives</th><th>Status</th></tr></thead>
        <tbody>${lines.map((l) => `
          <tr>
            <td class="font-medium text-heading">${esc(l.supplier_id)}</td>
            <td class="tabular">${fmtNum(l.quantity)} <span class="text-muted">/ ${fmtNum(l.available_qty)}${simBadge()}</span></td>
            <td class="tabular">${fmtCurrency(l.unit_cost)}</td>
            <td class="tabular">${fmtCurrency(l.expense)}</td>
            <td>${fmtDate(l.estimated_arrival_date)}</td>
            <td><span class="badge ${statusBadgeClass(l.supplier_status)}">${esc(l.supplier_status)}</span></td>
          </tr>`).join("")}</tbody>
      </table>
    </div>`;
}

function renderPipelineResult(result) {
  const container = document.getElementById("pipeline-result");
  const focus = result.focus_fact;
  const plan = result.planning;
  const rec = result.knowledge.recommended_supplier;
  const sup = result.supplier;
  const urgentWh = focus ? focus.lanes.filter((l) => l.Urgent).map((l) => l.warehouse_id) : [];

  container.innerHTML = `
    <div class="fade-in grid grid-cols-1 lg:grid-cols-2 gap-5 mt-2">
      <div class="card hoverable p-5">
        <h2 class="font-semibold text-heading mb-3">Perceived State — ${focus ? esc(focus.SKU) : "N/A"}</h2>
        ${focus ? `
          <div class="grid grid-cols-2 gap-3 text-sm">
            <div><div class="text-xs text-muted">Snapshot Date</div><div class="font-medium text-heading">${fmtDate(focus.snapshot_date)}</div></div>
            <div><div class="text-xs text-muted">Urgent Warehouses</div><div class="font-medium text-heading">${urgentWh.length ? esc(urgentWh.join(", ")) : `none of ${focus.lane_count}`}</div></div>
            <div><div class="text-xs text-muted">Inventory (${esc(focus.focus_scope)})</div><div class="font-medium text-heading tabular">${fmtNum(focus.Inventory)}</div></div>
            <div><div class="text-xs text-muted">Reorder Point</div><div class="font-medium text-heading tabular">${fmtNum(focus.ReorderPoint)}</div></div>
            <div><div class="text-xs text-muted">Required Quantity</div><div class="font-medium text-danger tabular">${fmtNum(focus.RequiredQuantity)}</div></div>
            <div><div class="text-xs text-muted">Status</div><div>${focus.Urgent ? '<span class="badge badge-urgent badge-live">Urgent Procurement</span>' : '<span class="badge badge-standard">Monitoring</span>'}</div></div>
          </div>` : `<p class="text-sm text-muted">No focus SKU.</p>`}

        <h3 class="font-semibold text-heading mt-5 mb-2 text-sm">${plan?.split_order ? "Split Order" : "Recommended Supplier"}</h3>
        ${plan?.split_order ? `
          <div class="border border-soft rounded-lg p-3 text-sm">
            <div class="font-medium text-heading">Order split across ${plan.allocations.length} suppliers</div>
            <div class="text-xs text-body mt-1">No single supplier can supply ${fmtNum(sup.required_quantity)} units alone (combined availability ${fmtNum(sup.combined_available_qty)}${simBadge()}). The order is filled from the best-ranked suppliers in turn; High Risk suppliers are used only where the others cannot cover it.</div>
            ${splitAllocationsTable(plan.allocations)}
          </div>` : rec ? `
          <div class="border border-soft rounded-lg p-3 text-sm flex items-center justify-between">
            <div>
              <div class="font-medium text-heading">${esc(rec.supplier_id)}</div>
              <div class="text-xs text-body">${fmtCurrency(rec.unit_cost)}/unit · ${rec.lead_time_days}d lead · ${fmtNum(rec.available_qty)} available${simBadge()} · Score ${rec.heuristic_score?.toFixed?.(1) ?? rec.heuristic_score}</div>
            </div>
            <span class="badge ${statusBadgeClass(rec.supplier_status)}">${esc(rec.supplier_status)}</span>
          </div>` : sup.no_eligible_supplier ? `
          <div class="border border-soft rounded-lg p-3 text-sm">
            <div class="font-medium text-danger">No supplier selected</div>
            <div class="text-xs text-body mt-1">None of the ${sup.pool_size} candidates can supply ${fmtNum(sup.required_quantity)} units alone (combined availability ${fmtNum(sup.combined_available_qty)}${simBadge()}). Even a split order across suppliers could not cover it.</div>
          </div>` : `<p class="text-sm text-muted">No eligible supplier.</p>`}

        ${plan && plan.error ? `<p class="text-sm text-body mt-4">${esc(plan.error)}</p>` : ""}

        ${plan && !plan.error ? `
        <h3 class="font-semibold text-heading mt-5 mb-2 text-sm">Plan Snapshot</h3>
        <div class="grid grid-cols-2 gap-3 text-sm">
          <div><div class="text-xs text-muted">Order Qty</div><div class="font-medium text-heading tabular">${fmtNum(plan.order_quantity)}</div></div>
          <div><div class="text-xs text-muted">${plan.split_order ? "Order Complete By" : "Est. Arrival"}</div><div class="font-medium text-heading">${fmtDate(plan.estimated_arrival_date)}</div></div>
          <div><div class="text-xs text-muted">Total Expense</div><div class="font-medium text-heading tabular">${fmtCurrency(plan.total_expense)}</div></div>
          <div><div class="text-xs text-muted">Backup Supplier</div><div class="font-medium text-heading">${esc(plan.backup_supplier_id || "—")}</div></div>
        </div>
        <button id="go-to-action" class="btn-primary w-full mt-4">Open in Action Hub →</button>
        ` : ""}
      </div>

      <div class="card p-5">
        <h2 class="font-semibold text-heading mb-1">Explainable AI (XAI) — Forward Chaining Trace</h2>
        <p class="text-xs text-body mb-3">Watch the Knowledge Agent's rule-by-rule reasoning play out, step by step.</p>
        ${renderXaiPanel()}
      </div>
    </div>
  `;

  document.getElementById("go-to-action")?.addEventListener("click", () => switchView("action"));
  initXaiPlayer(result);
}

// ---------------------------------------------------------------------------
// XAI trace — animated forward-chaining player.
//
// The Knowledge Agent (backend/agents/knowledge_agent.py) is the sole source
// of truth: every value, condition, operator, threshold, matched flag and
// outcome rendered below is read directly off the structured `reasoning_trace`
// the backend returns. Nothing here recomputes a rule or a threshold — this
// layer only sequences and animates the reveal of facts the agent already
// decided.
//
// Player scope: to keep a single guided run legible, the animated sequence
// covers the focus SKU plus the *recommended* supplier (the one that actually
// drove the downstream plan). Any other ranked candidates the Knowledge Agent
// also evaluated are listed underneath as fully-detailed, click-to-expand
// cards — same real data, just not force-animated one after another.
// ---------------------------------------------------------------------------

const XAI_ICONS = {
  sku: `<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="M3.3 7 12 12l8.7-5M12 22V12"/></svg>`,
  supplier: `<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 17V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v10a1 1 0 0 0 1 1h2"/><path d="M15 17H9M19 17h2a1 1 0 0 0 1-1v-3.4a1 1 0 0 0-.22-.63L18.7 8.4A1 1 0 0 0 18 8h-4"/><circle cx="17" cy="17" r="2"/><circle cx="7" cy="17" r="2"/></svg>`,
  generic: `<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/></svg>`,
};

function prettyKey(key) {
  return key.replace(/([a-z0-9])([A-Z])/g, "$1 $2");
}

function xaiEntityKind(entity) {
  if (!entity) return "generic";
  if (entity.startsWith("SKU")) return "sku";
  if (entity.startsWith("SUP")) return "supplier";
  return "generic";
}

function decisionBadgeClass(value) {
  if (value === "Urgent Procurement") return "badge-urgent";
  if (value === "Monitoring") return "badge-standard";
  return statusBadgeClass(value);
}

// Groups the flat fact/rule/decision trace into per-entity blocks (one SKU
// block, one block per evaluated supplier) — same grouping the backend's own
// sequential fact→rules→decision emission order already implies.
function buildXaiBlocks(trace) {
  const blocks = [];
  const notes = [];
  let current = null;
  for (const item of trace) {
    if (item.type === "fact") {
      current = {
        entity: item.entity,
        entityKind: item.entityKind || xaiEntityKind(item.entity),
        values: item.values || {},
        rules: [],
        decision: null,
      };
      blocks.push(current);
    } else if (item.type === "rule" && current) {
      current.rules.push(item);
    } else if (item.type === "decision") {
      if (current && item.entity === current.entity && !current.decision) current.decision = item;
      else notes.push(item);
    }
  }
  return { blocks, notes };
}

// ---- small presentational helpers (formatting only — no thresholds/logic) --

function xaiConditionMathText(cond) {
  if (cond.label === "Prior Status") return cond.result ? "No prior status recorded" : "Prior status already set";
  return cond.expression;
}

function xaiVerdictText(rule) {
  if (rule.elseText) return rule.matched ? "THEN branch executed" : "ELSE branch executed";
  return rule.matched ? "RULE PASSED" : "RULE FAILED";
}

function xaiStepKindLabel(step) {
  const map = { fact: "Fact", "rule-intro": "Rule", condition: "Condition", logic: "Logic", "rule-result": "Result", decision: "Decision", summary: "Summary" };
  return map[step.kind] || "Step";
}

function xaiStatusCaption(step) {
  if (!step) return "Ready — press Play to begin.";
  switch (step.kind) {
    case "fact": return `Reading facts for ${step.block.entity}…`;
    case "rule-intro": return `Evaluating Rule ${step.rule.ruleId} — ${step.rule.name}…`;
    case "condition": return `Checking ${step.condition.label}… ${xaiConditionMathText(step.condition)} → ${step.condition.result ? "TRUE ✓" : "FALSE ✗"}`;
    case "logic": return `Combining conditions with ${step.rule.logic}…`;
    case "rule-result": return step.rule.matched ? `✓ Rule ${step.rule.ruleId} matched.` : `Rule ${step.rule.ruleId} not matched.`;
    case "decision": return `✓ Decision made for ${step.block.entity}.`;
    case "summary": return "Forward chaining complete.";
    default: return "";
  }
}

// ---- card builders --------------------------------------------------------
// Each takes an optional `stepsRef` array: when provided, the piece registers
// itself as an animation step (pushed in traversal order) and renders hidden
// until the player reveals it; when null, the piece renders fully visible —
// used for the static "other suppliers" list, which shows complete real data
// without the timed walkthrough.

function xaiFactCardHtml(block, stepsRef) {
  const stepIdx = stepsRef ? stepsRef.push({ kind: "fact", block }) - 1 : null;
  const stepAttrs = stepIdx == null ? "" : `data-step-idx="${stepIdx}"`;
  const stepClass = stepIdx == null ? "" : "xai-step";
  const valuesHtml = Object.entries(block.values || {}).map(([k, v]) => `
    <span class="xai-fact-value-chip"><span class="k">${esc(prettyKey(k))}</span><span class="v">${esc(String(v))}</span></span>
  `).join("");
  return `
    <div class="xai-fact-card ${stepClass}" ${stepAttrs}>
      <div class="xai-fact-kicker">${XAI_ICONS[block.entityKind] || XAI_ICONS.generic} Fact Detected</div>
      <div class="xai-fact-entity">${esc(block.entity)}</div>
      <div class="xai-fact-values">${valuesHtml}</div>
    </div>`;
}

function xaiConditionRowHtml(cond, index, total, stepIdx) {
  const cls = cond.result ? "cond-true" : "cond-false";
  const conditionName = cond.label === "Prior Status" ? "No Prior Status" : `${cond.label} ${cond.operator} ${cond.threshold}`;
  const label = total > 1 ? `Condition ${index + 1} — ${esc(conditionName)}` : esc(cond.label);
  const stepAttrs = stepIdx == null ? "" : `data-step-idx="${stepIdx}"`;
  const stepClass = stepIdx == null ? "" : "xai-step";
  return `
    <div class="xai-condition-row ${cls} ${stepClass}" ${stepAttrs}>
      <div class="xai-condition-label">${label}</div>
      <div class="xai-condition-math">
        <span class="xai-condition-expr">${esc(xaiConditionMathText(cond))}</span>
        <span class="xai-condition-result">${cond.result ? "TRUE ✓" : "FALSE ✗"}</span>
      </div>
    </div>`;
}

function xaiLogicHtml(rule, stepIdx) {
  const expr = rule.conditions.map((c) => (c.result ? "TRUE" : "FALSE")).join(` ${rule.logic} `);
  const verdict = xaiVerdictText(rule);
  const cls = rule.matched ? "is-pass" : "is-fail";
  const stepAttrs = stepIdx == null ? "" : `data-step-idx="${stepIdx}"`;
  const stepClass = stepIdx == null ? "" : "xai-step";
  return `
    <div class="xai-logic-combine ${cls} ${stepClass}" ${stepAttrs}>
      <div class="expr">${esc(expr)}</div>
      <div class="arrow">↓</div>
      <div class="verdict">${esc(verdict)}</div>
    </div>`;
}

function xaiRuleResultHtml(rule, stepIdx) {
  const stepAttrs = stepIdx == null ? "" : `data-step-idx="${stepIdx}"`;
  const stepClass = stepIdx == null ? "" : "xai-step";
  const outcomeText = rule.matched ? rule.thenText : rule.elseText;
  const bannerCls = rule.matched ? "st-matched" : "st-not-matched";
  const bannerLabel = rule.matched ? `✓ RULE ${rule.ruleId} MATCHED` : `Rule ${rule.ruleId} → NOT MATCHED`;
  return `
    <div class="xai-rule-result ${stepClass}" ${stepAttrs}>
      ${outcomeText ? `<div class="xai-rule-outcome">${esc(outcomeText)}</div>` : ""}
      <div class="xai-rule-result-banner ${bannerCls}">${bannerLabel}</div>
    </div>`;
}

function xaiRuleCardHtml(block, rule, stepsRef) {
  const animated = !!stepsRef;
  const push = (kind, extra) => (animated ? stepsRef.push({ kind, block, rule, ...extra }) - 1 : null);

  const introIdx = push("rule-intro");
  const condsHtml = rule.conditions.map((cond, ci) => {
    const condIdx = push("condition", { condition: cond });
    return xaiConditionRowHtml(cond, ci, rule.conditions.length, condIdx);
  }).join(rule.conditions.length > 1 ? `<div class="xai-and-divider"><span>${esc(rule.logic)}</span></div>` : "");

  let logicHtml = "";
  if (rule.conditions.length > 1) {
    const logicIdx = push("logic");
    logicHtml = xaiLogicHtml(rule, logicIdx);
  }

  const resultIdx = push("rule-result");
  const resultHtml = xaiRuleResultHtml(rule, resultIdx);

  const cardStepAttrs = introIdx == null ? "" : `data-step-idx="${introIdx}"`;
  const cardStepClass = introIdx == null ? "" : "xai-step";
  const tag = animated ? "button" : "div";
  const badge = animated
    ? { text: "Pending", cls: "" }
    : { text: rule.matched ? "Matched" : "Not Matched", cls: rule.matched ? "st-matched" : "st-not-matched" };

  return `
    <div class="xai-rule-card ${cardStepClass} ${rule.matched ? "is-matched" : "is-not-matched"}" ${cardStepAttrs}>
      <${tag} class="xai-rule-header" ${animated ? `type="button" data-jump-step="${resultIdx}"` : ""}>
        <span class="xai-rule-num">R${rule.ruleId}</span>
        <span class="xai-rule-title">Rule ${rule.ruleId} — ${esc(rule.name)}</span>
        <span class="xai-rule-status-badge ${badge.cls}">${badge.text}</span>
      </${tag}>
      <div class="xai-rule-body">
        <div class="xai-if-then">${esc(rule.ifText)}\n${esc(rule.thenText)}${rule.elseText ? "\n" + esc(rule.elseText) : ""}</div>
        <div class="xai-conditions">${condsHtml}</div>
        ${logicHtml}
        ${resultHtml}
      </div>
    </div>`;
}

function xaiDecisionCardHtml(block, stepsRef) {
  const stepIdx = stepsRef ? stepsRef.push({ kind: "decision", block }) - 1 : null;
  const stepAttrs = stepIdx == null ? "" : `data-step-idx="${stepIdx}"`;
  const stepClass = stepIdx == null ? "" : "xai-step";
  const value = block.decision?.value ?? "—";
  return `
    <div class="xai-decision-card ${stepClass}" ${stepAttrs}>
      <div class="xai-decision-kicker">✓ Decision Made</div>
      <div class="xai-decision-value">${esc(block.entity)} → <span class="badge ${decisionBadgeClass(value)}">${esc(value)}</span></div>
    </div>`;
}

// ---- summary (computed purely from the already-revealed trace data) -------

function xaiBuildReasonText(supplierBlock, finalStatus, highRisk) {
  const parts = [];
  const r5 = supplierBlock.rules.find((r) => r.ruleId === 5);
  if (r5) parts.push(r5.matched ? "Available quantity covers the required quantity." : "Available quantity is below the required quantity.");
  const r3 = supplierBlock.rules.find((r) => r.ruleId === 3);
  if (r3) parts.push(r3.matched ? "Defect rate exceeded the high-risk threshold." : "Defect rate did not exceed the high-risk threshold.");
  if (finalStatus === "Preferred") parts.push("Lead time and defect rate both cleared the Preferred bar.");
  else if (finalStatus === "Reliable") parts.push("Fulfillment rate cleared the required threshold with no prior status.");
  else if (finalStatus === "Standard" && !highRisk) parts.push("The supplier did not satisfy the Preferred or Reliable conditions.");
  return parts.join(" ");
}

function xaiBuildSummary(primaryBlocks) {
  const supplierBlock = primaryBlocks.find((b) => b.entityKind === "supplier");
  const skuBlock = primaryBlocks.find((b) => b.entityKind === "sku");
  if (!supplierBlock) return null;
  const finalStatus = supplierBlock.decision?.value ?? "—";
  const highRisk = supplierBlock.rules.some((r) => r.ruleId === 3 && r.matched);
  return {
    supplier: supplierBlock.entity,
    sku: skuBlock?.entity ?? null,
    finalStatus,
    rulesEvaluated: supplierBlock.rules.length,
    rulesMatched: supplierBlock.rules.filter((r) => r.matched).length,
    highRisk,
    reason: xaiBuildReasonText(supplierBlock, finalStatus, highRisk),
  };
}

function xaiSummaryHtml(stepIdx, primaryBlocks) {
  const s = xaiBuildSummary(primaryBlocks);
  if (!s) return "";
  return `
    <div class="xai-summary-card xai-step" data-step-idx="${stepIdx}">
      <div class="xai-summary-title">✓ Forward Chaining Complete</div>
      <div class="xai-summary-grid">
        <div class="xai-summary-item"><div class="k">Supplier</div><div class="v">${esc(s.supplier)}</div></div>
        <div class="xai-summary-item"><div class="k">Final Status</div><div class="v"><span class="badge ${decisionBadgeClass(s.finalStatus)}">${esc(s.finalStatus)}</span></div></div>
        <div class="xai-summary-item"><div class="k">Rules Evaluated</div><div class="v">${s.rulesEvaluated}</div></div>
        <div class="xai-summary-item"><div class="k">Rules Matched</div><div class="v">${s.rulesMatched}</div></div>
        <div class="xai-summary-item"><div class="k">High Risk</div><div class="v">${s.highRisk ? "YES" : "NO"}</div></div>
      </div>
      ${s.reason ? `<div class="xai-summary-reason">${esc(s.reason)}</div>` : ""}
    </div>`;
}

// ---- timeline assembly ------------------------------------------------

function renderXaiTimeline(primaryBlocks) {
  const steps = [];
  let html = "";
  primaryBlocks.forEach((block) => {
    html += xaiFactCardHtml(block, steps) + `<div class="xai-connector">↓</div>`;
    block.rules.forEach((rule) => {
      html += xaiRuleCardHtml(block, rule, steps) + `<div class="xai-connector">↓</div>`;
    });
    if (block.decision) html += xaiDecisionCardHtml(block, steps) + `<div class="xai-connector">↓</div>`;
  });
  const summaryIdx = steps.push({ kind: "summary" }) - 1;
  html += xaiSummaryHtml(summaryIdx, primaryBlocks);
  return { steps, html };
}

function renderXaiStaticBlock(block) {
  const decisionValue = block.decision?.value ?? "—";
  return `
    <div class="xai-static-block">
      <button class="xai-static-header" type="button" data-toggle-static>
        <span class="xai-entity-icon">${XAI_ICONS[block.entityKind] || XAI_ICONS.generic}</span>
        <span class="xai-entity-name">${esc(block.entity)}</span>
        <span class="badge ${decisionBadgeClass(decisionValue)}">${esc(decisionValue)}</span>
        <span class="xai-rule-chevron">▾</span>
      </button>
      <div class="xai-static-body">
        ${xaiFactCardHtml(block, null)}
        <div class="xai-connector">↓</div>
        ${block.rules.map((r) => xaiRuleCardHtml(block, r, null) + `<div class="xai-connector">↓</div>`).join("")}
        ${block.decision ? xaiDecisionCardHtml(block, null) : ""}
      </div>
    </div>`;
}

// ---- panel chrome (progress bar, controls, status line) -------------------

function renderXaiPanel() {
  return `
    <div class="xai-progress-head">
      <div class="flex items-center justify-between gap-3">
        <span class="xai-progress-title">Forward Chaining Trace</span>
        <span class="xai-progress-step-label" id="xai-step-label">—</span>
      </div>
      <div class="xai-progress-track"><div class="xai-progress-fill" id="xai-progress-fill"></div></div>
    </div>
    <div class="xai-controls">
      <button id="xai-btn-play" class="xai-ctrl-btn is-primary" type="button">▶ Play</button>
      <button id="xai-btn-pause" class="xai-ctrl-btn" type="button">⏸ Pause</button>
      <button id="xai-btn-next" class="xai-ctrl-btn" type="button">→ Next Step</button>
      <button id="xai-btn-replay" class="xai-ctrl-btn" type="button">↺ Replay</button>
      <div class="xai-speed-group" role="group" aria-label="Playback speed">
        <button class="xai-speed-btn" data-speed="0.5" type="button">0.5x</button>
        <button class="xai-speed-btn is-active" data-speed="1" type="button">1x</button>
        <button class="xai-speed-btn" data-speed="2" type="button">2x</button>
      </div>
    </div>
    <p class="xai-status-line"><span class="xai-status-dot"></span><span id="xai-status-text">Ready.</span></p>
    <div class="max-h-[460px] overflow-y-auto pr-1" id="xai-scroll-region">
      <div class="xai-timeline" id="xai-timeline"></div>
    </div>
  `;
}

function xaiProgressTicksHtml(steps) {
  const total = steps.length;
  if (total < 2) return "";
  return steps.map((s, i) => {
    if (!["fact", "rule-result", "decision", "summary"].includes(s.kind)) return "";
    const pct = (i / (total - 1)) * 100;
    return `<span class="xai-progress-tick" style="left:${pct}%"></span>`;
  }).join("");
}

// ---- player engine ----------------------------------------------------

function xaiRevealStepEl(el, step) {
  el.classList.add("is-revealed");
  const card = el.closest(".xai-rule-card");
  if (!card) return;
  const badge = card.querySelector(".xai-rule-status-badge");
  if (!badge) return;
  if (step.kind === "rule-intro") {
    badge.textContent = "Evaluating…";
    badge.className = "xai-rule-status-badge st-evaluating";
  } else if (step.kind === "rule-result") {
    badge.textContent = step.rule.matched ? "Matched" : "Not Matched";
    badge.className = `xai-rule-status-badge ${step.rule.matched ? "st-matched" : "st-not-matched"}`;
  }
}

function xaiUpdateChrome() {
  const p = state.xaiPlayer;
  if (!p) return;
  const total = p.steps.length;
  const idx = p.currentIndex;
  const label = document.getElementById("xai-step-label");
  const fill = document.getElementById("xai-progress-fill");
  const statusText = document.getElementById("xai-status-text");
  if (label) label.textContent = idx < 0 ? `Step 0 of ${total}` : `Step ${idx + 1} of ${total} — ${xaiStepKindLabel(p.steps[idx])}`;
  if (fill) fill.style.width = `${total ? Math.max(0, ((idx + 1) / total) * 100) : 0}%`;
  if (statusText) statusText.textContent = idx < 0 ? "Ready — press Play to begin." : xaiStatusCaption(p.steps[idx]);

  const atEnd = idx >= total - 1;
  const playBtn = document.getElementById("xai-btn-play");
  const pauseBtn = document.getElementById("xai-btn-pause");
  const nextBtn = document.getElementById("xai-btn-next");
  if (playBtn) playBtn.disabled = p.playing || atEnd;
  if (pauseBtn) pauseBtn.disabled = !p.playing;
  if (nextBtn) nextBtn.disabled = atEnd;
}

function xaiSetStep(newIndex) {
  const p = state.xaiPlayer;
  if (!p) return;
  newIndex = Math.max(-1, Math.min(newIndex, p.steps.length - 1));
  const container = document.getElementById("xai-timeline");
  if (!container) return;
  for (let i = Math.max(0, p.currentIndex + 1); i <= newIndex; i++) {
    const el = container.querySelector(`[data-step-idx="${i}"]`);
    if (el) xaiRevealStepEl(el, p.steps[i]);
  }
  container.querySelectorAll(".is-active-glow").forEach((el) => el.classList.remove("is-active-glow"));
  if (newIndex >= 0) {
    const activeEl = container.querySelector(`[data-step-idx="${newIndex}"]`);
    if (activeEl) {
      activeEl.classList.add("is-active-glow");
      activeEl.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "center" });
    }
  }
  p.currentIndex = newIndex;
  xaiUpdateChrome();
}

function xaiResetVisual() {
  const container = document.getElementById("xai-timeline");
  if (!container) return;
  container.querySelectorAll(".is-revealed").forEach((el) => el.classList.remove("is-revealed"));
  container.querySelectorAll(".is-active-glow").forEach((el) => el.classList.remove("is-active-glow"));
  container.querySelectorAll(".xai-rule-card .xai-rule-status-badge").forEach((b) => {
    b.textContent = "Pending";
    b.className = "xai-rule-status-badge";
  });
}

function xaiPlay() {
  const p = state.xaiPlayer;
  if (!p || p.currentIndex >= p.steps.length - 1) return;
  p.playing = true;
  xaiUpdateChrome();
  const tick = () => {
    if (!p.playing) return;
    xaiSetStep(p.currentIndex + 1);
    if (p.currentIndex >= p.steps.length - 1) { p.playing = false; xaiUpdateChrome(); return; }
    p.timer = setTimeout(tick, 1300 / p.speed);
  };
  tick();
}
function xaiPause() {
  const p = state.xaiPlayer;
  if (!p) return;
  p.playing = false;
  if (p.timer) clearTimeout(p.timer);
  xaiUpdateChrome();
}
function xaiNext() {
  xaiPause();
  const p = state.xaiPlayer;
  if (p) xaiSetStep(p.currentIndex + 1);
}
function xaiReplay() {
  const p = state.xaiPlayer;
  if (!p) return;
  xaiPause();
  xaiResetVisual();
  p.currentIndex = -1;
  xaiUpdateChrome();
  xaiPlay();
}
function xaiJumpToStep(idx) {
  const p = state.xaiPlayer;
  if (!p) return;
  xaiPause();
  if (idx <= p.currentIndex) {
    const container = document.getElementById("xai-timeline");
    container?.querySelector(`[data-step-idx="${idx}"]`)
      ?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "center" });
    return;
  }
  xaiSetStep(idx);
}

function initXaiPlayer(result) {
  xaiPause();
  const trace = result?.knowledge?.reasoning_trace || [];
  const timeline = document.getElementById("xai-timeline");
  if (!timeline) return;

  const { blocks, notes } = buildXaiBlocks(trace);
  if (!blocks.length) {
    timeline.innerHTML = `<p class="text-sm text-muted">No reasoning trace available.</p>`;
    document.querySelectorAll(".xai-controls .xai-ctrl-btn").forEach((b) => { b.disabled = true; });
    return;
  }

  const recommendedId = result?.knowledge?.recommended_supplier?.supplier_id ?? null;
  const skuBlocks = blocks.filter((b) => b.entityKind === "sku");
  const primarySupplier = blocks.find((b) => b.entityKind === "supplier" && b.entity === recommendedId);
  const otherSuppliers = blocks.filter((b) => b.entityKind === "supplier" && b !== primarySupplier);
  const primaryBlocks = primarySupplier ? [...skuBlocks, primarySupplier] : skuBlocks;

  const { steps, html } = renderXaiTimeline(primaryBlocks);

  let timelineHtml = html;
  if (otherSuppliers.length) {
    timelineHtml += `
      <div class="xai-secondary-head"><h3>Other Suppliers Evaluated (${otherSuppliers.length})</h3></div>
      ${otherSuppliers.map(renderXaiStaticBlock).join("")}
    `;
  }
  if (notes.length) {
    timelineHtml += notes.map((n) => `<div class="xai-note">${esc(n.text || n.value || "")}</div>`).join("");
  }
  timeline.innerHTML = timelineHtml;

  const track = document.querySelector(".xai-progress-track");
  if (track) track.innerHTML = `<div class="xai-progress-fill" id="xai-progress-fill"></div>${xaiProgressTicksHtml(steps)}`;

  const carryOverIndex = (state.xaiPlayer && state.xaiPlayer.resultRef === result) ? state.xaiPlayer.currentIndex : -1;
  const carryOverSpeed = state.xaiPlayer?.speed || 1;
  state.xaiPlayer = { steps, currentIndex: -1, playing: false, speed: carryOverSpeed, timer: null, resultRef: result };

  document.getElementById("xai-btn-play").addEventListener("click", xaiPlay);
  document.getElementById("xai-btn-pause").addEventListener("click", xaiPause);
  document.getElementById("xai-btn-next").addEventListener("click", xaiNext);
  document.getElementById("xai-btn-replay").addEventListener("click", xaiReplay);
  document.querySelectorAll(".xai-speed-btn").forEach((btn) => {
    if (Number(btn.dataset.speed) === carryOverSpeed) btn.classList.add("is-active"); else btn.classList.remove("is-active");
    btn.addEventListener("click", () => {
      document.querySelectorAll(".xai-speed-btn").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      state.xaiPlayer.speed = Number(btn.dataset.speed);
    });
  });
  document.querySelectorAll("[data-jump-step]").forEach((el) => {
    el.addEventListener("click", () => xaiJumpToStep(Number(el.dataset.jumpStep)));
  });
  document.querySelectorAll("[data-toggle-static]").forEach((btn) => {
    btn.addEventListener("click", () => btn.closest(".xai-static-block")?.classList.toggle("is-expanded"));
  });

  xaiUpdateChrome();
  if (carryOverIndex >= 0) {
    xaiSetStep(carryOverIndex);
  } else if (prefersReducedMotion()) {
    xaiSetStep(steps.length - 1);
  } else {
    xaiPlay();
  }
}

// ===========================================================================
// VIEW 3 — Heuristic Supplier Decision Matrix
// ===========================================================================

async function loadSuppliersView() {
  const root = document.getElementById("view-suppliers");
  root.innerHTML = `
    <div class="space-y-6">
      <div class="skeleton h-7 w-80"></div>
      <div class="card p-5"><div class="skeleton h-56 w-full"></div></div>
      <div class="card p-5"><div class="skeleton h-64 w-full"></div></div>
    </div>`;
  try {
    await ensureSkus();
    const sid = defaultSkuId();
    const data = await Api.suppliers(sid);
    state.selectedSkuId = sid;
    state.lastSuppliers = data;
    renderSuppliersView(data);
  } catch (e) {
    root.innerHTML = `<div class="card p-8 text-center text-danger">Failed to load suppliers: ${esc(e.message)}</div>`;
  }
}

function renderSuppliersView(data) {
  const root = document.getElementById("view-suppliers");
  const rows = data.ranked_candidates;
  const w = data.weights;
  const pct = (k) => `${(w[k] * 100).toFixed(2).replace(/\.?0+$/, "")}`;

  root.innerHTML = `
    <div class="fade-in space-y-6">
      <div class="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 class="page-title">Heuristic Supplier Decision Matrix — ${esc(data.sku_id)}</h1>
          <p class="text-sm text-body mt-1">H = ${pct("quality")}%·Quality + ${pct("cost")}%·Cost + ${pct("fulfillment")}%·Fulfillment + ${pct("delivery")}%·Delivery — Cost and Delivery are Min-Max normalized across eligible suppliers; Quality (1 − defect rate) and Fulfillment are used as absolute rates</p>
          <p class="text-xs text-muted mt-1">Only suppliers that actually supply this SKU in the supply-chain data are candidates. ${simBadge()} = simulated field (supplier availability and quality are not in the source data).</p>
        </div>
        <div class="flex flex-wrap items-center gap-2">
          <select id="matrix-sku" class="input-field min-w-[220px]">${skuOptionsHtml(data.sku_id)}</select>
          <label class="text-xs text-body" for="required-filter">Required qty</label>
          <input id="required-filter" type="number" min="0" value="${data.required_quantity}" class="input-field w-28" />
          <button id="apply-required" class="btn-secondary">Apply</button>
        </div>
      </div>

      ${rows.length ? `
      <div class="card p-5">
        <h2 class="font-semibold text-heading mb-3">Heuristic Score by Eligible Supplier</h2>
        <div class="h-64"><canvas id="chart-heuristic"></canvas></div>
      </div>

      <div class="card p-5">
        <h2 class="font-semibold text-heading mb-1">Score Composition — the four weighted heuristic components</h2>
        <p class="text-xs text-body mb-3">Click any row in the table below to see how a supplier's Raw Score was assembled.</p>
        <div class="h-72"><canvas id="chart-breakdown"></canvas></div>
      </div>

      <div class="card p-0 overflow-hidden">
        <div class="overflow-x-auto">
          <table class="table-clean w-full">
            <thead><tr>
              <th>Rank</th><th>Supplier</th><th>Unit Cost</th><th>Lead Time (d)</th>
              <th>Available${simBadge()}</th><th>Defect Rate${simBadge()}</th><th>Fulfillment${simBadge()}</th><th>Heuristic Score</th>
            </tr></thead>
            <tbody>${rows.map(supplierRow).join("")}</tbody>
          </table>
        </div>
      </div>` : `
      <div class="card p-6 text-sm">
        <div class="font-semibold text-danger">No eligible supplier for ${fmtNum(data.required_quantity)} units</div>
        <p class="text-body mt-1">Combined availability across all ${data.pool_size} candidates is ${fmtNum(data.combined_available_qty)} units${simBadge()}. ${data.split_order_possible ? "No single supplier can cover it, but the order can be split across suppliers — run the Agent Pipeline for this SKU to see the split plan." : "Even a split order could not cover it. No supplier is ranked or recommended."}</p>
      </div>`}

      ${data.ineligible_suppliers.length ? `
      <div class="card p-0 overflow-hidden">
        <div class="px-5 py-4 border-b border-soft">
          <h2 class="font-semibold text-heading">Excluded — cannot supply ${fmtNum(data.required_quantity)} units</h2>
          <p class="text-xs text-body">Not scored: Rule 5 (AvailableQty ≥ RequiredQty) failed.</p>
        </div>
        <div class="overflow-x-auto">
          <table class="table-clean w-full">
            <thead><tr><th>Supplier</th><th>Available${simBadge()}</th><th>Shortfall</th><th>Unit Cost</th><th>Lead Time (d)</th><th>Warehouses</th></tr></thead>
            <tbody>${data.ineligible_suppliers.map((s) => `
              <tr>
                <td class="font-medium text-heading">${esc(s.supplier_id)}</td>
                <td class="tabular">${fmtNum(s.available_qty)}</td>
                <td class="tabular text-danger">${fmtNum(s.shortfall)}</td>
                <td class="tabular">${fmtCurrency(s.unit_cost)}</td>
                <td class="tabular">${s.lead_time_days}</td>
                <td class="text-body">${esc(s.warehouses)}</td>
              </tr>`).join("")}</tbody>
          </table>
        </div>
      </div>` : ""}
    </div>
  `;

  const reload = async (sku, qty) => {
    try {
      const next = await Api.suppliers(sku, qty);
      state.selectedSkuId = sku;
      state.lastSuppliers = next;
      renderSuppliersView(next);
      toast(`${sku}: ${next.eligible_count}/${next.pool_size} suppliers can supply ${next.required_quantity} units.`, "info");
    } catch (e) { toast("Failed to re-rank suppliers: " + e.message, "error"); }
  };
  document.getElementById("apply-required").addEventListener("click", () =>
    reload(document.getElementById("matrix-sku").value, Math.max(0, Number(document.getElementById("required-filter").value) || 0))
  );
  document.getElementById("matrix-sku").addEventListener("change", (e) => reload(e.target.value, null));

  root.querySelectorAll("[data-supplier-toggle]").forEach((tr) =>
    tr.addEventListener("click", () => {
      const id = tr.dataset.supplierToggle;
      const detail = root.querySelector(`[data-supplier-detail="${CSS.escape(id)}"]`);
      if (!detail) return;
      const opening = detail.hasAttribute("hidden");
      detail.toggleAttribute("hidden", !opening);
      tr.classList.toggle("is-expanded", opening);
    })
  );

  destroyChart("heuristic");
  destroyChart("breakdown");
  if (rows.length) {
    renderHeuristicChart(rows);
    renderBreakdownChart(rows.slice(0, 5));
  }
  staggerIn(root, ".card");
}

// Component display order + labels. Weights are never hard-coded here — they
// come from the backend's scoring.weights so the UI can't drift from the
// formula the Supplier Intelligence Agent actually applied.
const SCORE_COMPONENTS = [
  ["quality", "Quality"],
  ["cost", "Cost"],
  ["fulfillment", "Fulfillment"],
  ["delivery", "Delivery"],
];

function scoreBreakdownHtml(s) {
  const sc = s.scoring;
  if (!sc) return `<p class="text-sm text-muted">No score breakdown available for this supplier.</p>`;
  const parts = SCORE_COMPONENTS.map(([key, label]) => {
    const score = sc.components[key];
    const weight = sc.weights[key];
    const contribution = sc.contributions[key];
    return `
      <div class="score-part">
        <div class="score-part-head">
          <span class="score-part-name">${esc(label)}</span>
          <span class="score-part-weight">${(weight * 100).toFixed(0)}%</span>
        </div>
        <div class="score-part-bar"><div class="score-part-fill" style="width:${(score * 100).toFixed(1)}%"></div></div>
        <div class="score-part-math">
          <span>${score.toFixed(3)} × ${weight.toFixed(2)}</span>
          <span class="score-part-contrib">= ${contribution.toFixed(3)}</span>
        </div>
      </div>`;
  }).join("");
  return `
    <div class="score-breakdown">
      <div class="score-parts">${parts}</div>
      <div class="score-total">
        <div><span class="k">Raw Score</span><span class="v tabular">${sc.raw_score.toFixed(4)}</span></div>
        <div><span class="k">Final Score</span><span class="v tabular">${sc.final_score.toFixed(1)}</span></div>
        <div><span class="k">Rank</span><span class="v tabular">#${s.rank}</span></div>
      </div>
    </div>`;
}

function supplierRow(s) {
  const pct = Math.max(0, Math.min(100, s.heuristic_score));
  return `
    <tr class="supplier-row" data-supplier-toggle="${esc(s.supplier_id)}">
      <td class="font-semibold text-heading">#${s.rank}</td>
      <td>
        <div class="font-medium text-heading">${esc(s.supplier_id)}</div>
        <div class="text-xs text-muted">${esc(s.warehouses)}</div>
      </td>
      <td class="tabular" title="${s.lane_count > 1 ? `mean of ${s.lane_count} warehouse lanes: ${fmtCurrency(s.unit_cost_min)}–${fmtCurrency(s.unit_cost_max)}` : "single warehouse lane"}">${fmtCurrency(s.unit_cost)}${s.lane_count > 1 ? " *" : ""}</td>
      <td class="tabular">${s.lead_time_days}</td>
      <td class="tabular">${fmtNum(s.available_qty)}</td>
      <td class="tabular">${(s.defect_rate * 100).toFixed(2)}%</td>
      <td class="tabular">${(s.fulfillment_rate * 100).toFixed(1)}%</td>
      <td>
        <div class="flex items-center gap-2">
          <div class="w-20 h-1.5 surface-2 rounded-full overflow-hidden"><div class="h-full" style="width:${pct}%; background:var(--accent)"></div></div>
          <span class="text-xs font-semibold text-heading tabular">${s.heuristic_score.toFixed(1)}</span>
          <span class="supplier-chevron">▾</span>
        </div>
      </td>
    </tr>
    <tr class="supplier-detail-row" data-supplier-detail="${esc(s.supplier_id)}" hidden>
      <td colspan="8">${scoreBreakdownHtml(s)}</td>
    </tr>`;
}

function renderHeuristicChart(rows) {
  destroyChart("heuristic");
  const pal = chartPalette();
  const sorted = [...rows].sort((a, b) => b.heuristic_score - a.heuristic_score);
  charts.heuristic = new Chart(document.getElementById("chart-heuristic"), {
    type: "bar",
    data: {
      labels: sorted.map((s) => s.supplier_id),
      datasets: [{
        label: "Heuristic Score", data: sorted.map((s) => s.heuristic_score),
        backgroundColor: barGradient(pal.series1, false), hoverBackgroundColor: pal.series1,
        borderRadius: 4, maxBarThickness: 34,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: barEntranceAnimation(),
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 100, grid: { color: pal.grid }, ticks: { color: pal.inkMuted } },
        x: { grid: { display: false }, ticks: { color: pal.inkSecondary } },
      },
    },
  });
}

function renderBreakdownChart(top5) {
  destroyChart("breakdown");
  const pal = chartPalette();
  charts.breakdown = new Chart(document.getElementById("chart-breakdown"), {
    type: "bar",
    data: {
      labels: top5.map((s) => s.supplier_id),
      datasets: [
        // The four terms of the Raw Score, in weight order. Values and weights
        // both come from the backend's scoring object — nothing recomputed here.
        { key: "quality", label: `Quality (${pctWeight(top5, "quality")})`, color: pal.series2 },
        { key: "cost", label: `Cost (${pctWeight(top5, "cost")})`, color: pal.series1 },
        { key: "fulfillment", label: `Fulfillment (${pctWeight(top5, "fulfillment")})`, color: pal.series4 },
        { key: "delivery", label: `Delivery (${pctWeight(top5, "delivery")})`, color: pal.series3 },
      ].map((d) => ({
        label: d.label,
        data: top5.map((s) => (s.scoring ? s.scoring.components[d.key] * 100 : null)),
        backgroundColor: barGradient(d.color, false),
        hoverBackgroundColor: d.color,
        borderRadius: 3,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: barEntranceAnimation(),
      plugins: {
        legend: { position: "bottom", labels: { color: pal.inkSecondary, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const s = top5[ctx.dataIndex];
              const key = ["quality", "cost", "fulfillment", "delivery"][ctx.datasetIndex];
              if (!s?.scoring) return `${ctx.dataset.label}: —`;
              const score = s.scoring.components[key];
              const weight = s.scoring.weights[key];
              return `${ctx.dataset.label}: ${score.toFixed(3)} × ${weight.toFixed(2)} = ${(score * weight).toFixed(3)}`;
            },
          },
        },
      },
      scales: {
        y: { min: 0, max: 100, grid: { color: pal.grid }, ticks: { color: pal.inkMuted }, title: { display: true, text: "Component score (0–100)", color: pal.inkMuted, font: { size: 10 } } },
        x: { grid: { display: false }, ticks: { color: pal.inkSecondary } },
      },
    },
  });
}

function pctWeight(rows, key) {
  const w = rows.find((s) => s.scoring)?.scoring.weights[key];
  return w == null ? "—" : `${(w * 100).toFixed(0)}%`;
}

// ===========================================================================
// VIEW 4 — Procurement Action & Approval Hub
// ===========================================================================

async function loadActionView() {
  const root = document.getElementById("view-action");
  const plan = state.lastPipeline?.planning;
  const knowledge = state.lastPipeline?.knowledge;
  const focus = state.lastPipeline?.focus_fact;

  let history = [];
  try { history = (await Api.orders()).purchase_orders; } catch (_) {}

  if (!plan || plan.error || !focus) {
    root.innerHTML = `
      <div class="fade-in card p-10 text-center">
        <h1 class="text-xl font-bold text-heading mb-2">No Active Procurement Plan</h1>
        <p class="text-sm text-body mb-4">Run the Agent Pipeline for a SKU first to generate an action plan.</p>
        <button class="btn-primary" onclick="switchView('pipeline')">Go to Agent Pipeline</button>
      </div>
      ${renderOrderHistory(history)}
    `;
    return;
  }

  const rec = knowledge.recommended_supplier;
  root.innerHTML = `
    <div class="fade-in space-y-6">
      <div>
        <h1 class="page-title">Procurement Action &amp; Approval Hub</h1>
        <p class="text-sm text-body mt-1">Review the generated plan for ${esc(focus.SKU)} and execute.</p>
      </div>

      <div class="card p-6">
        <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-5">
          <div>
            <div class="text-xs text-muted uppercase tracking-wide">Complete Action Plan</div>
            <div class="text-lg font-bold text-heading">${esc(focus.SKU)} → ${plan.split_order ? plan.allocations.map((l) => esc(l.supplier_id)).join(" + ") : esc(plan.primary_supplier_id)}</div>
          </div>
          ${plan.split_order
            ? `<span class="badge badge-standard">Split order · ${plan.allocations.length} suppliers</span>`
            : `<span class="badge ${statusBadgeClass(rec?.supplier_status)}">${esc(rec?.supplier_status || "N/A")}</span>`}
        </div>

        ${plan.split_order ? `<div class="mb-6">${splitAllocationsTable(plan.allocations)}</div>` : ""}

        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          ${planStat("Recommended Order Qty", fmtNum(plan.order_quantity) + " units")}
          ${planStat("Projected Delivery", fmtDate(plan.estimated_arrival_date))}
          ${planStat("Backup Vendor", plan.backup_supplier_id || "—")}
          ${planStat("Total Cost", fmtCurrency(plan.total_expense))}
        </div>

        <div class="h-56 mb-6"><canvas id="chart-replenishment"></canvas></div>

        <div class="border-t border-soft pt-4">
          <h3 class="text-sm font-semibold text-heading mb-2">Planning Rationale</h3>
          <ul class="text-sm text-body space-y-1 list-disc pl-5">
            ${plan.reasoning.map((r) => `<li>${esc(r)}</li>`).join("")}
          </ul>
        </div>

        <div class="flex items-center gap-3 mt-6">
          <button id="approve-btn" class="btn-primary">One-Click Approve &amp; Execute Purchase Order</button>
          <span id="approve-status" class="text-sm text-success font-medium"></span>
        </div>
      </div>

      ${renderOrderHistory(history)}
    </div>
  `;

  renderReplenishmentChart(plan.replenishment_curve, plan.estimated_arrival_date);
  staggerIn(root, ".stat-tile");

  document.getElementById("approve-btn").addEventListener("click", async (e) => {
    e.target.disabled = true;
    e.target.textContent = "Processing…";
    try {
      const payload = {
        sku_id: focus.sku_id,
        supplier_id: plan.primary_supplier_id,
        supplier_name: rec?.supplier_id || plan.primary_supplier_id,
        quantity: plan.order_quantity,
        unit_cost: plan.unit_cost,
        estimated_arrival: plan.estimated_arrival_date,
        backup_supplier_id: plan.backup_supplier_id,
      };
      if (plan.split_order) {
        const res = await Api.approveSplitOrder({
          sku_id: focus.sku_id,
          allocations: plan.allocations.map((l) => ({
            supplier_id: l.supplier_id,
            quantity: l.quantity,
            unit_cost: l.unit_cost,
            estimated_arrival: l.estimated_arrival_date,
          })),
          backup_supplier_id: plan.backup_supplier_id,
        });
        document.getElementById("approve-status").textContent =
          `✓ ${res.purchase_orders.length} purchase orders approved (#${res.purchase_orders.map((p) => p.id).join(", #")}).`;
        toast("Split order approved: one purchase order per supplier.", "success");
      } else {
        const res = await Api.approveOrder(payload);
        document.getElementById("approve-status").textContent = `✓ Purchase Order #${res.purchase_order.id} approved.`;
        toast("Purchase order approved and logged.", "success");
      }
      const hist = (await Api.orders()).purchase_orders;
      document.querySelectorAll(".order-history-root").forEach((el) => (el.outerHTML = renderOrderHistory(hist)));
      e.target.textContent = "Approved ✓";
    } catch (err) {
      toast("Failed to approve order: " + err.message, "error");
      e.target.disabled = false;
      e.target.textContent = "One-Click Approve & Execute Purchase Order";
    }
  });
}

function planStat(label, value) {
  return `<div class="stat-tile"><div class="text-xs text-body">${esc(label)}</div><div class="text-xl font-bold text-heading mt-1 tabular">${value}</div></div>`;
}

function renderReplenishmentChart(curve, arrivalDate) {
  destroyChart("replenishment");
  const pal = chartPalette();
  charts.replenishment = new Chart(document.getElementById("chart-replenishment"), {
    type: "line",
    data: {
      labels: curve.map((c) => fmtDate(c.date)),
      datasets: [{
        label: "Projected Inventory",
        data: curve.map((c) => c.projected_inventory),
        borderColor: pal.series1,
        backgroundColor: areaGradient(pal.series1),
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 6,
        pointHoverBackgroundColor: pal.series1,
        pointHoverBorderColor: isDarkMode() ? "#0a1220" : "#ffffff",
        pointHoverBorderWidth: 2,
        borderWidth: 2.5,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: lineEntranceAnimation(),
      interaction: { intersect: false, mode: "index" },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.parsed.y} units${curve[ctx.dataIndex].date === arrivalDate ? "  (order arrives)" : ""}`,
          },
        },
      },
      scales: {
        y: { grid: { color: pal.grid }, ticks: { color: pal.inkMuted } },
        x: { grid: { display: false }, ticks: { color: pal.inkSecondary, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
      },
    },
  });
}

function renderOrderHistory(history) {
  return `
    <div class="card p-0 overflow-hidden order-history-root">
      <div class="px-5 py-4 border-b border-soft">
        <h2 class="font-semibold text-heading">Approved Purchase Order History</h2>
      </div>
      <div class="overflow-x-auto">
        <table class="table-clean w-full">
          <thead><tr><th>PO #</th><th>SKU</th><th>Supplier</th><th>Qty</th><th>Total Cost</th><th>ETA</th><th>Status</th></tr></thead>
          <tbody>
            ${history.length ? history.map((po) => `
              <tr>
                <td class="tabular">#${po.id}</td>
                <td class="font-medium text-heading">${esc(po.sku_id || po.product_name)}</td>
                <td class="text-body">${esc(po.supplier_name)}</td>
                <td class="tabular">${fmtNum(po.quantity)}</td>
                <td class="tabular">${fmtCurrency(po.total_cost)}</td>
                <td>${fmtDate(po.estimated_arrival)}</td>
                <td><span class="badge badge-preferred">${esc(po.status)}</span></td>
              </tr>`).join("") : `<tr><td colspan="7" class="text-center text-muted py-8">No approved purchase orders yet.</td></tr>`}
          </tbody>
        </table>
      </div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

refreshSystemHealth();
setInterval(refreshSystemHealth, 20000);
switchView("dashboard", { immediate: true });
