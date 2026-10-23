// Thin fetch wrapper for the InsightIQ REST API.
const Api = (() => {
  async function _req(path, opts = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      throw new Error(`${res.status} ${detail}`);
    }
    return res.json();
  }

  return {
    systemHealth: () => _req("/api/system-health"),
    dashboard: () => _req("/api/dashboard"),
    skus: (params = {}) => {
      const qs = new URLSearchParams(params).toString();
      return _req(`/api/skus?${qs}`);
    },
    runPipeline: (skuId) => _req(`/api/run-agent-pipeline?sku_id=${encodeURIComponent(skuId)}`, { method: "POST" }),
    suppliers: (skuId, requiredQuantity = null) => {
      const qs = new URLSearchParams({ sku_id: skuId });
      if (requiredQuantity !== null) qs.set("required_quantity", requiredQuantity);
      return _req(`/api/suppliers?${qs}`);
    },
    approveOrder: (payload) => _req("/api/approve-order", { method: "POST", body: JSON.stringify(payload) }),
    orders: () => _req("/api/orders"),
    forecastMetrics: () => _req("/api/forecast/metrics"),
    forecast: (skuId, horizonDays = 21) => _req(`/api/forecast?sku_id=${encodeURIComponent(skuId)}&horizon_days=${horizonDays}`),
  };
})();
