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
    products: (params = {}) => {
      const qs = new URLSearchParams(params).toString();
      return _req(`/api/products?${qs}`);
    },
    runPipeline: (productId) => _req(`/api/run-agent-pipeline?product_id=${productId}`, { method: "POST" }),
    suppliers: (requiredDeficit = 0) => _req(`/api/suppliers?required_deficit=${requiredDeficit}`),
    approveOrder: (payload) => _req("/api/approve-order", { method: "POST", body: JSON.stringify(payload) }),
    orders: () => _req("/api/orders"),
  };
})();
