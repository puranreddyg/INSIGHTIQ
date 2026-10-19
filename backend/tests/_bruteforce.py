import numpy as np
from itertools import combinations
from backend import config
def feasible(sub, R, budget):
    """Exhaustive integer enumeration (no solver): does any split of R over ALL of sub meet
    capacity (>= min line), quality <= 5% and budget?"""
    lo = config.MIN_SPLIT_LINE_QTY if len(sub) > 1 else min(config.MIN_SPLIT_LINE_QTY, R)
    cap=[int(c["available_qty"]) for c in sub]; d=[c["defect_rate"] for c in sub]; u=[c["unit_cost"] for c in sub]
    if len(sub)==1:
        q=np.array([R]); qs=[q]
    elif len(sub)==2:
        q1=np.arange(lo, cap[0]+1); q2=R-q1; qs=[q1,q2]
    else:
        q1,q2=np.meshgrid(np.arange(lo,cap[0]+1),np.arange(lo,cap[1]+1),indexing="ij"); q1=q1.ravel(); q2=q2.ravel(); qs=[q1,q2,R-q1-q2]
    ok=np.ones_like(qs[0],dtype=bool)
    for i,q in enumerate(qs): ok&=(q>=lo)&(q<=cap[i])
    defect=sum(q*di for q,di in zip(qs,d)); cost=sum(q*ui for q,ui in zip(qs,u))
    ok&=defect<=config.RULE_HIGH_RISK_DEFECT_RATE*R+1e-9
    if budget is not None: ok&=cost<=budget+1e-6
    return bool(ok.any())
def best_key(pool, R, budget):
    best=None
    for k in range(1, config.SEARCH_MAX_SUPPLIERS+1):
        for sub in combinations(pool,k):
            if sum(c["available_qty"] for c in sub)<R: continue
            if feasible(sub,R,budget):
                key=(sum(c["defect_rate"]>config.RULE_HIGH_RISK_DEFECT_RATE and (c.get("evidence_pos") or 0)>=config.RULE_MIN_EVIDENCE_POS for c in sub),k)
                best=key if best is None or key<best else best
    return best
