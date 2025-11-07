from __future__ import annotations
from typing import List, Dict, Any

_PHASE_ORDER = {
    "Kickoff / Assessment & Planning": 1,
    "Infrastructure Setup": 2,
    "Data Migration": 3,
    "Cutover & Go-Live": 4,
    "Post-Migration Optimization": 5,
}

def make_journey(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build phase view using the already-overwritten duration_days from ranker."""
    def _phase_of(r: Dict[str,Any]) -> str:
        reason = r.get("reason","")
        if reason.startswith("[") and "]" in reason:
            return reason[1:reason.index("]")]
        cat = (r.get("category_name") or "").lower()
        if "assessment" in cat or "fitcheck" in (r.get("service_name","").lower()):
            return "Kickoff / Assessment & Planning"
        if "deployment" in cat:
            return "Infrastructure Setup"
        if "migration" in cat:
            return "Data Migration"
        return "Post-Migration Optimization"

    phases: Dict[str, Dict[str, Any]] = {}
    for it in items:
        ph = _phase_of(it)
        bucket = phases.setdefault(ph, {"phase": ph, "services": [], "phase_days": 0, "phase_cost_usd": 0.0})
        days = int(it.get("duration_days") or 0)
        rate = float(it.get("price_man_day") or 0.0)
        bucket["services"].append({
            "id": it["id"],
            "name": it["service_name"],
            "family": it["product_family"],
            "type": it.get("service_type"),
            "days": days,
            "rate_usd_per_day": rate,
            "extended_usd": rate * days,
            "why": it.get("reason",""),
        })
        bucket["phase_days"] += days
        bucket["phase_cost_usd"] += rate * days

    ordered = sorted(phases.values(), key=lambda b: _PHASE_ORDER.get(b["phase"], 99))
    total_days = sum(p["phase_days"] for p in ordered)
    total_cost = round(sum(p["phase_cost_usd"] for p in ordered), 2)

    return {
        "phases": ordered,
        "totals": {"days": total_days, "cost_usd": total_cost}
    }
