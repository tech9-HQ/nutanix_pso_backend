# app/services/suggest_planner.py
from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz, process
from app.models.schemas import SuggestPlanRequest, RankedService
from app.services.proposals_repo import get_all_proposals
from app.utils.text import tokenize, any_token_in

_VENDOR_HINTS: Dict[str, List[str]] = {
    "cisco": ["cisco", "ucs", "fi", "m5", "x-series"],
    "dell": ["dell", "poweredge", "r760", "r660", "powerflex"],
    "hpe": ["hpe", "proliant", "dl380", "alletra", "nimble"],
    "lenovo": ["lenovo", "thinksystem", "sr650"],
}

_FAMILY_HINTS: Dict[str, List[str]] = {
    "NCI": ["ahv", "prism", "flow", "acu", "cluster", "vpc", "microseg"],
    "NDB": ["database", "oracle", "sql", "postgres", "ndb", "clone", "patch", "backup"],
    "NUS": ["files", "nas", "share", "smb", "nfs", "unified storage", "nus"],
    "NKP": ["kubernetes", "k8s", "containers", "nkp", "openshift"],
    "NC2": ["aws", "azure", "nc2", "vmc migration", "cloud cluster"],
}

def _vendor_boost(selected_vendor: str | None, text_tokens: List[str]) -> Dict[str, float]:
    boosts: Dict[str, float] = {}
    if selected_vendor:
        v = selected_vendor.lower()
        boosts[v] = 0.08
    # add implied vendor from BOQ words
    for v, hints in _VENDOR_HINTS.items():
        if any(h in text_tokens for h in hints):
            boosts[v] = max(boosts.get(v, 0), 0.05)
    return boosts

def _family_boost(all_text: str, explicit: List[str]) -> Dict[str, float]:
    boosts: Dict[str, float] = {}
    tokens = tokenize(all_text)
    for fam, hints in _FAMILY_HINTS.items():
        if any(h in tokens for h in hints):
            boosts[fam] = 0.12
    for fam in explicit:
        boosts[fam] = max(boosts.get(fam, 0), 0.15)
    return boosts

def _base_score(row: Dict[str, Any]) -> float:
    # deterministic base
    return 0.6 * float(row.get("priority_score", 0)) + 0.4 * float(row.get("popularity_score", 0))

def _text_match_score(query: str, row: Dict[str, Any]) -> float:
    hay = " | ".join([
        row.get("service_name",""),
        row.get("positioning",""),
        " ".join(row.get("canonical_names") or []),
        row.get("category_name",""),
        row.get("product_family",""),
        " ".join(row.get("target_platforms") or []),
    ])
    # Use partial_ratio to reward token overlap
    pr = fuzz.partial_ratio(query, hay)
    # Scale 0..100 -> 0..0.4
    return pr / 250.0

def _constraints_filter(row: Dict[str, Any], req: SuggestPlanRequest, tokens: List[str]) -> bool:
    c = req.constraints
    if c.product_families and row.get("product_family") not in c.product_families:
        return False
    if c.target_platforms:
        rp = set([p.lower() for p in (row.get("target_platforms") or [])])
        if not rp.intersection({p.lower() for p in c.target_platforms}):
            return False
    for t in c.must_include:
        if t.lower() not in row.get("service_name","").lower():
            return False
    for t in c.must_exclude:
        if t.lower() in row.get("service_name","").lower():
            return False
    return True

def _reason(row: Dict[str, Any], vendor_hit: str | None, fam_hit: str | None, pr: float) -> str:
    bits = []
    if fam_hit:
        bits.append(f"matches {fam_hit}")
    if vendor_hit:
        bits.append(f"aligned to {vendor_hit}")
    if row.get("supports_db_migration"):
        bits.append("migration-ready")
    if pr >= 60:
        bits.append("strong text match")
    return ", ".join(bits) or "relevant to inputs"

def plan_suggestions(req: SuggestPlanRequest) -> Tuple[List[RankedService], Dict[str, Any]]:
    catalog = get_all_proposals()

    # Build a single query text = requirements + BOQ strings
    boq_join = "\n".join([f"{b.vendor or ''} {b.sku or ''} {b.description}" for b in req.boq])
    query_text = f"{req.requirements_text}\n{boq_join}".strip()
    q_tokens = tokenize(query_text)

    # Boost maps
    vboost = _vendor_boost(req.selected_vendor, q_tokens)
    fboost = _family_boost(query_text, req.constraints.product_families)

    ranked: List[Tuple[float, Dict[str, Any], float, str | None, str | None]] = []
    for row in catalog:
        if not _constraints_filter(row, req, q_tokens):
            continue

        base = _base_score(row)
        tm = _text_match_score(query_text, row)

        fam = row.get("product_family")
        fam_b = fboost.get(fam, 0.0)

        vend_hit = None
        vend_b = 0.0
        for v, b in vboost.items():
            if v == "cisco" and "cisco" in row["service_name"].lower():
                vend_hit, vend_b = "Cisco", b
                break
            if v == "dell" and ("dell" in row["service_name"].lower() or "powerflex" in row["service_name"].lower()):
                vend_hit, vend_b = "Dell", b
                break

        # small bonus if migration focus requested implicitly by NDB services
        mig_b = 0.05 if (req.constraints.prefer_migration_ready and row.get("supports_db_migration")) else 0.0

        score = base + tm + fam_b + vend_b + mig_b
        ranked.append((score, row, tm*250, vend_hit, fam))

    ranked.sort(key=lambda x: x[0], reverse=True)
    out: List[RankedService] = []
    for score, row, pr, vend_hit, fam in ranked[: req.limit]:
        out.append(RankedService(
            id=row["id"],
            service_name=row["service_name"],
            category_name=row["category_name"],
            product_family=row["product_family"],
            score=round(score, 4),
            reason=_reason(row, vend_hit, fam, pr),
            duration_days=row["duration_days"],
            price_man_day=row["price_man_day"],
            service_type=row.get("service_type"),
            supports_db_migration=row.get("supports_db_migration", False),
            target_platforms=row.get("target_platforms") or [],
            canonical_names=row.get("canonical_names") or [],
            popularity_score=row.get("popularity_score", 0.0),
            priority_score=row.get("priority_score", 0.0),
        ))

    debug = {"vendor_boost": vboost, "family_boost": fboost, "q_len": len(q_tokens)}
    return out, debug
