# app/services/suggest_planner.py
from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple, Optional
from rapidfuzz import fuzz
from app.models.schemas import SuggestPlanRequest, RankedService
from app.services.proposals_repo import get_all_proposals
from app.utils.text import tokenize
from app.services.company_size import get_company_size

# -------- Hints --------
_VENDOR_HINTS: Dict[str, List[str]] = {
    "cisco": ["cisco", "ucs", "fi", "m5", "x-series", "x series"],
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

# Normalize tier words found in catalog rows
_TIER_KEYWORDS: Dict[str, List[str]] = {
    "starter": ["starter", "basic", "essentials", "essential", "foundation", "foundational"],
    "pro": ["pro", "standard", "advanced", "plus"],
    "ultimate": ["ultimate", "enterprise", "premium", "platinum", "ultra"],
}

# flatten all tier words for regex
_TIER_WORDS = {w.lower() for words in _TIER_KEYWORDS.values() for w in words}

def _tierless_key(row: Dict[str, Any]) -> str:
    """
    Build a grouping key that strips tier labels so only one of
    Starter/Pro/Ultimate variants is kept per base service.
    """
    name = (row.get("service_name") or "").lower()
    # remove " - Starter/Pro/Ultimate/Enterprise/..." suffixes
    name = re.sub(
        r"\s*-\s*(starter|pro|ultimate|enterprise|premium|platinum|basic|essentials|standard|advanced|plus)\b",
        "",
        name,
    )
    # remove any remaining tier words anywhere
    name = re.sub(r"\b(" + "|".join(map(re.escape, _TIER_WORDS)) + r")\b", "", name)
    name = re.sub(r"\s{2,}", " ", name).strip()

    cat = (row.get("category_name") or "").lower()
    fam = (row.get("product_family") or "").lower()
    # group by category + family + base name
    return f"{cat}|{fam}|{name}"


# -------- Scoring helpers --------
def _vendor_boost(selected_vendor: Optional[str], text_tokens: List[str]) -> Dict[str, float]:
    boosts: Dict[str, float] = {}
    if selected_vendor:
        boosts[selected_vendor.lower()] = 0.08
    for v, hints in _VENDOR_HINTS.items():
        if any(h in text_tokens for h in hints):
            boosts[v] = max(boosts.get(v, 0.0), 0.05)
    return boosts


def _family_boost(all_text: str, explicit: List[str]) -> Dict[str, float]:
    boosts: Dict[str, float] = {}
    tokens = tokenize(all_text)
    for fam, hints in _FAMILY_HINTS.items():
        if any(h in tokens for h in hints):
            boosts[fam] = 0.12
    for fam in explicit:
        boosts[fam] = max(boosts.get(fam, 0.0), 0.15)
    return boosts


def _base_score(row: Dict[str, Any]) -> float:
    return 0.6 * float(row.get("priority_score", 0.0)) + 0.4 * float(row.get("popularity_score", 0.0))


def _text_match_score(query: str, row: Dict[str, Any]) -> float:
    hay = " | ".join([
        row.get("service_name", ""),
        row.get("positioning", ""),
        " ".join(row.get("canonical_names") or []),
        row.get("category_name", ""),
        row.get("product_family", ""),
        " ".join(row.get("target_platforms") or []),
    ])
    pr = fuzz.partial_ratio(query, hay)  # 0..100
    return pr / 250.0  # scale to 0..0.4


def _constraints_filter(row: Dict[str, Any], req: SuggestPlanRequest, tokens: List[str]) -> bool:
    c = req.constraints
    if c.product_families and row.get("product_family") not in c.product_families:
        return False
    if c.target_platforms:
        rp = set([p.lower() for p in (row.get("target_platforms") or [])])
        if not rp.intersection({p.lower() for p in c.target_platforms}):
            return False
    for t in c.must_include:
        if t.lower() not in row.get("service_name", "").lower():
            return False
    for t in c.must_exclude:
        if t.lower() in row.get("service_name", "").lower():
            return False
    return True


# -------- Tier helpers --------
def _infer_row_tier(row: Dict[str, Any]) -> Optional[str]:
    text = " ".join([
        str(row.get("service_name", "")),
        str(row.get("positioning", "")),
        " ".join(row.get("canonical_names") or []),
        str(row.get("category_name", "")),
    ]).lower()
    for tier, kws in _TIER_KEYWORDS.items():
        if any(k in text for k in kws):
            return tier
    return None


def _parse_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(str(val).replace(",", "").strip())
    except Exception:
        return None


def _extract_company_size(req: SuggestPlanRequest) -> Optional[int]:
    # Prefer explicit numeric fields if present
    for obj in (req, getattr(req, "constraints", None), getattr(req, "meta", None)):
        if not obj:
            continue
        for attr in ("company_size", "employee_count", "employees", "org_size", "size"):
            n = _parse_int(getattr(obj, attr, None))
            if n is not None:
                return n
    # Fallback: infer using company name via Serper
    client_name = getattr(req, "client_name", None) or getattr(req, "company_name", None)
    return get_company_size(client_name)


def _desired_tier_for_size(size: Optional[int]) -> Optional[str]:
    if size is None:
        return None
    if size < 500:
        return "starter"
    if size < 1000:
        return "pro"
    return "ultimate"


def _reason(
    row: Dict[str, Any],
    vendor_hit: Optional[str],
    fam_hit: Optional[str],
    pr: float,
    tier_hit: Optional[str],
    desired_tier: Optional[str],
) -> str:
    bits: List[str] = []
    if fam_hit:
        bits.append(f"matches {fam_hit}")
    if vendor_hit:
        bits.append(f"aligned to {vendor_hit}")
    if row.get("supports_db_migration"):
        bits.append("migration-ready")
    if pr >= 60:
        bits.append("strong text match")
    if desired_tier and tier_hit == desired_tier:
        bits.append(f"tier={desired_tier}")
    return ", ".join(bits) or "relevant to inputs"


# -------- Main --------
def plan_suggestions(req: SuggestPlanRequest) -> Tuple[List[RankedService], Dict[str, Any]]:
    catalog = get_all_proposals()

    # Query text = requirements + BOQ
    boq_join = "\n".join([f"{b.vendor or ''} {b.sku or ''} {b.description}" for b in req.boq])
    query_text = f"{req.requirements_text}\n{boq_join}".strip()
    q_tokens = tokenize(query_text)

    # Determine desired tier from company size
    company_size = _extract_company_size(req)
    desired_tier = _desired_tier_for_size(company_size)

    # Boosts
    vboost = _vendor_boost(req.selected_vendor, q_tokens)
    fboost = _family_boost(query_text, req.constraints.product_families)

    ranked: List[Tuple[float, Dict[str, Any], float, Optional[str], Optional[str], Optional[str]]] = []

    for row in catalog:
        if not _constraints_filter(row, req, q_tokens):
            continue

        row_tier = _infer_row_tier(row)

        # If we know the desired tier, exclude rows tagged with a different tier.
        if desired_tier and row_tier and row_tier != desired_tier:
            continue

        base = _base_score(row)
        tm = _text_match_score(query_text, row)

        fam = row.get("product_family")
        fam_b = fboost.get(fam, 0.0)

        vend_hit = None
        vend_b = 0.0
        name_l = row.get("service_name", "").lower()
        for v, b in vboost.items():
            if v == "cisco" and "cisco" in name_l:
                vend_hit, vend_b = "Cisco", b
                break
            if v == "dell" and ("dell" in name_l or "powerflex" in name_l):
                vend_hit, vend_b = "Dell", b
                break

        mig_b = 0.05 if (req.constraints.prefer_migration_ready and row.get("supports_db_migration")) else 0.0
        tier_b = 0.08 if (desired_tier and row_tier and row_tier == desired_tier) else 0.0

        score = base + tm + fam_b + vend_b + mig_b + tier_b
        ranked.append((score, row, tm * 250, vend_hit, fam, row_tier))

    ranked.sort(key=lambda x: x[0], reverse=True)

    # keep only one suggestion per tiered base service
    out: List[RankedService] = []
    seen: set[str] = set()

    for score, row, pr, vend_hit, fam, row_tier in ranked:
        key = _tierless_key(row)
        if key in seen:
            continue
        seen.add(key)

        out.append(RankedService(
            id=row["id"],
            service_name=row["service_name"],
            category_name=row["category_name"],
            product_family=row["product_family"],
            score=round(score, 4),
            reason=_reason(row, vend_hit, fam, pr, row_tier, desired_tier) + " (deduped)",
            duration_days=row["duration_days"],
            price_man_day=row["price_man_day"],
            service_type=row.get("service_type"),
            supports_db_migration=row.get("supports_db_migration", False),
            target_platforms=row.get("target_platforms") or [],
            canonical_names=row.get("canonical_names") or [],
            popularity_score=row.get("popularity_score", 0.0),
            priority_score=row.get("priority_score", 0.0),
        ))

        if len(out) >= req.limit:
            break


    debug = {
        "vendor_boost": vboost,
        "family_boost": fboost,
        "q_len": len(q_tokens),
        "company_size": company_size,
        "desired_tier": desired_tier,
        "returned": len(out),
    }
    return out, debug
