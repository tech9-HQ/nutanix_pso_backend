# app/services/ranker.py
from typing import List, Tuple, Dict, Any, Optional
from app.models.schemas import SuggestPlanRequest, RankedService
from app.services.proposals_repo import suggest_services_repo
import re

# -------------------------- utils --------------------------

def _norm(x: float, lo: float, hi: float) -> float:
    if x is None or hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return 0.0 if v < 0 else (1.0 if v > 1 else v)

def _tokenize(s: str) -> List[str]:
    if not s:
        return []
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    toks = s.split()
    bigrams = [f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1)]
    return toks + bigrams

def _contains_any(text: str, tokens: List[str]) -> bool:
    if not tokens:
        return True
    t = text.lower()
    return any(tok.lower() in t for tok in tokens)

def _deployment_to_platforms(dep: Optional[str]) -> List[str]:
    if not dep:
        return []
    d = dep.lower().strip()
    if d in {"on prem", "on-prem", "onprem", "dark site", "dark-site", "darksite"}:
        return ["ahv"]
    if d in {"hybrid"}:
        return ["ahv", "aws", "azure"]
    if d in {"cloud"}:
        return ["aws", "azure"]
    return []

# -------------------------- intent + phases --------------------------

def classify_intent(req_text: str, boq: List[Any]) -> Dict[str, Any]:
    """Very light heuristic classifier. No external calls."""
    t = " ".join([(req_text or "")] + [f"{getattr(b,'sku',None) or ''} {getattr(b,'description',None) or ''}" for b in (boq or [])]).lower()

    wants_migration = any(k in t for k in ["migrate", "migration", "vmware", "move", "relocate"])
    wants_dr = any(k in t for k in ["dr", "disaster", "recovery", "metro", "witness", "protection domain", "protection-domain", "protection", "near sync", "nearsync", "sync"])
    db_signals = any(k in t for k in ["oracle", "postgres", "mysql", "sql server", "database", "ndb"])
    infra_signals = any(k in t for k in ["cluster", "nci", "ahv", "flow", "network", "microsegmentation", "vpc"])

    families: List[str] = []
    if db_signals:
        families.append("NDB")
    if infra_signals or wants_dr:
        families.append("NCI")
    # broaden if nothing detected
    if not families:
        families = ["NDB", "NCI", "NC2", "NKP", "NUS"]

    return {
        "wants_migration": wants_migration,
        "wants_dr": wants_dr,
        "families_hint": families
    }

def generate_phase_plan(intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return ordered phases to fetch candidates for."""
    phases: List[Dict[str, Any]] = []

    # Always consider assessment/design early
    phases.append({"name": "Assessment", "service_types": ["assessment"]})
    phases.append({"name": "Design", "service_types": ["design"]})

    # Deployment is common
    phases.append({"name": "Deployment", "service_types": ["deployment"]})

    if intent.get("wants_migration"):
        phases.append({"name": "Migration", "service_types": ["migration"]})

    if intent.get("wants_dr"):
        phases.append({"name": "DR", "service_types": ["deployment"], "keywords": ["dr", "metro", "protection"]})

    return phases

# -------------------------- scoring --------------------------

def _keyword_score(query: str, row: Dict[str, Any], wants_dr: bool, wants_migration: bool) -> Tuple[float, str]:
    q_tokens = _tokenize(query)
    hay = " ".join([
        row.get("service_name") or "",
        row.get("positioning") or "",
        " ".join(row.get("canonical_names") or []),
        row.get("category_name") or "",
        row.get("product_family") or "",
    ]).lower()

    hits = sum(1 for t in q_tokens if t in hay)
    exact = 1.0 if any(t in (row.get("canonical_names") or []) for t in q_tokens) else 0.0
    score = hits * 0.15 + exact * 0.4
    notes = [f"hits={hits}", f"exact={bool(exact)}"]

    if wants_dr and any(k in hay for k in [" dr ", "disaster", "recovery", "metro", "protection domain", "witness", "protection", "nearsync", "near sync", "sync"]):
        score += 0.15
        notes.append("dr_boost")
    if wants_migration and any(k in hay for k in ["migration", "move", "migrate"]):
        score += 0.10
        notes.append("mig_boost")

    return score, ", ".join(notes)

def score_results(query: str, rows: List[Dict[str, Any]], top_k: int, wants_dr: bool, wants_migration: bool) -> List[Tuple[float, str, Dict[str, Any]]]:
    pri = [r.get("priority_score") or 0.0 for r in rows]
    pop = [r.get("popularity_score") or 0.0 for r in rows]
    lo_p, hi_p = (min(pri) if pri else 0.0), (max(pri) if pri else 1.0)
    lo_q, hi_q = (min(pop) if pop else 0.0), (max(pop) if pop else 1.0)

    ranked: List[Tuple[float, str, Dict[str, Any]]] = []
    for r in rows:
        pri_n = _norm(float(r.get("priority_score") or 0.0), lo_p, hi_p)
        pop_n = _norm(float(r.get("popularity_score") or 0.0), lo_q, hi_q)
        kw, why = _keyword_score(query, r, wants_dr=wants_dr, wants_migration=wants_migration)
        score = (0.50 * pri_n) + (0.25 * pop_n) + (0.25 * kw)
        ranked.append((score, f"pri={pri_n:.2f}, pop={pop_n:.2f}, {why}", r))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:max(1, top_k)]

# -------------------------- repository fetch per phase --------------------------

def _fetch_candidates_for_phase(
    phase: Dict[str, Any],
    families: List[str],
    platforms: Optional[List[str]],
    prefer_mig: bool,
    per_family_cap: int
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    svc_types: List[str] = list(phase.get("service_types") or [])
    keywords: List[str] = list(phase.get("keywords") or [])

    for fam in families:
        # base pull by service_type(s)
        if svc_types:
            for st in svc_types:
                rows = suggest_services_repo(
                    product_family=fam,
                    platforms=platforms,
                    limit=per_family_cap,
                    service_type=st,
                    supports_db_migration=True if (prefer_mig and st == "migration") else None,
                    max_duration=None,
                    price_cap=None,
                    q=None,  # keep q empty for broad pull
                )
                out.extend(rows)
        else:
            rows = suggest_services_repo(
                product_family=fam,
                platforms=platforms,
                limit=per_family_cap,
                service_type=None,
                supports_db_migration=True if prefer_mig else None,
                q=None,
            )
            out.extend(rows)

        # keyword-narrowed pulls (safe tokens, repo handles tokenization)
        for kw in keywords:
            rows_kw = suggest_services_repo(
                product_family=fam,
                platforms=platforms,
                limit=per_family_cap,
                service_type="deployment" if fam in {"NCI", "NC2"} else None,
                supports_db_migration=None,
                q=kw,
            )
            out.extend(rows_kw)

    return out

# -------------------------- main planner --------------------------

def plan_suggestions(req: SuggestPlanRequest):
    # Build query text from requirement + BOQ + providers + vendor + deployment + industry
    boq_text = " ".join([f"{getattr(b,'sku',None) or ''} {getattr(b,'description',None) or ''} {getattr(b,'vendor',None) or ''}".strip()
                         for b in (req.boq or [])])
    providers_text = " ".join(req.providers or [])
    vendor_cue = req.selected_vendor or ""
    dep_text = req.deployment_type or ""
    industry_text = req.industry or ""

    query_text = " ".join([
        req.requirements_text or "",
        boq_text,
        providers_text,
        vendor_cue,
        dep_text,
        industry_text,
    ]).strip()

    # Intent + phases
    intent_info = classify_intent(req.requirements_text or "", (req.boq or []))
    phases = generate_phase_plan(intent_info)

    # Constraints and defaults
    platforms = req.constraints.target_platforms or _deployment_to_platforms(req.deployment_type)
    families_from_ui = req.constraints.product_families or []
    families_hint = intent_info.get("families_hint") or []
    families = families_from_ui or families_hint or ["NDB", "NCI", "NC2", "NKP", "NUS"]
    prefer_mig = bool(req.constraints.prefer_migration_ready)
    must_inc = req.constraints.must_include or []
    must_exc = req.constraints.must_exclude or []

    # Phase-wise fetch
    candidates_all: List[Dict[str, Any]] = []
    fetched_debug: Dict[str, int] = {}
    per_family_cap = max(10, req.limit * 3)

    for ph in phases:
        rows = _fetch_candidates_for_phase(
            phase=ph,
            families=families,
            platforms=platforms,
            prefer_mig=prefer_mig,
            per_family_cap=per_family_cap
        )
        fetched_debug[f"phase_{ph.get('name','unknown')}"] = len(rows)
        candidates_all.extend(rows)

    # Dedupe
    uniq: Dict[int, Dict[str, Any]] = {r["id"]: r for r in candidates_all}
    uniq_rows = list(uniq.values())

    # Flags for boosts
    q_tokens = set(_tokenize(query_text))
    wants_dr = any(t in q_tokens for t in {"dr", "disaster", "recovery", "disaster recovery", "metro", "witness", "protection", "protection domain", "nearsync", "near sync", "sync"})
    wants_migration = any(t in q_tokens for t in {"migrate", "migration", "move", "vmware", "vmware to nutanix", "oracle", "postgres"})

    # Score
    ranked = score_results(
        query=query_text,
        rows=uniq_rows,
        top_k=max(req.limit * 3, 20),
        wants_dr=wants_dr,
        wants_migration=wants_migration
    )

    # Include/exclude and provider bias
    filtered: List[Tuple[float, str, Dict[str, Any]]] = []
    provider_tokens = [p.lower() for p in (req.providers or []) if p] + ([vendor_cue.lower()] if vendor_cue else [])
    for score, why, r in ranked:
        namecat = f"{r.get('service_name','')} {r.get('category_name','')}"
        if must_inc and not _contains_any(namecat, must_inc):
            continue
        if must_exc and _contains_any(namecat, must_exc):
            continue

        hay = " ".join([r.get("service_name",""), r.get("positioning","")]).lower()
        if provider_tokens and any(p in hay for p in provider_tokens):
            score += 0.05
            why = f"{why}, provider_bias"

        filtered.append((score, why, r))

    topN = filtered[: req.limit]

    # Marshal
    items: List[RankedService] = []
    for score, why, r in topN:
        items.append(RankedService(
            id=r["id"],
            service_name=r["service_name"],
            category_name=r["category_name"],
            product_family=r["product_family"],
            score=round(float(score), 6),
            reason=why,
            duration_days=int(r.get("duration_days") or 0),
            price_man_day=float(r.get("price_man_day") or 0.0),
            service_type=r.get("service_type"),
            supports_db_migration=bool(r.get("supports_db_migration") or False),
            target_platforms=list(r.get("target_platforms") or []),
            canonical_names=list(r.get("canonical_names") or []),
            popularity_score=float(r.get("popularity_score") or 0.0),
            priority_score=float(r.get("priority_score") or 0.0),
        ))

    debug = {
        "query_text": query_text,
        "requested_platforms": platforms or [],
        "requested_families": families,
        "supports_db_migration": prefer_mig,
        "flags": {"wants_dr": wants_dr, "wants_migration": wants_migration},
        "fetched_batches": fetched_debug,
        "candidates_total": len(candidates_all),
        "unique_after_dedupe": len(uniq_rows),
        "providers": req.providers,
        "deployment_type": req.deployment_type,
        "industry": req.industry,
        "proposal_type": req.proposal_type,
    }
    return items, debug
