# app/services/proposals_repo.py
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from app.config import get_settings
import re

_s = get_settings()
_supabase: Client = create_client(_s.supabase_url, _s.supabase_anon_key)

_FIELDS = (
    "id,category_name,service_name,positioning,duration_days,price_man_day,"
    "canonical_names,service_type,supports_db_migration,target_platforms,"
    "priority_score,popularity_score,product_family"
)

STOPWORDS = {
    "the","and","for","with","from","into","on","to","of","a","an","in",
    "by","is","are","be","need","needs","this","that","it","as","or"
}

def _safe_tokens(text: str, max_tokens: int = 12) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    out: list[str] = []
    for t in toks:
        if t in STOPWORDS:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= max_tokens:
            break
    return out

def _normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    r["price_man_day"] = float(r.get("price_man_day") or 0)
    r["duration_days"] = int(r.get("duration_days") or 0)
    r["canonical_names"] = r.get("canonical_names") or []
    r["target_platforms"] = r.get("target_platforms") or []
    r["supports_db_migration"] = bool(r.get("supports_db_migration") or False)
    r["priority_score"] = float(r.get("priority_score") or 0)
    r["popularity_score"] = float(r.get("popularity_score") or 0)
    return r

def _apply_exact_filters(qb, *,
                         product_family: Optional[str],
                         service_type: Optional[str],
                         supports_db_migration: Optional[bool],
                         max_duration: Optional[int],
                         price_cap: Optional[float]):
    if product_family:
        qb = qb.eq("product_family", product_family)
    if service_type:
        qb = qb.eq("service_type", service_type)
    if supports_db_migration is not None:
        qb = qb.eq("supports_db_migration", supports_db_migration)
    if max_duration is not None:
        qb = qb.lte("duration_days", max_duration)
    if price_cap is not None:
        qb = qb.lte("price_man_day", price_cap)
    return qb

def _apply_text_search(qb, q: Optional[str]):
    if not q:
        return qb
    tokens = _safe_tokens(q)
    if not tokens:
        return qb
    ors = []
    for t in tokens:
        ors.append(f"service_name.ilike.%{t}%")
        ors.append(f"positioning.ilike.%{t}%")
        ors.append(f"product_family.ilike.%{t}%")
        ors.append(f"canonical_names.ov.{{{t}}}")
    return qb.or_(",".join(ors))

def _order_and_limit(qb, limit: int):
    return (
        qb.order("priority_score", desc=True, nullsfirst=False)
          .order("popularity_score", desc=True, nullsfirst=False)
          .order("duration_days", desc=False, nullsfirst=True)
          .limit(limit)
    )

def _py_sort(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r):
        return (
            -(r.get("priority_score") or 0.0),
            -(r.get("popularity_score") or 0.0),
            (r.get("duration_days") or 10**9)
        )
    return sorted(rows, key=key)

def get_all_proposals() -> List[Dict[str, Any]]:
    resp = _supabase.table("proposals_updated").select(_FIELDS).execute()
    rows: List[Dict[str, Any]] = getattr(resp, "data", []) or []
    return [_normalize_row(r) for r in rows]

def suggest_services_repo(
    product_family: Optional[str],
    platforms: Optional[List[str]],
    limit: int,
    service_type: Optional[str] = None,
    supports_db_migration: Optional[bool] = None,
    max_duration: Optional[int] = None,
    price_cap: Optional[float] = None,
    q: Optional[str] = None,
) -> List[Dict[str, Any]]:

    # No platform filter -> single query
    if not platforms:
        qb = _supabase.table("proposals_updated").select(_FIELDS)
        qb = _apply_exact_filters(
            qb,
            product_family=product_family,
            service_type=service_type,
            supports_db_migration=supports_db_migration,
            max_duration=max_duration,
            price_cap=price_cap,
        )
        qb = _apply_text_search(qb, q)
        qb = _order_and_limit(qb, limit)
        resp = qb.execute()
        rows: List[Dict[str, Any]] = getattr(resp, "data", []) or []
        return [_normalize_row(r) for r in rows]

    # Platform filter present -> do 3 queries and merge

    # Q1: overlap with requested platforms
    qb1 = _supabase.table("proposals_updated").select(_FIELDS)
    qb1 = _apply_exact_filters(
        qb1,
        product_family=product_family,
        service_type=service_type,
        supports_db_migration=supports_db_migration,
        max_duration=max_duration,
        price_cap=price_cap,
    )
    qb1 = qb1.overlaps("target_platforms", platforms)
    qb1 = _apply_text_search(qb1, q)
    qb1 = _order_and_limit(qb1, limit)
    r1 = getattr(qb1.execute(), "data", []) or []

    # Q2a: target_platforms IS NULL
    qb2a = _supabase.table("proposals_updated").select(_FIELDS)
    qb2a = _apply_exact_filters(
        qb2a,
        product_family=product_family,
        service_type=service_type,
        supports_db_migration=supports_db_migration,
        max_duration=max_duration,
        price_cap=price_cap,
    )
    qb2a = qb2a.is_("target_platforms", 'null')
    qb2a = _apply_text_search(qb2a, q)
    qb2a = _order_and_limit(qb2a, limit)
    r2a = getattr(qb2a.execute(), "data", []) or []

    # Q2b: NOT OVERLAPS any known platforms
    ALL_PLATFORMS = ["ahv", "aws", "azure"]
    qb2b = _supabase.table("proposals_updated").select(_FIELDS)
    qb2b = _apply_exact_filters(
        qb2b,
        product_family=product_family,
        service_type=service_type,
        supports_db_migration=supports_db_migration,
        max_duration=max_duration,
        price_cap=price_cap,
    )
    qb2b = qb2b.not_.overlaps("target_platforms", ALL_PLATFORMS)
    qb2b = _apply_text_search(qb2b, q)
    qb2b = _order_and_limit(qb2b, limit)
    r2b = getattr(qb2b.execute(), "data", []) or []

    # Merge + dedupe by id, sort, trim
    merged: Dict[int, Dict[str, Any]] = {}
    for r in (r1 + r2a + r2b):
        merged[r["id"]] = r
    rows = [_normalize_row(v) for v in merged.values()]
    rows = _py_sort(rows)[:limit]
    return rows
