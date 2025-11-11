# app/services/proposals_repo.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import re
import logging

import httpx  # needed for fetch_candidates_smart
from supabase import create_client, Client
from urllib.parse import quote_plus

from app.config import get_settings

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Supabase client
# -----------------------------------------------------------------------------
_s = get_settings()
_supabase: Client = create_client(_s.supabase_url, _s.supabase_anon_key)

# Export a read-only handle for other modules
supabase_ro: Client = _supabase

_FIELDS = (
    "id,category_name,service_name,positioning,duration_days,price_man_day,"
    "canonical_names,service_type,supports_db_migration,target_platforms,"
    "priority_score,popularity_score,product_family"
)

# -----------------------------------------------------------------------------
# Optional PDF chunk infra (used by generate_proposal pipeline)
# -----------------------------------------------------------------------------
# If you store this in settings, prefer that. Otherwise use env fallback.
REFERENCE_PDF_PATH: str = getattr(_s, "reference_pdf_path", None) or os.getenv("REFERENCE_PDF_PATH", "")

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

PDF_CHUNKS: List[Dict[str, Any]] = []  # [{'page': int, 'text': str}, ...]

def load_pdf_chunks(path: str) -> List[Dict[str, Any]]:
    """
    Minimal chunk loader. Returns [{'page': int, 'text': str}] list.
    If PyMuPDF is not available or file missing, returns empty list.
    """
    global PDF_CHUNKS
    if not (path and os.path.exists(path) and fitz):
        logger.warning("PDF chunk load skipped (missing file or PyMuPDF). path=%r", path)
        PDF_CHUNKS = []
        return PDF_CHUNKS
    try:
        doc = fitz.open(path)
        chunks: List[Dict[str, Any]] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            text = text.strip()
            if text:
                chunks.append({"page": i + 1, "text": text})
        PDF_CHUNKS = chunks
        logger.info("Loaded %d PDF chunks from %s", len(PDF_CHUNKS), path)
        return PDF_CHUNKS
    except Exception:
        logger.exception("Failed to load PDF chunks from %s", path)
        PDF_CHUNKS = []
        return PDF_CHUNKS

# -----------------------------------------------------------------------------
# Search helpers
# -----------------------------------------------------------------------------
# Broad synonym packs used by the ranker labels
TERM_PACKS: Dict[str, List[str]] = {
    "fitcheck": [
        "fitcheck", "assessment", "health check", "discovery", "sizing",
        "evaluation", "design workshop", "readiness"
    ],
    "infra": [
        "infrastructure", "cluster", "deployment", "commission", "expansion",
        "rack and stack", "configure", "setup", "build"
    ],
    "migration": [
        "migration", "migrate", "move", "relocate", "cutover",
        "vmware", "vsphere", "hyperflex", "source to nc2", "lift and shift"
    ],
    "database": [
        "database", "db", "ndb", "data services", "sql", "oracle",
        "postgres", "mysql", "mongodb", "schema migration"
    ],
    "dr": [
        "dr", "disaster", "recovery", "data protection", "protection domain",
        "protection domains", "replication", "metro availability", "metro",
        "leap", "site failover", "failback"
    ],
}

STOPWORDS = {
    "the","and","for","with","from","into","on","to","of","a","an","in",
    "by","is","are","be","need","needs","this","that","it","as","or"
}

def _safe_tokens(text: str, max_tokens: int = 12) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    out: List[str] = []
    for t in toks:
        if t in STOPWORDS:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= max_tokens:
            break
    return out

def _expand_query_terms(q: Optional[str]) -> List[str]:
    if not q:
        return []
    key = q.strip().lower()
    if key in TERM_PACKS:
        return TERM_PACKS[key]
    # fallback: tokenize free text
    return _safe_tokens(key, max_tokens=12)

def _normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    r["price_man_day"] = float(r.get("price_man_day") or 0)
    r["duration_days"] = int(r.get("duration_days") or 0)
    r["canonical_names"] = r.get("canonical_names") or []
    r["target_platforms"] = r.get("target_platforms") or []
    r["supports_db_migration"] = bool(r.get("supports_db_migration") or False)
    r["priority_score"] = float(r.get("priority_score") or 0)
    r["popularity_score"] = float(r.get("popularity_score") or 0)
    return r

def _apply_exact_filters(qb,
                         *,
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

def _apply_text_search_multi(qb, terms: List[str]):
    """
    Build a single OR() across service_name, positioning, product_family, canonical_names
    for all provided terms. Uses ilike on text fields and ov on canonical_names array.
    """
    if not terms:
        return qb
    ors: List[str] = []
    for t in terms:
        t_esc = t.replace(",", " ")
        ors.append(f"service_name.ilike.%{t_esc}%")
        ors.append(f"positioning.ilike.%{t_esc}%")
        ors.append(f"product_family.ilike.%{t_esc}%")
        # canonical_names is text[]; ov checks membership
        ors.append(f"canonical_names.ov.{{{t_esc}}}")
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

def _dedupe_keep_best(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        rid = r["id"]
        prev = best.get(rid)
        if not prev:
            best[rid] = r
            continue
        cur_key = (
            r.get("priority_score", 0.0),
            r.get("popularity_score", 0.0),
            -(r.get("duration_days") or 1e9)
        )
        prev_key = (
            prev.get("priority_score", 0.0),
            prev.get("popularity_score", 0.0),
            -(prev.get("duration_days") or 1e9)
        )
        if cur_key > prev_key:
            best[rid] = r
    return list(best.values())

# -----------------------------------------------------------------------------
# Public repo API
# -----------------------------------------------------------------------------
def get_all_proposals() -> List[Dict[str, Any]]:
    resp = _supabase.table("proposals_updated").select(_FIELDS).execute()
    rows: List[Dict[str, Any]] = getattr(resp, "data", []) or []
    return [_normalize_row(r) for r in rows]

def _base_qb(product_family: Optional[str],
             service_type: Optional[str],
             supports_db_migration: Optional[bool],
             max_duration: Optional[int],
             price_cap: Optional[float]):
    qb = _supabase.table("proposals_updated").select(_FIELDS)
    qb = _apply_exact_filters(
        qb,
        product_family=product_family,
        service_type=service_type,
        supports_db_migration=supports_db_migration,
        max_duration=max_duration,
        price_cap=price_cap,
    )
    return qb

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
    """
    Broad, synonym-aware fetch. Runs three buckets for platform:
    1) target_platforms overlaps requested platforms
    2) target_platforms IS NULL
    3) target_platforms NOT overlaps {ahv,aws,azure}  (mis-tagged)
    For text search, expands q via TERM_PACKS and ORs all terms across fields.
    Falls back to no-text search if zero rows.
    """
    q_terms = _expand_query_terms(q)

    def run_one(qb_builder, terms: List[str]) -> List[Dict[str, Any]]:
        qb = qb_builder()
        qb = _apply_text_search_multi(qb, terms)
        qb = _order_and_limit(qb, limit)
        data = getattr(qb.execute(), "data", []) or []
        return data

    rows: List[Dict[str, Any]] = []

    # === No platform filter ===
    if not platforms:
        def builder():
            return _base_qb(product_family, service_type, supports_db_migration, max_duration, price_cap)
        rows.extend(run_one(builder, q_terms))
        if not rows and q_terms:
            # fallback breadth without keyword constraint
            rows.extend(run_one(builder, []))
        return _py_sort([_normalize_row(r) for r in _dedupe_keep_best(rows)])[:limit]

    # === Platform filter present: run three buckets ===
    ALL_PLATFORMS = ["ahv", "aws", "azure"]

    # Q1: overlaps requested platforms
    def b1():
        qb = _base_qb(product_family, service_type, supports_db_migration, max_duration, price_cap)
        return qb.overlaps("target_platforms", platforms)

    # Q2a: target_platforms IS NULL
    def b2a():
        qb = _base_qb(product_family, service_type, supports_db_migration, max_duration, price_cap)
        return qb.is_("target_platforms", "null")

    # Q2b: NOT OVERLAPS any known platforms (catches dirty data)
    def b2b():
        qb = _base_qb(product_family, service_type, supports_db_migration, max_duration, price_cap)
        return qb.not_.overlaps("target_platforms", ALL_PLATFORMS)

    # Run with terms first
    rows.extend(run_one(b1, q_terms))
    rows.extend(run_one(b2a, q_terms))
    rows.extend(run_one(b2b, q_terms))

    # Fallback breadth if nothing matched
    if not rows and q_terms:
        rows.extend(run_one(b1, []))
        rows.extend(run_one(b2a, []))
        rows.extend(run_one(b2b, []))

    rows = [_normalize_row(r) for r in _dedupe_keep_best(rows)]
    rows = _py_sort(rows)[:limit]
    return rows

# -----------------------------------------------------------------------------
# Direct REST fetch (if you need to bypass supabase-py for special filters)
# -----------------------------------------------------------------------------
def build_or_ilike(keys: List[str], columns: List[str]) -> str:
    """
    Build a URL-encoded Supabase 'or' filter that applies ilike or ov to columns.
    For array columns use suffix '.ov' in the column name passed in `columns`.
    Example:
      build_or_ilike(["fitcheck","workshop"], ["service_name", "positioning", "canonical_names.ov"])
    """
    parts: List[str] = []
    for k in keys:
        if not k:
            continue
        like = f"%{quote_plus(k)}%"
        for col in columns:
            if col.endswith(".ov"):
                # array overlap: canonical_names.ov.{term}
                parts.append(f"{col}.%7B{quote_plus(k)}%7D")
            else:
                # text ilike: service_name.ilike.%term%
                parts.append(f"{col}.ilike.{like}")
    return f"({','.join(parts)})" if parts else ""

def fetch_candidates_smart(
    sb_url: str,
    sb_key: str,
    table: str,
    product_family: str,
    platform_bucket: str,      # 'azure' | 'aws' | 'ahv' | 'null' | 'other'
    or_filter: str,
    limit: int = 100,
):
    sel = ",".join([
        "id","category_name","service_name","positioning","duration_days","price_man_day",
        "canonical_names","service_type","supports_db_migration","target_platforms",
        "priority_score","popularity_score","product_family","embedding"
    ])
    params = {
        "select": sel,
        "product_family": f"eq.{product_family}",
        "order": "priority_score.desc.nullslast,popularity_score.desc.nullslast,duration_days.asc.nullsfirst",
        "limit": str(limit)
    }
    if platform_bucket == "null":
        params["target_platforms"] = "is.null"
    elif platform_bucket == "other":
        params["target_platforms"] = "not.ov.%7Bahv,aws,azure%7D"
    else:
        params["target_platforms"] = f"ov.%7B{platform_bucket}%7D"
    if or_filter:
        params["or"] = or_filter

    url = f"{sb_url.rstrip('/')}/rest/v1/{table}"
    headers = {"apikey": sb_key, "Authorization": f"Bearer {sb_key}"}
    with httpx.Client(timeout=httpx.Timeout(10.0, read=20.0)) as cx:
        r = cx.get(url, headers=headers, params=params)
        r.raise_for_status()
        return r.json()

# -----------------------------------------------------------------------------
# Explicit exports
# -----------------------------------------------------------------------------
__all__ = [
    "supabase_ro",
    "REFERENCE_PDF_PATH",
    "PDF_CHUNKS",
    "fitz",
    "load_pdf_chunks",
    "get_all_proposals",
    "suggest_services_repo",
    "build_or_ilike",
    "fetch_candidates_smart",
]
