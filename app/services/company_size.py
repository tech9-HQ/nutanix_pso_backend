# app/services/company_size.py
from __future__ import annotations
import os, re, time, json
from typing import Optional, Dict, Tuple
import httpx

# 24h in-memory cache
_CACHE: Dict[str, Tuple[float, Optional[int]]] = {}
_TTL = 60 * 60 * 24

# Regexes
NUM = re.compile(r"\b\d[\d,]*\b")
KNUM = re.compile(r"\b(\d+(?:\.\d+)?)\s*[kK]\b")            # 10k → 10000
RANGE = re.compile(r"(\d[\d,]*)\s*[-–]\s*(\d[\d,]*)")       # 1,001–5,000
PLUS = re.compile(r"(\d[\d,]*)\s*\+")                       # 10,001+

# -------------------------
# Helper conversions
# -------------------------
def _to_int(s: str) -> int:
    # Accepts values with commas like "1,234" -> 1234
    return int(s.replace(",", ""))


def _normalize_count(text: str) -> Optional[int]:
    """
    Heuristic extractor to find an integer headcount in a text blob.
    - Handles "5k", "12.3K" (k or K), numeric ranges (uses upper bound),
      trailing plus "10,001+" -> returns 10002 (i.e. +1), and bare numbers.
    - Returns the first match by priority (KNUM > RANGE > PLUS > NUM).
    Notes / caveats:
      * Picking the upper bound for ranges is a deliberate business choice
        (useful for gating). Document this behavior.
      * "10k" -> 10000; "12.3K" -> 12300 (float→int).
      * The regexes assume en-US digit grouping; other locales may break.
      * For ambiguous text (multiple numbers) this returns first / upper-bound.
    """
    t = text or ""
    # 5k, 12.3K, etc.
    m = KNUM.search(t)
    if m:
        # float conversion is OK; we multiply by 1k. Consider rounding policy.
        return int(float(m.group(1)) * 1000)

    m = RANGE.search(t)
    if m:
        # choose upper bound to be conservative for gating/pricing
        return _to_int(m.group(2))

    m = PLUS.search(t)
    if m:
        # "10,001+" -> return 10002 (i.e., minimal number greater than reported)
        return _to_int(m.group(1)) + 1

    m = NUM.search(t)
    if m:
        return _to_int(m.group(0))

    return None


# -------------------------
# Provider-specific scrapers / lookups
# -------------------------
def _serper_lookup(name: str, client: httpx.Client) -> Optional[int]:
    """
    Use Serper (google.serper.dev) to search likely sources (linkedin/craft/zoominfo/zoominfo/crunchbase).
    - Requires SERPER_API_KEY env var.
    - Returns the maximum candidate found in 'near employees' snippets, otherwise first normalized number.
    Notes / concerns:
      * This is a brittle heuristic — results vary widely by company and region.
      * Respect rate limits; the function uses a fixed timeout (12s) but no retry/backoff.
      * Consider adding caching at call-site (already done) and a request limiter.
      * The query biases toward certain domains; you may want to expand or change sources.
      * We intentionally limit to top 10 organic results.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return None

    # Bias to sources that usually mention headcount
    q = (
        f'{name} employees OR "employee count" OR "company size" '
        'site:linkedin.com OR site:craft.co OR site:rocketreach.co OR site:zoominfo.com OR site:crunchbase.com'
    )
    try:
        r = client.post(
            "https://google.serper.dev/search",
            json={"q": q, "num": 10},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=12.0,
        )
        if r.status_code != 200:
            # consider logging non-200 responses with r.text (careful with PII)
            return None

        data = r.json()
        blobs: list[str] = []
        for it in data.get("organic", [])[:10]:
            blobs.append(it.get("title", ""))
            blobs.append(it.get("snippet", ""))
            if it.get("richSnippet"):
                # rich snippets may include numbers/structured data; stringify them
                blobs.append(json.dumps(it["richSnippet"], ensure_ascii=False))
        if data.get("knowledgeGraph"):
            blobs.append(json.dumps(data["knowledgeGraph"], ensure_ascii=False))

        text = " ".join(blobs)
        # Prefer phrases near "employees": extract contextual snippets
        near = re.findall(r"(.{0,40}employees.{0,40})", text, flags=re.I)
        candidates = []
        for chunk in near:
            n = _normalize_count(chunk)
            if n:
                candidates.append(n)
        if not candidates:
            # fallback: search entire blob for any number-like matches
            n = _normalize_count(text)
            return n
        # return the largest candidate to be conservative (avoid underestimating)
        return max(candidates)
    except Exception:
        # swallowing exceptions is OK here (best-effort). Consider logging with logger.exception
        return None


# -------------------------
# Public API
# -------------------------
def get_company_size(name: Optional[str]) -> Optional[int]:
    """
    Public helper to return estimated employee count for a company name.
    Behavior:
      - Uses in-memory cache for _TTL seconds (24h by default)
      - Read provider from SEARCH_PROVIDER env var (supports 'serper' now)
      - Returns None if not available
    Operational notes:
      - In-memory cache is per-process; in a multi-worker deployment it's not shared.
        If you need global caching, use Redis or another shared cache.
      - This is synchronous and uses httpx.Client (sync). If you need an async path,
        add an async variant using httpx.AsyncClient.
      - Consider adding telemetry (cache hit/miss, provider failures).
    """
    if not name:
        return None
    provider = (os.getenv("SEARCH_PROVIDER") or "").lower()
    key = name.strip().lower()
    now = time.time()

    # cache lookup
    if key in _CACHE and now - _CACHE[key][0] < _TTL:
        return _CACHE[key][1]

    # HTTP client: synchronous context manager. Keep small timeout and follow_redirects True.
    with httpx.Client(follow_redirects=True) as client:
        n: Optional[int] = None
        if provider == "serper":
            n = _serper_lookup(name, client)
        # add other providers later if needed

    # store in cache even if None (prevents repeated failed lookups for TTL)
    _CACHE[key] = (now, n)
    return n
