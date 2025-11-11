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

def _to_int(s: str) -> int:
    return int(s.replace(",", ""))

def _normalize_count(text: str) -> Optional[int]:
    t = text or ""
    # 5k, 12.3K, etc.
    m = KNUM.search(t)
    if m:
        return int(float(m.group(1)) * 1000)

    m = RANGE.search(t)
    if m:
        return _to_int(m.group(2))  # use upper bound for gating

    m = PLUS.search(t)
    if m:
        return _to_int(m.group(1)) + 1

    m = NUM.search(t)
    if m:
        return _to_int(m.group(0))

    return None

def _serper_lookup(name: str, client: httpx.Client) -> Optional[int]:
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
            return None

        data = r.json()
        blobs: list[str] = []
        for it in data.get("organic", [])[:10]:
            blobs.append(it.get("title", ""))
            blobs.append(it.get("snippet", ""))
            if it.get("richSnippet"):
                blobs.append(json.dumps(it["richSnippet"], ensure_ascii=False))
        if data.get("knowledgeGraph"):
            blobs.append(json.dumps(data["knowledgeGraph"], ensure_ascii=False))

        text = " ".join(blobs)
        # Prefer phrases near "employees"
        near = re.findall(r"(.{0,40}employees.{0,40})", text, flags=re.I)
        candidates = []
        for chunk in near:
            n = _normalize_count(chunk)
            if n:
                candidates.append(n)
        if not candidates:
            n = _normalize_count(text)
            return n
        return max(candidates)
    except Exception:
        return None

def get_company_size(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    provider = (os.getenv("SEARCH_PROVIDER") or "").lower()
    key = name.strip().lower()
    now = time.time()

    # cache
    if key in _CACHE and now - _CACHE[key][0] < _TTL:
        return _CACHE[key][1]

    with httpx.Client(follow_redirects=True) as client:
        n: Optional[int] = None
        if provider == "serper":
            n = _serper_lookup(name, client)
        # add other providers later if needed

    _CACHE[key] = (now, n)
    return n
