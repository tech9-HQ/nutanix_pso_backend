# app/services/llm_selector.py
from __future__ import annotations
import os, json, re, logging, requests
from typing import List, Dict, Any, Optional

log = logging.getLogger("llm_selector")

AZURE_ENDPOINT   = os.getenv("AZURE_ENDPOINT", "").rstrip("/")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "")
AZURE_API_KEY    = os.getenv("AZURE_API_KEY", "")
SUPABASE_URL     = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY     = os.getenv("SUPABASE_KEY", "")

def _fetch_supabase(url: str, params: Dict[str, str]) -> List[Dict[str, Any]]:
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Supabase fetch failed: {e}")
        return []

def _azure_chat(prompt: str, max_tokens: int = 400) -> Optional[str]:
    if not (AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_KEY):
        log.warning("Azure LLM not configured.")
        return None
    try:
        url = f"{AZURE_ENDPOINT}/chat/completions"
        headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
        payload = {
            "model": AZURE_DEPLOYMENT,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            log.error(f"Azure LLM error {r.status_code}: {r.text[:300]}")
            return None
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        log.error(f"Azure LLM call failed: {e}")
        return None

def _score_match(text: str, row: Dict[str, Any]) -> float:
    t = text.lower()
    name = (row.get("service_name") or "").lower()
    fam = (row.get("product_family") or "").lower()
    stype = (row.get("service_type") or "").lower()
    targets = " ".join([p or "" for p in (row.get("target_platforms") or [])]).lower()
    score = 0.0
    if "nc2" in t and ("nc2" in name or fam == "nc2"): score += 2.0
    if "azure" in t and "azure" in targets: score += 2.0
    if "aws" in t and "aws" in targets: score += 1.0
    if ("db" in t or "database" in t) and (("ndb" in name) or fam == "ndb"): score += 2.0
    if "migration" in t and "migration" in stype: score += 1.5
    if "deployment" in t and "deployment" in stype: score += 1.0
    if "assessment" in t and "assessment" in stype: score += 0.8
    if ("dr" in t or "disaster" in t) and ("dr" in name or "recovery" in name): score += 1.2
    if fam in {"other"} and not any(k in t for k in ["euc","ai ","naigpt"]): score -= 0.6
    return score

def llm_pick_services(
    requirements_text: str,
    *,
    providers: Optional[List[str]] = None,
    shortlist_rows: Optional[List[Dict[str, Any]]] = None,
    limit: int = 8,
    required_platforms: Optional[List[str]] = None,
    allowed_families: Optional[List[str]] = None,
    scope_context: Optional[Dict[str, Any]] = None,
) -> List[int]:
    log.info("Starting LLM-based service selection")

    shortlist_rows = shortlist_rows or []
    if not shortlist_rows:
        base = f"{SUPABASE_URL}/rest/v1/proposals_updated"
        fams = (allowed_families or ["NC2","NDB","NCI","NUS","NKP"])
        plats = (required_platforms or ["azure"])
        for fam in fams:
            params_base = {
                "select": "id,category_name,service_name,service_type,product_family,target_platforms,priority_score,popularity_score",
                "product_family": f"eq.{fam}",
                "order": "priority_score.desc.nullslast,popularity_score.desc.nullslast",
                "limit": "120",
            }
            for tp in plats:
                params = dict(params_base)
                params["target_platforms"] = f"ov.{{{tp}}}"
                shortlist_rows += _fetch_supabase(base, params)
            # also pull generic rows with null targets
            params_null = dict(params_base)
            shortlist_rows += _fetch_supabase(base, params_null)

    db_preview = "\n".join(
        [f"- {r['id']}: {r['service_name']} ({r['product_family']}, {r['service_type']})"
         for r in shortlist_rows[:30]]
    )
    provider_txt = ", ".join(providers or [])
    scope_json = json.dumps(scope_context or {}, ensure_ascii=False)
    platform_txt = ", ".join(required_platforms or [])
    family_txt = ", ".join(allowed_families or [])

    prompt = f"""
You are a Nutanix Professional Services recommender.
Pick the best {limit} service IDs for the project. Use only IDs from the list.

Available services (sample):
{db_preview}

Project scope:
{requirements_text}

Providers: {provider_txt}
Target Platforms: {platform_txt}
Product Families: {family_txt}
Scope Context: {scope_json}

Rules:
- Prefer matching product_family and target_platform.
- Include assessment if sizing/discovery implied.
- Include deployment for NC2/NCI when moving to cloud/AHV.
- Include migration for VMs; include database services for DB move.
- Avoid EUC/OTHER unless EUC/AI is in scope.
Return JSON only: {{"ids":[<ids>]}}.
"""
    llm_raw = _azure_chat(prompt, max_tokens=600) or ""
    ids: List[int] = []
    if llm_raw:
        try:
            parsed = json.loads(llm_raw)
            ids = [int(x) for x in parsed.get("ids", [])]
        except Exception:
            ids = [int(x) for x in re.findall(r"\b\d{1,5}\b", llm_raw)]

    if not ids:
        hits = []
        for r in shortlist_rows:
            score = _score_match(requirements_text, r)
            if score > 0:
                hits.append((score, r["id"]))
        hits.sort(reverse=True)
        ids = [hid for _, hid in hits[:limit]]

    if not ids:
        ids = [r["id"] for r in shortlist_rows[:limit]]

    log.info(f"LLM selected service IDs: {ids}")
    return ids
