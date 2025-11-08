# app/services/llm_selector.py
from __future__ import annotations
import os, json, re, logging, requests
from typing import List, Dict, Any, Optional

log = logging.getLogger("llm_selector")

# =====================================================================
# ENVIRONMENT
# =====================================================================
AZURE_ENDPOINT   = os.getenv("AZURE_ENDPOINT", "").rstrip("/")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "")
AZURE_API_KEY    = os.getenv("AZURE_API_KEY", "")
SUPABASE_URL     = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY     = os.getenv("SUPABASE_KEY", "")

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def _fetch_supabase(url: str, params: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fetch rows from Supabase REST API."""
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Supabase fetch failed: {e}")
        return []


def _azure_chat(prompt: str, max_tokens: int = 400) -> Optional[str]:
    """Call Azure OpenAI chat completions."""
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
    """Simple keyword-based score to rank services."""
    t = text.lower()
    name = (row.get("service_name") or "").lower()
    fam = (row.get("product_family") or "").lower()
    stype = (row.get("service_type") or "").lower()
    score = 0.0
    if "nc2" in t and "nc2" in name: score += 2.0
    if "azure" in t and any("azure" in (p or "").lower() for p in (row.get("target_platforms") or [])): score += 2.0
    if "aws" in t and any("aws" in (p or "").lower() for p in (row.get("target_platforms") or [])): score += 2.0
    if "db" in t or "database" in t:
        if "ndb" in name or fam == "ndb": score += 2.0
    if "migration" in t and "migration" in stype: score += 1.5
    if "deployment" in t and "deployment" in stype: score += 1.0
    if "assessment" in t and "assessment" in stype: score += 1.0
    if "dr" in t or "disaster" in t:
        if "dr" in name or "recovery" in name: score += 1.5
    return score


# =====================================================================
# MAIN FUNCTION
# =====================================================================

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
    """
    Unified AI + fallback service selector.
    Compatible with plan_suggestions() arguments.
    """
    log.info("Starting LLM-based service selection")

    # ---------------------------------------------------------------
    # STEP 1: Load shortlist if not provided
    # ---------------------------------------------------------------
    shortlist_rows = shortlist_rows or []
    if not shortlist_rows:
        base = f"{SUPABASE_URL}/rest/v1/proposals_updated"
        for fam in (allowed_families or ["NC2", "NDB"]):
            params = {
                "select": "id,category_name,service_name,service_type,product_family,target_platforms,"
                          "priority_score,popularity_score",
                "product_family": f"eq.{fam}",
                "order": "priority_score.desc.nullslast,popularity_score.desc.nullslast",
                "limit": "60",
            }
            for tp in (required_platforms or ["azure"]):
                params["target_platforms"] = f"ov.{{{tp}}}"
            shortlist_rows += _fetch_supabase(base, params)

    # ---------------------------------------------------------------
    # STEP 2: Prepare LLM prompt
    # ---------------------------------------------------------------
    db_preview = "\n".join(
        [f"- {r['id']}: {r['service_name']} ({r['product_family']}, {r['service_type']})"
         for r in shortlist_rows[:20]]
    )
    provider_txt = ", ".join(providers or [])
    scope_json = json.dumps(scope_context or {}, indent=2)
    platform_txt = ", ".join(required_platforms or [])
    family_txt = ", ".join(allowed_families or [])

    prompt = f"""
You are a Nutanix Professional Services recommender.
Choose the most relevant service IDs from this list for the following project.

Available services:
{db_preview}

Scope:
{requirements_text}

Providers: {provider_txt}
Target Platforms: {platform_txt}
Product Families: {family_txt}
Scope Context: {scope_json}

Return valid JSON only:
{{"ids":[<matching ids>],"reasoning":"short explanation"}}.
"""

    # ---------------------------------------------------------------
    # STEP 3: Call Azure LLM
    # ---------------------------------------------------------------
    llm_raw = _azure_chat(prompt, max_tokens=500) or ""
    ids, reasoning = [], ""
    if llm_raw:
        try:
            parsed = json.loads(llm_raw)
            ids = [int(x) for x in parsed.get("ids", [])]
            reasoning = parsed.get("reasoning", "")
        except Exception:
            ids = [int(x) for x in re.findall(r"\b\d{1,4}\b", llm_raw)]
            reasoning = llm_raw[:200]

    # ---------------------------------------------------------------
    # STEP 4: Fallback matching if LLM empty
    # ---------------------------------------------------------------
    if not ids:
        hits = []
        for r in shortlist_rows:
            score = _score_match(requirements_text, r)
            if score > 0:
                hits.append((score, r["id"]))
        hits.sort(reverse=True)
        ids = [hid for _, hid in hits[:limit]]
        reasoning = reasoning or "Fallback keyword-based selection"

    # ---------------------------------------------------------------
    # STEP 5: Final safety fallback
    # ---------------------------------------------------------------
    if not ids:
        log.warning("LLM returned no matches; using default core services")
        ids = [r["id"] for r in shortlist_rows[:limit]]

    log.info(f"LLM selected service IDs: {ids}")
    return ids
