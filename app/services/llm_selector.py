# app/services/llm_selector.py
from __future__ import annotations
import os
import json
import logging
import requests
from typing import List, Dict, Any

log = logging.getLogger("llm_selector")

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "").rstrip("/")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-10-01")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure").lower()


def _azure_chat(messages: List[Dict[str, str]], timeout_s: int = 20) -> str:
    if not (AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_KEY):
        raise RuntimeError("Azure LLM env not set: AZURE_ENDPOINT/AZURE_DEPLOYMENT/AZURE_API_KEY")
    url = f"{AZURE_ENDPOINT}/chat/completions?api-version={AZURE_API_VERSION}"
    body = {
        "model": AZURE_DEPLOYMENT,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 400,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    r = requests.post(url, headers=headers, json=body, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _build_catalog(shortlist_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cat = []
    for r in shortlist_rows:
        cat.append({
            "id": r.get("id"),
            "name": r.get("service_name"),
            "family": r.get("product_family"),
            "category": r.get("category_name"),
            "type": r.get("service_type"),
            "platforms": r.get("target_platforms") or [],
            "priority": r.get("priority_score", 0),
            "popularity": r.get("popularity_score", 0),
        })
    return cat


_SYSTEM = (
    "You are a Nutanix services architect. Select the MINIMUM set of services to deliver the stated scope. "
    "Only include essential services. Return strict JSON only."
)

_USER_TMPL = """Client Requirements:
{query}

SCOPE CONTEXT:
- VMs to migrate: {vm_count}
- Nodes to configure: {node_count}
- Source platform: {source_platform}
- Migration needed: {needs_migration}
- Deployment needed: {needs_deployment}

Providers: {providers}
Required platforms: {platforms}
Product families: {families}
Max items to return: {limit}

Available Services (shortlist):
{catalog}

RULES:
1) If scope says "VM migration + N nodes", suggest:
   - 1 Assessment (FitCheck)
   - 1 Infrastructure service (Deployment/Expansion)
   - 1 Migration service (prefer Nutanix Move for VMware/HyperFlex)
2) Do not add unrelated services.
3) Keep it minimal, usually 2-3 items.

Return JSON exactly:
{{
  "ids": [<ordered service IDs>],
  "reasoning": "<brief one-line explanation>"
}}
"""


def llm_pick_services(
    *,
    query_text: str,
    providers: List[str],
    shortlist_rows: List[Dict[str, Any]],
    limit: int,
    required_platforms: List[str],
    allowed_families: List[str],
    scope_context: Dict[str, Any] | None = None,
) -> List[int]:
    try:
        scope_context = scope_context or {}
        catalog = _build_catalog(shortlist_rows[:60])

        user = _USER_TMPL.format(
            query=query_text.strip(),
            vm_count=scope_context.get("vm_count", 0) or "Not specified",
            node_count=scope_context.get("node_count", 0) or "Not specified",
            source_platform=scope_context.get("source_platform", "Not specified") or "Not specified",
            needs_migration="Yes" if scope_context.get("needs_migration") else "No",
            needs_deployment="Yes" if scope_context.get("needs_deployment") else "No",
            providers=", ".join(providers) if providers else "none",
            platforms=", ".join(required_platforms) if required_platforms else "any",
            families=", ".join(allowed_families) if allowed_families else "any",
            limit=max(1, limit),
            catalog=json.dumps(catalog, ensure_ascii=False, indent=2),
        )

        if LLM_PROVIDER != "azure":
            log.warning("Unsupported LLM_PROVIDER=%s. Skipping LLM pick.", LLM_PROVIDER)
            return []

        content = _azure_chat(
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": user}],
            timeout_s=25,
        )

        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            log.warning("LLM content not JSON: %s", content[:200])
            return []

        obj = json.loads(content[start:end + 1])
        ids = obj.get("ids") or []
        ids = [int(x) for x in ids if isinstance(x, (int, str)) and str(x).isdigit()]

        allowed_ids = {r["id"] for r in shortlist_rows}
        ids = [i for i in ids if i in allowed_ids][:max(1, limit)]

        if obj.get("reasoning"):
            log.info("LLM reasoning: %s", obj["reasoning"])

        return ids

    except Exception as e:
        log.warning("llm_pick_services failed: %s", e)
        return []
