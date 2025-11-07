# app/services/duration_estimator.py
from __future__ import annotations
import os, json, logging, requests
from typing import List, Dict, Any, Optional

log = logging.getLogger("duration_estimator")

# -------- ENV --------
# Preferred path
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", "").rstrip("/")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "serper").lower()

# Fallback path (Azure)
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "").rstrip("/")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-10-01")

# -------- Helpers: search --------
def _serper_search(q: str, max_results: int = 6) -> List[Dict[str, Any]]:
    if not SERPER_API_KEY:
        log.info("serper api key missing")
        return []
    url = "https://google.serper.dev/search"
    payload = {"q": q, "num": max_results}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    if r.status_code >= 400:
        log.warning("serper search http %s: %s", r.status_code, r.text[:200])
        return []
    data = r.json()
    results = (data.get("organic") or []) + (data.get("knowledgeGraph") or [])
    out = []
    for it in results[:max_results]:
        out.append({
            "title": it.get("title") or it.get("name") or "",
            "url": it.get("link") or it.get("url") or "",
            "content": it.get("snippet") or it.get("description") or "",
        })
    return out

def _web_snippets(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    try:
        if SEARCH_PROVIDER == "serper":
            res = _serper_search(query, max_results=max_results)
            log.info("search provider=serper, hits=%d", len(res))
            return res
        # default to serper anyway
        res = _serper_search(query, max_results=max_results)
        log.info("search provider=default-serper, hits=%d", len(res))
        return res
    except Exception as e:
        log.warning("web search failed: %s", e)
        return []

# -------- Helpers: LLM --------
def _azure_chat(messages: List[Dict[str, str]], max_tokens: int = 200, timeout_s: int = 25) -> str:
    if not (AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_KEY):
        raise RuntimeError("Azure LLM env not set")
    url = f"{AZURE_ENDPOINT}/chat/completions?api-version={AZURE_API_VERSION}"
    body = {
        "model": AZURE_DEPLOYMENT,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "top_p": 1.0,
    }
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    r = requests.post(url, headers=headers, json=body, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def _gemini_estimate(prompt: str, max_output_tokens: int = 256, timeout_s: int = 25) -> str:
    """
    Minimal REST call for non-streaming generateContent.
    """
    if not (GEMINI_ENDPOINT and GEMINI_API_KEY and GEMINI_MODEL):
        raise RuntimeError("Gemini env not set")
    url = f"{GEMINI_ENDPOINT}/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_output_tokens,
            "topP": 1.0,
            "topK": 1
        }
    }
    r = requests.post(url, json=body, timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"gemini http {r.status_code}: {r.text[:200]}")
    data = r.json()
    # Extract text
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        log.warning("gemini unexpected payload: %s", json.dumps(data)[:400])
        return ""

# -------- Prompts --------
_SYSTEM = (
    "You are a professional services estimator. Given a task description and a few web snippets, "
    "output ONLY a JSON object with an integer field 'days' estimating the typical effort "
    "in business days for a senior consultant team. Prefer conservative but realistic estimates. "
    "No text besides the JSON."
)

_USER_TMPL = """Task:
{task}

Context:
- Client industry: {industry}
- Deployment type: {deploy_type}
- Source platform: {source}
- Target platform: {target}
- VM count: {vms}
- Nodes: {nodes}

Evidence (snippets):
{snippets}

Rules:
- Consider planning, execution, validation, and basic handover.
- If range, return the upper bound as an integer.
- If not enough info, infer from similar scope in the snippets.

Return:
{{"days": <integer>}}
"""

def _build_prompt(*, task_text: str, industry: Optional[str], deployment_type: Optional[str],
                  source_platform: Optional[str], target_platform: Optional[str],
                  vm_count: Optional[int], node_count: Optional[int],
                  snippets: List[Dict[str, Any]]) -> str:
    snip_txt = "\n".join([
        f"- {s.get('title') or ''}: {s.get('content') or ''} (src: {s.get('url') or ''})"[:400]
        for s in snippets
    ])[:1800] or "- no external snippets found"
    return _USER_TMPL.format(
        task=(task_text or "")[:500],
        industry=industry or "unspecified",
        deploy_type=deployment_type or "unspecified",
        source=source_platform or "unspecified",
        target=target_platform or "unspecified",
        vms=vm_count or 0,
        nodes=node_count or 0,
        snippets=snip_txt
    )

# -------- Public API --------
def estimate_days_from_web(
    *,
    task_text: str,
    industry: Optional[str],
    deployment_type: Optional[str],
    source_platform: Optional[str],
    target_platform: Optional[str],
    vm_count: Optional[int],
    node_count: Optional[int],
    search_hints: List[str] | None = None,
) -> Optional[int]:
    """
    Search the web and ask an LLM to output {"days": <int>}.
    Primary: Serper + Gemini. Fallback: Azure if Gemini env missing.
    """
    try:
        # Build query
        q_parts = [task_text or ""]
        if industry: q_parts.append(industry)
        if deployment_type: q_parts.append(deployment_type)
        if source_platform: q_parts.append(source_platform)
        if target_platform: q_parts.append(target_platform)
        if vm_count: q_parts.append(f"{vm_count} VMs")
        if node_count: q_parts.append(f"{node_count} nodes")
        if search_hints: q_parts.extend(search_hints)
        query = " ".join([p for p in q_parts if p]).strip()

        snippets = _web_snippets(query, max_results=6)
        log.info("duration_estimator: query='%s' snippets=%d", query, len(snippets))

        user_prompt = _build_prompt(
            task_text=task_text,
            industry=industry,
            deployment_type=deployment_type,
            source_platform=source_platform,
            target_platform=target_platform,
            vm_count=vm_count,
            node_count=node_count,
            snippets=snippets
        )

        # Prefer Gemini; fallback to Azure if Gemini not configured
        if GEMINI_ENDPOINT and GEMINI_API_KEY and GEMINI_MODEL:
            content = _gemini_estimate(prompt=f"{_SYSTEM}\n\n{user_prompt}", max_output_tokens=128, timeout_s=30)
            if not content:
                log.info("gemini returned empty content")
        else:
            log.info("gemini env missing, falling back to azure")
            content = _azure_chat(
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": user_prompt}],
                max_tokens=120,
                timeout_s=30
            )

        # Parse JSON
        raw_preview = (content or "").strip()
        log.info("duration_estimator raw LLM: %s", raw_preview[:200].replace("\n", " "))
        start = raw_preview.find("{")
        end = raw_preview.rfind("}")
        if start == -1 or end == -1 or end <= start:
            log.warning("Estimator LLM did not return JSON. content=%s", raw_preview[:400])
            return None

        obj = json.loads(raw_preview[start:end+1])
        days = obj.get("days", None)
        try:
            days_i = int(days)
            if days_i <= 0:
                return None
            return min(days_i, 365)
        except Exception:
            log.warning("Estimator JSON non-integer days: %r", days)
            return None

    except Exception as e:
        log.warning("estimate_days_from_web failed: %s", e)
        return None

def pick_days_with_rule(db_days: int, ai_days: Optional[int]) -> int:
    """
    Rule:
      - If ai_days < db_days -> keep db_days
      - Else use ai_days
      - If ai_days None -> db_days
    """
    if ai_days is None:
        return db_days
    if ai_days < db_days:
        return db_days
    return ai_days
