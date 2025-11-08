# app/services/duration_estimator.py
from __future__ import annotations
import os, json, logging, re, requests
from typing import Any, Dict, List, Optional

log = logging.getLogger("duration_estimator")

# --- ENVIRONMENT VARIABLES ---
SERPER_API_KEY   = os.getenv("SERPER_API_KEY", "")
SEARCH_PROVIDER  = (os.getenv("SEARCH_PROVIDER", "serper") or "serper").lower()

# Gemini fallback
GEMINI_ENDPOINT  = os.getenv("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com").rstrip("/")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")

# Azure primary provider
AZURE_ENDPOINT   = os.getenv("AZURE_ENDPOINT", "").rstrip("/")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "")
AZURE_API_KEY    = os.getenv("AZURE_API_KEY", "")
LLM_PROVIDER     = (os.getenv("LLM_PROVIDER", "azure") or "azure").lower()

LLM_CONNECT_TIMEOUT = int(os.getenv("LLM_CONNECT_TIMEOUT", "15"))
LLM_READ_TIMEOUT    = int(os.getenv("LLM_READ_TIMEOUT", "180"))


# -------------------------------------------------------------------------
#  LLM CALL HANDLER
# -------------------------------------------------------------------------
def call_llm(prompt: str, *, max_tokens: int = 400, temperature: float = 0.0) -> Optional[str]:
    """Call Azure (preferred) or Gemini fallback."""
    if not prompt:
        return None

    # --- Azure route ---
    if LLM_PROVIDER in ("azure", "ms", "microsoft"):
        if not (AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_KEY):
            log.warning("Azure configuration missing, skipping LLM call")
            return None
        try:
            url = f"{AZURE_ENDPOINT}/chat/completions"
            headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
            payload = {
                "model": AZURE_DEPLOYMENT,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(temperature),
                "max_tokens": int(min(max_tokens, 800)),
                "stream": False,
            }
            log.info("Calling Azure Chat Completions: %s (model=%s)", url, AZURE_DEPLOYMENT)
            resp = requests.post(
                url, headers=headers, json=payload,
                timeout=(LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT)
            )
            if resp.status_code == 429:
                log.warning("Azure rate limit 429")
                return None
            if resp.status_code >= 400:
                log.error(f"Azure LLM error {resp.status_code}: {resp.text[:300]}")
                return None
            data = resp.json()
            return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        except Exception as e:
            log.error(f"Azure LLM call failed: {e}")
            return None

    # --- Gemini fallback route ---
    if GEMINI_API_KEY:
        try:
            url = f"{GEMINI_ENDPOINT}/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
            body = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(url, json=body, timeout=(LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT))
            if r.status_code >= 400:
                log.error(f"Gemini error {r.status_code}: {r.text[:300]}")
                return None
            jr = r.json()
            return ((jr.get("candidates") or [{}])[0].get("content") or {}).get("parts", [{}])[0].get("text", "")
        except Exception as e:
            log.error(f"Gemini LLM call failed: {e}")
            return None

    log.warning("No LLM provider configured")
    return None


# -------------------------------------------------------------------------
#  SERPER WEB SEARCH
# -------------------------------------------------------------------------
def _serper_search(q: str, num: int = 6) -> List[str]:
    """Query Serper and return text snippets."""
    if not SERPER_API_KEY:
        log.warning("SERPER_API_KEY not configured, skipping search")
        return []
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    body = {"q": q, "num": max(1, min(num, 10))}

    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        snippets: List[str] = []
        for grp in ("knowledgeGraph", "answerBox", "organic", "peopleAlsoAsk", "topStories"):
            items = data.get(grp)
            if isinstance(items, dict):
                for v in items.values():
                    if isinstance(v, str):
                        snippets.append(v)
            elif isinstance(items, list):
                for it in items:
                    for key in ("snippet", "title", "answer", "description"):
                        val = it.get(key)
                        if isinstance(val, str) and val:
                            snippets.append(val)
        log.info(f"Serper search returned {len(snippets)} snippets for: {q[:80]}")
        return snippets[:10]
    except Exception as e:
        log.error(f"Serper search failed: {e}")
        return []


# -------------------------------------------------------------------------
#  MAIN AI ESTIMATOR
# -------------------------------------------------------------------------
def estimate_days_ai(service_name: str, *, context: Dict[str, Any]) -> Optional[int]:
    """
    Estimate delivery duration (in days) using LLM and optional web search.
    Returns integer or None.
    """
    svc = (service_name or "").strip()
    if not svc:
        return None

    # Construct natural query
    qry = f"{svc} typical duration days estimate Nutanix Professional Services"
    if context:
        parts = []
        if context.get("target_platform"):
            parts.append(str(context["target_platform"]))
        if context.get("source_platform"):
            parts.append("from " + str(context["source_platform"]))
        if context.get("vm_count"):
            parts.append(f"{int(context['vm_count'])} VMs")
        if context.get("node_count"):
            parts.append(f"{int(context['node_count'])} nodes")
        if parts:
            qry = f"{svc} {' '.join(parts)} typical duration days estimate"

    log.info("Estimator query: %s", qry)

    # Search for supporting snippets
    snippets = _serper_search(qry, num=6) if SEARCH_PROVIDER == "serper" else []
    log.info("Snippets: %d", len(snippets))

    # Prepare evidence text
    snippet_text = "- " + "\n- ".join(snippets[:8]) if snippets else "No evidence available"

    # Build prompt clearly
    prompt = f"""You are estimating delivery duration in days for a Nutanix Professional Services engagement.

Service: {svc}
Context: {json.dumps(context, ensure_ascii=False)}

Evidence snippets:
{snippet_text}

Rules:
- Return ONLY an integer number of days. No words. No units.
- If unclear, output your best conservative integer guess.
- Example valid answers: 3, 5, 10

Answer:"""

    try:
        out = call_llm(prompt, max_tokens=20, temperature=0.0)
        if not out:
            log.warning(f"LLM returned no output for service: {svc}")
            return None

        # Extract the first standalone integer from model output
        m = re.search(r"\b(\d{1,3})\b(?!(\s*(%|percent|year)))", out.lower())
        if not m:
            log.warning(f"Could not extract days from LLM output: {out[:120]}")
            return None

        days = int(m.group(1))
        log.info(f"AI estimated {days} days for: {svc}")
        return days
    except Exception as e:
        log.error(f"AI duration estimation failed: {e}")
        return None


# -------------------------------------------------------------------------
#  PUBLIC ENTRYPOINT (used by ranker)
# -------------------------------------------------------------------------
def estimate_days_from_web(
    task_text: str,
    *,
    industry: Optional[str] = None,
    deployment_type: Optional[str] = None,
    source_platform: Optional[str] = None,
    target_platform: Optional[str] = None,
    vm_count: Optional[int] = None,
    node_count: Optional[int] = None,
    search_hints: Optional[List[str]] = None,
) -> Optional[int]:
    """Wrapper for compatibility with ranker.py."""
    context = {
        "industry": industry,
        "deployment_type": deployment_type,
        "source_platform": source_platform,
        "target_platform": target_platform,
        "vm_count": vm_count,
        "node_count": node_count,
    }

    service_name = task_text.strip() if task_text else ""
    return estimate_days_ai(service_name, context=context)


# -------------------------------------------------------------------------
#  FINAL RULE FOR CHOOSING AI VS DB
# -------------------------------------------------------------------------
def pick_days_with_rule(db_days: int, ai_days: Optional[int]) -> int:
    """
    Pick final days conservatively.
      - If AI days >= db_days, prefer AI.
      - If AI shorter but within 70%, keep DB.
      - If AI much shorter, take average.
    """
    if ai_days is None:
        return max(1, db_days)

    if ai_days >= db_days:
        return max(1, ai_days)

    if ai_days >= (db_days * 0.7):
        return max(1, db_days)

    avg = int((db_days + ai_days) / 2)
    return max(1, avg)
