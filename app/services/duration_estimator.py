# app/services/duration_estimator.py
from __future__ import annotations
import os, json, logging, re, requests
from typing import Any, Dict, List, Optional

log = logging.getLogger("duration_estimator")

# ---------------------------
# ENVIRONMENT & CONFIG
# ---------------------------
# Keep configuration near top so it's obvious what can be tuned.
# Consider validating required vars at startup rather than at call-time.
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

# Timeouts for synchronous requests
LLM_CONNECT_TIMEOUT = int(os.getenv("LLM_CONNECT_TIMEOUT", "15"))
LLM_READ_TIMEOUT    = int(os.getenv("LLM_READ_TIMEOUT", "180"))


# -------------------------------------------------------------------------
#  LLM CALL HANDLER
# -------------------------------------------------------------------------
def call_llm(prompt: str, *, max_tokens: int = 400, temperature: float = 0.0) -> Optional[str]:
    """Call Azure (preferred) or Gemini fallback.

    Observations & recommendations:
    - This is synchronous and blocks the thread. If called from async endpoints,
      run in threadpool or provide an async variant.
    - Do not log full prompt or API keys; prompts may contain PII.
    - Respect provider rate limits and implement retry with backoff for 429/5xx.
    - For Azure, payload size and token accounting should be validated before sending.
    - The Gemini path uses a simple query param key pattern; prefer Authorization headers if provider supports it.
    """
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
            # INFO-level log is okay but avoid logging 'prompt' or secrets.
            log.info("Calling Azure Chat Completions: %s (model=%s)", url, AZURE_DEPLOYMENT)
            resp = requests.post(
                url, headers=headers, json=payload,
                timeout=(LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT)
            )
            # Handle rate limiting and errors – returning None is fine for best-effort.
            if resp.status_code == 429:
                log.warning("Azure rate limit 429")
                return None
            if resp.status_code >= 400:
                # log truncated body for debugging but avoid sensitive leakage
                log.error(f"Azure LLM error {resp.status_code}: {resp.text[:300]}")
                return None
            data = resp.json()
            # Defensive: structure can vary; handle missing keys gracefully
            return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        except Exception as e:
            # Log exception in structured way; consider counting metrics
            log.error(f"Azure LLM call failed: {e}")
            return None

    # --- Gemini fallback route ---
    # Note: uses query parameter key - review provider docs for auth best practice.
    if GEMINI_API_KEY:
        try:
            url = f"{GEMINI_ENDPOINT}/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
            body = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(url, json=body, timeout=(LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT))
            if r.status_code >= 400:
                log.error(f"Gemini error {r.status_code}: {r.text[:300]}")
                return None
            jr = r.json()
            # Defensive traversal; providers may change response shape
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
    """Query Serper and return text snippets.

    Observations:
    - This is best-effort and returns up to 10 snippets.
    - Uses a 30s timeout; that's large. Consider lowering and/or making retry/backoff.
    - The logic extracts text from multiple keys; it's robust but be careful with response shape changes.
    - Consider caching popular queries to avoid API costs.
    """
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
        # Pull text from several places — good for coverage
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
        # Consider distinguishing between transient network errors and permanent issues
        log.error(f"Serper search failed: {e}")
        return []


# -------------------------------------------------------------------------
#  MAIN AI ESTIMATOR
# -------------------------------------------------------------------------
def estimate_days_ai(service_name: str, *, context: Dict[str, Any]) -> Optional[int]:
    """
    Estimate delivery duration (in days) using LLM and optional web search.
    Returns integer or None.

    Key points:
    - This is a best-effort estimator using noisy sources (web + LLM).
    - We try to coax an integer-only response from the LLM using a strict prompt.
    - The regex used to extract integers is conservative but may still pick the wrong number
      if the model returns explanation text. Consider stricter enforcement or post-validation.
    - Consider caching results by (service_name, context hash) to avoid repeated LLM calls.
    """
    svc = (service_name or "").strip()
    if not svc:
        return None

    # Construct natural query for web evidence
    qry = f"{svc} typical duration days estimate Nutanix Professional Services"
    if context:
        parts = []
        if context.get("target_platform"):
            parts.append(str(context["target_platform"]))
        if context.get("source_platform"):
            parts.append("from " + str(context["source_platform"]))
        if context.get("vm_count"):
            # defensive: ensure ints
            try:
                parts.append(f"{int(context['vm_count'])} VMs")
            except Exception:
                pass
        if context.get("node_count"):
            try:
                parts.append(f"{int(context['node_count'])} nodes")
            except Exception:
                pass
        if parts:
            qry = f"{svc} {' '.join(parts)} typical duration days estimate"

    log.info("Estimator query: %s", qry)

    # Search for supporting snippets if provider is serper
    snippets = _serper_search(qry, num=6) if SEARCH_PROVIDER == "serper" else []
    log.info("Snippets: %d", len(snippets))

    # Prepare evidence text for the prompt
    snippet_text = "- " + "\n- ".join(snippets[:8]) if snippets else "No evidence available"

    # Build prompt clearly and instruct model to return only an integer.
    # NOTE: LLMs are fallible; keep explicit examples and rules.
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
        # Caveats:
        # - This regex will pick up any 1-3 digit number not followed by %/year.
        # - If model returns "I think 7 but could be 8", we will extract 7.
        # - Consider stricter parsing: require whole-response match like r'^\s*\d+\s*$'
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
    """Wrapper for compatibility with ranker.py.

    Minor note: search_hints parameter is accepted but not used. Either consume it
    (e.g., pass as parts into query) or remove it to avoid confusion.
    """
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
    Comments:
    - This is a business policy embedded here. Consider moving thresholds to config/env.
    - Add unit tests covering boundary cases (e.g., ai = 0, ai very small).
    """
    if ai_days is None:
        return max(1, db_days)

    if ai_days >= db_days:
        return max(1, ai_days)

    if ai_days >= (db_days * 0.7):
        return max(1, db_days)

    avg = int((db_days + ai_days) / 2)
    return max(1, avg)
