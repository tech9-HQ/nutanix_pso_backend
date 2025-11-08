# app/services/keyword_expander.py
import json, httpx, os, logging, time
from functools import lru_cache

log = logging.getLogger("keyword_expander")
TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)

AZOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZOAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZOAI_DEPLOYMENT_CHAT = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT", "grok-3")

def _heuristic(intent: str, product_family: str):
    t = (intent or "").lower()
    toks = []
    for w in [
        "assessment","readiness","design workshop","migration","vmware","vsphere","hyperflex",
        "cutover","dr","disaster recovery","replication","protection domain","metro",
        "ndb","database","oracle","sql","postgres","mysql","mongodb",
        "nc2","nci","azure","aws","ahv","sizing","evaluation","health check","fitcheck",
        "infrastructure","documentation","workshop"
    ]:
        if w in t:
            toks.append(w)
    return {
        "mandatory": [],
        "desirable": sorted(set(toks)),
        "negative": [],
        "synonyms": [],
        "tags": [product_family.lower()] if product_family else []
    }

@lru_cache(maxsize=256)
def expand(intent: str, product_family: str):
    if not (AZOAI_ENDPOINT and AZOAI_KEY):
        return _heuristic(intent, product_family)

    url = f"{AZOAI_ENDPOINT}/openai/deployments/{AZOAI_DEPLOYMENT_CHAT}/chat/completions?api-version=2024-10-01-preview"
    system = (
        "You produce compact search tokens to find consulting services in a database. "
        "Return strict JSON with keys: mandatory, desirable, negative, synonyms, tags."
    )
    user = f"Product family: {product_family}\nIntent: {intent}\nReturn only JSON."
    payload = {"messages":[{"role":"system","content":system},{"role":"user","content":user}],
               "temperature":0.2,"max_tokens":300}
    try:
        with httpx.Client(timeout=TIMEOUT) as cx:
            r = cx.post(url, headers={"api-key": AZOAI_KEY, "Content-Type":"application/json"}, json=payload)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
        txt = txt[txt.find("{"): txt.rfind("}")+1]
        data = json.loads(txt)
        norm = {k: [str(v).strip() for v in data.get(k, []) if str(v).strip()]
                for k in ("mandatory","desirable","negative","synonyms","tags")}
        if product_family and product_family.lower() not in [t.lower() for t in norm["tags"]]:
            norm["tags"].append(product_family.lower())
        return norm
    except Exception as e:
        log.warning(f"AI expand failed: {e}")
        return _heuristic(intent, product_family)
