# app/services/ranker.py — Intelligent planner (one suggestion, tier-gated, phase-aware)
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Set
import json, re, logging, os
import numpy as np
import httpx

from app.models.schemas import SuggestPlanRequest
from app.services.proposals_repo import suggest_services_repo
from app.services.duration_estimator import estimate_days_from_web, pick_days_with_rule, call_llm
from app.services.company_size import get_company_size

log = logging.getLogger("ranker")

# optional repo helpers (if available)
try:
    from app.services import repo  # repo.build_or_ilike, repo.fetch_candidates_smart
except Exception:
    repo = None

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SERVICE_TABLE = os.getenv("SERVICE_TABLE", "proposals_updated")

AZOAI_ENDPOINT = os.getenv("AZOAI_ENDPOINT")
AZOAI_KEY = os.getenv("AZOAI_KEY")
AZOAI_DEPLOYMENT_EMBED = os.getenv("AZOAI_DEPLOYMENT_EMBED")


_ai_scope_cache = {}
_embed_cache = {}

# ------------------ utilities ------------------

def _tokenize(s: str) -> List[str]:
    if not s:
        return []
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return s.split()

def _safe_json_loads(text: str) -> Dict[str, Any] | List[Any]:
    try:
        return json.loads(text or "")
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", text or "", flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return {}
        return {}

def _simple_tokenize(text: Optional[str]) -> List[str]:
    if not text:
        return []
    toks = re.findall(r"\w+", text.lower())
    return [t for t in toks if len(t) > 2]

def _to_py(val: Any):
    if val is None:
        return None
    try:
        import numpy as _np
        if isinstance(val, (_np.integer,)):
            return int(val)
        if isinstance(val, (_np.floating,)):
            return float(val)
        if isinstance(val, _np.ndarray):
            return val.tolist()
    except Exception:
        pass
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return str(val)
    try:
        from decimal import Decimal
        if isinstance(val, Decimal):
            return float(val)
    except Exception:
        pass
    if isinstance(val, (list, tuple, set)):
        return [_to_py(v) for v in list(val)]
    if isinstance(val, dict):
        return {str(k): _to_py(v) for k, v in val.items()}
    return val

# ------------------ tier helpers ------------------

_TIER_KEYWORDS = {
    "starter":  ["starter","basic","essentials","essential","foundation","foundational"],
    "pro":      ["pro","standard","advanced","plus"],
    "ultimate": ["ultimate","enterprise","premium","platinum","ultra"],
}
_TIER_WORDS = {w for arr in _TIER_KEYWORDS.values() for w in arr}

def _infer_row_tier(row: Dict[str, Any]) -> Optional[str]:
    text = " ".join([
        str(row.get("service_name","")),
        str(row.get("positioning","")),
        " ".join(row.get("canonical_names") or []),
        str(row.get("category_name","")),
    ]).lower()
    for tier, words in _TIER_KEYWORDS.items():
        if any(w in text for w in words):
            return tier
    return None

def _tierless_key(row: Dict[str, Any]) -> str:
    name = (row.get("service_name") or "").lower()
    name = re.sub(
        r"\s*-\s*(starter|pro|ultimate|enterprise|premium|platinum|basic|essentials|standard|advanced|plus)\b",
        "",
        name,
    )
    name = re.sub(r"\b(" + "|".join(map(re.escape, _TIER_WORDS)) + r")\b", "", name)
    name = re.sub(r"\s{2,}", " ", name).strip()
    fam  = (row.get("product_family") or "").lower()
    cat  = (row.get("category_name") or "").lower()
    return f"{cat}|{fam}|{name}"

def _parse_int_maybe(v) -> Optional[int]:
    try:
        return int(str(v).replace(",", "").strip())
    except Exception:
        return None

def _desired_tier_for_size(n: Optional[int]) -> Optional[str]:
    if n is None: return None
    if n < 500:   return "starter"
    if n < 1000:  return "pro"
    return "ultimate"

def _extract_company_size_from_req(req) -> Optional[int]:
    for obj in (req, getattr(req, "constraints", None), getattr(req, "meta", None)):
        if not obj: continue
        for attr in ("company_size","employee_count","employees","org_size","size"):
            n = _parse_int_maybe(getattr(obj, attr, None))
            if n is not None:
                return n
    client = getattr(req, "client_name", None) or getattr(req, "company_name", None)
    return get_company_size(client)

# ------------------ keyword expansion ------------------

def expand_keywords(text: Optional[str], product_family: Optional[str] = None) -> Dict[str, List[str]]:
    tokens: Set[str] = set(_simple_tokenize(text or ""))
    if product_family:
        tokens.add(product_family.lower())

    synonym_map = {
        "ndb": ["ndb", "nutanix database service", "nutanix db service", "dbs", "database", "sql", "postgres"],
        "nci": ["nci", "prism", "flow", "ncm", "ahv"],
        "nus": ["nus", "nutanix unified storage", "files"],
        "nkp": ["nkp", "nutanix kubernetes platform", "kubernetes"],
        "nc2": ["nc2", "nutanix cloud clusters", "nutanix cloud cluster", "azure", "aws"],
    }

    synonyms: Set[str] = set()
    pf = (product_family or "").strip().lower()
    if pf in synonym_map:
        synonyms.update(synonym_map[pf])
    for t in list(tokens):
        if t in synonym_map:
            synonyms.update(synonym_map[t])

    if any(k in tokens for k in ("migrate", "migration", "migrating", "move", "rehost")):
        synonyms.update(["migration", "migrate", "move", "cutover"])
    if any(k in tokens for k in ("deploy", "deployment", "configure", "setup")):
        synonyms.update(["deployment", "deploy", "configure", "setup", "landing", "networking"])
    if any(k in tokens for k in ("dr", "disaster", "recovery", "protection", "replication", "metro", "near")):
        synonyms.update(["dr", "disaster", "recovery", "protection", "replication", "nearsync", "sync"])

    mandatory: Set[str] = set()
    desirable: Set[str] = set()
    tags: Set[str] = set()
    negative: Set[str] = set()

    must_includes = re.findall(r"must include[:\s]*([a-z0-9,\s\-\_]+)", (text or ""), flags=re.I)
    for group in must_includes:
        for token in re.split(r"[,\n]+", group):
            t = token.strip().lower()
            if t:
                mandatory.add(t)

    neg_matches = re.findall(r"do not include[:\s]*([a-z0-9,\s\-\_]+)", (text or ""), flags=re.I)
    for group in neg_matches:
        for token in re.split(r"[,\n]+", group):
            t = token.strip().lower()
            if t:
                negative.add(t)

    for t in tokens:
        if t not in {"the", "and", "for", "with", "to", "of", "in", "on"}:
            desirable.add(t)
    for s in synonyms:
        desirable.add(s)
    if pf:
        tags.add(pf)
        for s in synonym_map.get(pf, []):
            tags.add(s)

    if re.search(r"ahv-only|ahv only", (text or ""), flags=re.I):
        negative.add("ahv-only")

    return {
        "mandatory": sorted(mandatory),
        "desirable": sorted(desirable),
        "synonyms": sorted(list(synonyms)),
        "tags": sorted(tags),
        "negative": sorted(negative),
    }

# ------------------ AI scope analysis ------------------
def _intelligent_scope_analysis(req: SuggestPlanRequest) -> Dict[str, Any]:
    """
    LLM + rule-backed scope extraction with caching and normalization.
    Returns dict with:
      - source_platform, target_platform
      - product_families (subset of ["NCI","NC2","NDB","NKP","NUS"])
      - required_phases ({"phases":[...]}) and _required_phases_list
      - estimated_vm_count, estimated_db_count
      - hard_constraints: {"must_include":[], "must_exclude":[]}
    """

    # ---------- cache key ----------
    try:
        pf_list = (getattr(req, "constraints", None) and getattr(req.constraints, "product_families", None)) or []
        pf_from_constraints = pf_list[0] if pf_list else None
    except Exception:
        pf_from_constraints = None

    pf_hint = getattr(req, "product_family", None) or pf_from_constraints or ""
    key = f"{getattr(req,'requirements_text','')}|{pf_hint}|{getattr(req,'deployment_type','')}"

    if key in _ai_scope_cache:
        return _ai_scope_cache[key]

    # ---------- constants ----------
    VALID_SOURCES = {"vmware","hyperv","aws","azure","gcp","cisco_hyperflex"}
    VALID_TARGETS = {"ahv","aws","azure","gcp"}
    VALID_PF      = {"NCI","NC2","NDB","NKP","NUS"}
    VALID_PHASES  = ["assessment","deployment","migration","database","dr"]

    # ---------- prompt ----------
    try:
        boq_json = json.dumps([b.model_dump() for b in (req.boq or [])], ensure_ascii=False)
    except Exception:
        boq_json = "[]"

    prompt = f"""
You are a Nutanix Professional Services planner. Read the project details and return ONLY compact JSON.

Return strictly this JSON shape:
{{
  "source_platform": "vmware|hyperv|aws|azure|gcp|cisco_hyperflex|null",
  "target_platform": "ahv|aws|azure|gcp|null",
  "product_families": ["NCI","NC2","NDB","NKP","NUS"],
  "required_phases": ["assessment","deployment","migration","database","dr"],
  "estimated_vm_count": <integer>,
  "estimated_db_count": <integer>,
  "hard_constraints": {{
    "must_include": [<strings>],
    "must_exclude": [<strings>]
  }}
}}

Project:
- Industry: {req.industry}
- Deployment type: {req.deployment_type}
- Proposal type: {req.proposal_type}
- Requirements: {req.requirements_text}
- Providers: {list(req.providers or [])}
- Selected vendor: {req.selected_vendor}
- BOQ: {boq_json}

Respond with JSON only. No prose.
"""
    ai = call_llm(prompt, max_tokens=350, temperature=0.0) or ""
    data = _safe_json_loads(ai) or {}

    # ---------- normalize ----------
    def norm_platform(v: Optional[str], allowed: set[str]) -> Optional[str]:
        if not isinstance(v, str):
            return None
        v = v.strip().lower() or None
        return v if v in allowed else None

    sp = norm_platform(data.get("source_platform"), VALID_SOURCES)
    tp = norm_platform(data.get("target_platform"), VALID_TARGETS)

    # --- normalize product families ---
    fams_in = data.get("product_families") or []
    fams: list[str] = []
    for f in fams_in:
        if isinstance(f, str) and f.upper() in VALID_PF:
            fams.append(f.upper())

    # auto-infer NDB if database-related words exist
    ql = f"{req.requirements_text or ''}".lower()
    if any(k in ql for k in ["database","ndb","oracle","sql","postgres","mysql","db "]) and "NDB" not in fams:
        fams.append("NDB")

    # keep unique order
    seen_f = set()
    fams = [x for x in fams if not (x in seen_f or seen_f.add(x))]

    # --- normalize phases ---
    phases_in = data.get("required_phases") or []
    phases_norm = []
    for p in phases_in:
        if isinstance(p, str) and p.lower() in VALID_PHASES:
            phases_norm.append(p.lower())

    # rule fallback if LLM fails
    if not phases_norm:
        if any(k in ql for k in ["assess","fitcheck","readiness","size","poc"]):
            phases_norm.append("assessment")
        if any(k in ql for k in ["deploy","deployment","infrastructure","cluster","expand","setup","configure"]):
            phases_norm.append("deployment")
        if any(k in ql for k in ["migrate","migration","move","rehost","cutover"]) or re.search(r"\b\d+\s*vms?\b", ql):
            phases_norm.append("migration")
        if any(k in ql for k in ["database","ndb","oracle","sql","postgres","db "]):
            phases_norm.append("database")
        if any(k in ql for k in ["dr","disaster","recovery","replication","metro","nearsync","leap","protection"]):
            phases_norm.append("dr")

    # preserve canonical order
    ordered = []
    seen_p = set()
    for p in VALID_PHASES:
        if p in phases_norm and p not in seen_p:
            ordered.append(p)
            seen_p.add(p)

    # counts
    def to_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default

    est_vm = to_int(data.get("estimated_vm_count"), 0)
    est_db = to_int(data.get("estimated_db_count"), 0)

    # merge hard constraints
    hc = data.get("hard_constraints") or {}
    inc_ai = [s for s in (hc.get("must_include") or []) if isinstance(s, str)]
    exc_ai = [s for s in (hc.get("must_exclude") or []) if isinstance(s, str)]

    inc_req = []
    exc_req = []
    try:
        if getattr(req, "constraints", None):
            inc_req = list(getattr(req.constraints, "must_include", []) or [])
            exc_req = list(getattr(req.constraints, "must_exclude", []) or [])
    except Exception:
        pass

    def uniq_lower(seq):
        out, seen = [], set()
        for s in seq:
            if not isinstance(s, str):
                continue
            t = s.strip()
            if not t:
                continue
            k = t.lower()
            if k not in seen:
                out.append(t)
                seen.add(k)
        return out

    must_include = uniq_lower(inc_ai + inc_req)
    must_exclude = uniq_lower(exc_ai + exc_req)

    out = {
        "source_platform": sp,
        "target_platform": tp,
        "product_families": fams,
        "required_phases": {"phases": ordered},
        "_required_phases_list": list(ordered),
        "estimated_vm_count": est_vm,
        "estimated_db_count": est_db,
        "hard_constraints": {
            "must_include": must_include,
            "must_exclude": must_exclude,
        },
    }

    _ai_scope_cache[key] = out
    return out



# ------------------ rule scope extraction ------------------

def _extract_scope_details(req: SuggestPlanRequest) -> Dict[str, Any]:
    qtext = f"{req.requirements_text or ''} {' '.join([b.description or '' for b in (req.boq or [])])}".lower()

    def find_int(rx, default=0):
        m = re.search(rx, qtext)
        return int(m.group(1)) if m else default

    vm_count = max(find_int(r'(\d+)\s*vms?\b'), find_int(r'(\d+)\s*virtual\s*machines?\b'))
    db_count = max(find_int(r'(\d+)\s*databases?\b'), find_int(r'(\d+)\s*dbs?\b'))
    node_count = max(find_int(r'(\d+)\s*nodes?\b'), find_int(r'(\d+)\s*servers?\b'))
    sites = find_int(r'(\d+)\s*sites?\b')
    nodes_per_site = find_int(r'(\d+)\s*nodes?\s*per\s*site')

    source_platform = None
    if any(k in qtext for k in ['vmware','vsphere','vcenter','esxi']): source_platform = 'vmware'
    if any(k in qtext for k in ['hyperflex','cisco hyperflex','ucs hyperflex']): source_platform = 'cisco_hyperflex'
    if 'hyper-v' in qtext or 'hyperv' in qtext: source_platform = 'hyperv'
    if 'on-prem' in qtext or 'onprem' in qtext: source_platform = source_platform or 'onprem'

    target_platform = None
    if 'ahv' in qtext or 'acropolis' in qtext: target_platform = 'ahv'
    if 'nc2' in qtext and 'azure' in qtext: target_platform = 'azure'
    if 'nc2' in qtext and 'aws' in qtext: target_platform = 'aws'
    if 'azure' in qtext and not target_platform: target_platform = 'azure'
    if 'aws' in qtext and not target_platform: target_platform = 'aws'

    has_database = any(k in qtext for k in ['database','oracle','sql server','postgres','mysql','mongodb','rds','ndb'])
    wants_dr = any(k in qtext for k in ['dr','disaster','recovery','replication','nearsync','metro','leap'])
    wants_euc = any(k in qtext for k in ['euc','vdi','citrix','horizon'])
    wants_k8s = any(k in qtext for k in ['kubernetes','nkp','openshift'])
    wants_storage = any(k in qtext for k in ['nus','unified storage','files','nas','nfs','smb'])

    return {
        "vm_count": vm_count,
        "db_count": db_count,
        "node_count": node_count,
        "sites": sites,
        "nodes_per_site": nodes_per_site,
        "source_platform": source_platform,
        "target_platform": target_platform,
        "has_database": has_database or (db_count > 0),
        "wants_dr": wants_dr,
        "wants_euc": wants_euc,
        "wants_k8s": wants_k8s,
        "wants_storage": wants_storage,
        "has_boq": bool(req.boq),
    }


def _detect_needs(req: SuggestPlanRequest, scope: Dict[str, Any], ai_scope: Dict[str, Any]) -> Dict[str, Any]:
    tokens = set(_tokenize(
        f"{req.requirements_text or ''} {' '.join([b.description or '' for b in (req.boq or [])])}"
    ))
    return {
        "needs_assessment": (
            "assessment" in (ai_scope.get("required_phases", {}).get("phases") if isinstance(ai_scope.get("required_phases"), dict) else ai_scope.get("required_phases") or []) or
            not scope["has_boq"] or scope["vm_count"] > 0 or any(t in tokens for t in ['assess','fitcheck','sizing','poc'])
        ),
        "needs_deployment": (
            "deployment" in (ai_scope.get("required_phases", {}).get("phases") if isinstance(ai_scope.get("required_phases"), dict) else ai_scope.get("required_phases") or []) or
            scope["node_count"] > 0 or scope.get("needs_cluster_config", False) or any(t in tokens for t in ['deploy','setup','configure','infrastructure','expansion'])
        ),
        "needs_migration": (
            "migration" in (ai_scope.get("required_phases", {}).get("phases") if isinstance(ai_scope.get("required_phases"), dict) else ai_scope.get("required_phases") or []) or
            (scope["vm_count"] > 0 and scope["source_platform"] is not None) or
            any(t in tokens for t in ['migrate','migration','move','consolidate','rehost'])
        ),
        "needs_db_migration": (
            "database" in (ai_scope.get("required_phases", {}).get("phases") if isinstance(ai_scope.get("required_phases"), dict) else ai_scope.get("required_phases") or []) or
            (scope["has_database"] and any(t in tokens for t in ['migrate','migration','move']))
        ),
        "needs_dr": (
            "dr" in (ai_scope.get("required_phases", {}).get("phases") if isinstance(ai_scope.get("required_phases"), dict) else ai_scope.get("required_phases") or []) or
            any(t in tokens for t in ['dr','disaster','recovery','protection','near','metro','replication'])
        ),
    }

# ------------------ embedding helpers ------------------

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _embed(text: str) -> np.ndarray | None:
    if not text:
        return None
    if text in _embed_cache:
        return _embed_cache[text]
    if not (AZOAI_ENDPOINT and AZOAI_KEY and AZOAI_DEPLOYMENT_EMBED):
        return None
    url = f"{AZOAI_ENDPOINT}/openai/deployments/{AZOAI_DEPLOYMENT_EMBED}/embeddings?api-version=2024-10-01-preview"
    payload = {"input": text}
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0, read=20.0)) as cx:
            r = cx.post(url, headers={"api-key": AZOAI_KEY, "Content-Type": "application/json"}, json=payload)
            r.raise_for_status()
            emb = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
            _embed_cache[text] = emb
            return emb
    except Exception:
        return None


def _parse_emb(row) -> np.ndarray | None:
    v = row.get("embedding")
    if v is None:
        return None
    if isinstance(v, list):
        return np.array(v, dtype=np.float32)
    try:
        return np.array(json.loads(v), dtype=np.float32)
    except Exception:
        return None

# ------------------ constraint gating ------------------

def _honor_constraints(row: Dict[str, Any], req: SuggestPlanRequest, scope: Dict[str, Any], ai_scope: Dict[str, Any]) -> bool:
    """
    Dynamic constraint gate with no hardcoded service names.
    - Respect req.constraints.target_platforms
    - Respect must_exclude: ["AHV-only services"]
    - Respect allowed product_families if provided
    - Soft-exclude EUC/OTHER unless text implies EUC/AI
    """
    txt = f"{req.requirements_text or ''} {' '.join([b.description or '' for b in (req.boq or [])])}".lower()

    allowed_fams = set((getattr(req, "constraints", None) and (req.constraints.product_families or [])) or (ai_scope.get("product_families") or []))
    fam = (row.get("product_family") or "").upper()
    if allowed_fams and fam and fam not in allowed_fams:
        return False

    wanted_platforms = set((getattr(req, "constraints", None) and (req.constraints.target_platforms or [])) or [])
    targets = set([t.lower() for t in (row.get("target_platforms") or []) if t])
    if wanted_platforms:
        if targets and not targets.intersection(wanted_platforms):
            return False

    must_ex = set((getattr(req, "constraints", None) and (req.constraints.must_exclude or [])) or (ai_scope.get("hard_constraints", {}).get("must_exclude") or []))
    if "AHV-only services" in must_ex:
        if targets == {"ahv"} or ("ahv" in targets and not targets.intersection({"azure","aws","gcp"})):
            return False

    if (fam == "OTHER" or "euc" in (row.get("service_name","").lower())) and not any(k in txt for k in ["euc","end user","gold image","app layering","broker","naigpt","ai "]):
        return False

    # Drop Dell PowerFlex unless requested
    if "powerflex" in (row.get("service_name","").lower() + " " + row.get("positioning","").lower()) and "powerflex" not in txt:
        return False
    return True

def _dynamic_service_fetch(req: SuggestPlanRequest, scope: Dict[str, Any], ai_scope: Dict[str, Any]) -> List[Dict[str, Any]]:
    fam_from_req = list(getattr(req, "constraints", None) and (req.constraints.product_families or []) or [])
    fam_from_ai  = list(ai_scope.get("product_families") or [])
    families = fam_from_req or fam_from_ai or ["NCI","NC2","NDB","NKP","NUS"]

    if scope.get("has_database") and "NDB" not in families:
        families.append("NDB")
    if "NC2" in families and "NCI" not in families:
        families.append("NCI")

    req_platforms = list(getattr(req, "constraints", None) and (req.constraints.target_platforms or []) or [])
    ai_platform   = [ai_scope.get("target_platform")] if ai_scope.get("target_platform") else []
    platforms     = req_platforms or ai_platform

    queries = [None, "fitcheck", "assessment", "infrastructure", "deployment",
               "migration", "move", "database", "dr", "recovery", "protection", "metro"]

    seen: set[int] = set()
    out: List[Dict[str, Any]] = []

    for fam in families:
        fam_platforms = platforms if fam in ("NC2",) and platforms else None
        for q in queries:
            try:
                rows = suggest_services_repo(
                    product_family=fam,
                    platforms=fam_platforms,
                    limit=80,
                    service_type=None,
                    q=q,
                )
            except Exception:
                rows = []
            for r in rows:
                rid = r.get("id")
                if isinstance(rid, int) and rid not in seen:
                    seen.add(rid)
                    out.append(r)
    return out

# ------------------ scoring ------------------

def _kw_score(row: dict, tokens: dict) -> float:
    blob = " ".join([
        str(row.get("service_name") or ""),
        str(row.get("positioning") or ""),
        str(row.get("category_name") or ""),
        " ".join(row.get("canonical_names") or []),
        str(row.get("product_family") or "")
    ]).lower()
    s = 0.0
    for w in tokens.get("mandatory", []):
        if w.lower() in blob: s += 3.5
        else: s -= 2.0
    for w in tokens.get("desirable", []):
        if w.lower() in blob: s += 1.5
    for w in tokens.get("synonyms", []):
        if w.lower() in blob: s += 1.0
    for w in tokens.get("tags", []):
        if w.lower() in blob: s += 0.8
    for w in tokens.get("negative", []):
        if w.lower() in blob: s -= 3.0
    return s

def _calc_relevance(
    svc: Dict[str, Any],
    needs: Dict[str, Any],
    scope: Dict[str, Any],
    query_text: str,
    prefer_migration_ready: bool
) -> Tuple[float, List[str]]:
    """
    Computes weighted contextual relevance for a Nutanix service candidate.
    Balances keyword, family, platform, and phase intent alignment.
    """
    score = 0.0
    reasons: List[str] = []

    name = (svc.get("service_name") or "").lower()
    category = (svc.get("category_name") or "").lower()
    stype = (svc.get("service_type") or "").lower()
    family = (svc.get("product_family") or "").upper()
    targets = [t.lower() for t in (svc.get("target_platforms") or [])]
    qt = query_text.lower()

    # ---------------- PLATFORM ALIGNMENT ----------------
    if scope.get("target_platform"):
        if scope["target_platform"] in targets:
            score += 0.25; reasons.append(f"Targets {scope['target_platform']}")
        elif not targets:
            score += 0.05
        else:
            score -= 0.05

    # ---------------- FAMILY-SPECIFIC LOGIC ----------------
    is_nc2 = (family == "NC2")
    is_ndb = (family == "NDB")
    is_nkp = (family == "NKP")
    is_nus = (family == "NUS")

    # NC2 on Azure/AWS context boosts
    if is_nc2 and scope.get("target_platform") in {"azure", "aws"}:
        if "fitcheck" in name or "readiness" in name:
            score += 0.30; reasons.append(f"NC2 {scope['target_platform']} FitCheck")
        if any(k in name for k in ["dr","disaster","recovery","nearsync","metro","replication","leap"]):
            score += 0.25; reasons.append(f"NC2 {scope['target_platform']} DR")

    # Database (NDB) focus
    if scope.get("has_database") or any(k in qt for k in ["database","ndb","oracle","sql","postgres","db "]):
        if is_ndb or "database" in category or " ndb" in f" {name}":
            score += 0.40; reasons.append("Database scope")
            if prefer_migration_ready and stype in ("migration","design"):
                score += 0.10; reasons.append("Migration-ready DB service")

    # EUC / VDI scope
    if scope.get("wants_euc") and any(k in name for k in ["euc","vdi","citrix","horizon","broker"]):
        score += 0.35; reasons.append("EUC scope")

    # Kubernetes (NKP)
    if scope.get("wants_k8s") and (is_nkp or "kubernetes" in name or "nkp" in name):
        score += 0.35; reasons.append("Kubernetes scope")

    # Storage (NUS)
    if scope.get("wants_storage") and (is_nus or any(k in name for k in ["nus","files","unified storage","nas","nfs","smb"])):
        score += 0.35; reasons.append("Storage scope")

    # ---------------- PHASE LOGIC ----------------
    if needs.get("needs_assessment") and (stype == "assessment" or "assessment" in category or "fitcheck" in name):
        score += 0.20; reasons.append("Phase: assessment")

    if needs.get("needs_deployment") and (stype == "deployment" or "deployment" in name or "infrastructure" in name or "cluster" in name):
        score += 0.25; reasons.append("Phase: deployment")
        if scope.get("node_count", 0) > 0:
            score += 0.10; reasons.append(f"Nodes={scope['node_count']}")

    # Migration alignment
    if needs.get("needs_migration") and stype == "migration":
        src = (scope.get("source_platform") or "").lower()
        tgt = (scope.get("target_platform") or "").lower()
        align = 0.0
        if src and src in name: align += 0.35
        if tgt and tgt in name: align += 0.35
        if is_nc2 and tgt in {"azure","aws"}: align += 0.20
        if "move" in name or "migration" in name: align += 0.10
        score += align
        reasons.append(f"Migration {src}->{tgt}".strip("-"))
        if scope.get("vm_count", 0) > 0:
            score += 0.10; reasons.append(f"VMs={scope['vm_count']}")

    # Database phase detection
    if (needs.get("needs_db_migration") or scope.get("has_database")) and is_ndb:
        score += 0.25; reasons.append("Phase: database")

    # DR emphasis
    if (needs.get("needs_dr") or scope.get("wants_dr") or scope.get("sites",0)>=2):
        if any(k in name for k in [" dr","disaster","recovery","replication","metro","nearsync","leap","sync"]):
            score += 0.30; reasons.append("DR required")

    # ---------------- NOISE & PENALTIES ----------------
    # Penalize irrelevant vendor
    if "powerflex" in name and "powerflex" not in qt:
        score -= 0.60; reasons.append("Irrelevant vendor (PowerFlex)")

    # Penalize EUC or OTHER families unless explicitly requested
    if family in {"OTHER"} and not any(k in qt for k in ["euc","end user","naigpt","ai "]):
        score -= 0.35

    # ---------------- TOKEN OVERLAP BOOST ----------------
    qtok = set(_tokenize(query_text))
    stok = set(_tokenize(f"{name} {category} {family}"))
    overlap = len(qtok & stok)
    if overlap > 0:
        score += min(0.15, overlap * 0.03)
        reasons.append(f"{overlap} keyword matches")

    return score, reasons


# ------------------ journey helpers ------------------

def _is_dr_service(r: Dict[str, Any]) -> bool:
    name = (r.get("service_name") or "").lower()
    cat  = (r.get("category_name") or "").lower()
    keys = [" dr", "disaster", "recovery", "protection", "metro", "nearsync", "replication", "leap", "sync"]
    return any(k in name for k in keys) or any(k in cat for k in ["dr","disaster","recovery","protection","metro","replication"])

def _infer_phase(service: Dict[str, Any]) -> str:
    name  = (service.get("service_name") or "").lower()
    cat   = (service.get("category_name") or "").lower()
    stype = (service.get("service_type") or "").lower()

    if "assessment" in stype or "assessment" in cat or any(k in name for k in ["fitcheck", "readiness"]):
        return "assessment"
    if "migration" in name or "move" in name or "transition" in name or "migration" in cat:
        return "migration"
    if any(k in name for k in [" dr", "disaster", "recovery", "protection", "metro", "nearsync", "replication", "leap"]) \
       or any(k in cat for k in ["dr","disaster","recovery","protection","metro","replication"]):
        return "dr"
    if any(k in name for k in ["database", " ndb", " db "]) or "database" in cat:
        return "database"
    if "deployment" in stype or "deployment" in name or "infrastructure" in name or "implementation" in name:
        return "deployment"
    return "deployment"

def _build_journey(items_serialized: list, daily_rate_field: str = "price_man_day") -> Dict[str, Any]:
    phases_order = ["assessment", "deployment", "migration", "database", "dr"]
    bucket = {p: [] for p in phases_order}

    seen_ids = set()
    for it in items_serialized:
        sid = it.get("id")
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        ph = _infer_phase(it)
        bucket[ph].append(it)

    out_phases = []
    for ph in phases_order:
        services = bucket[ph]
        days = sum(int((s.get("estimate") or {}).get("chosen_days") or s.get("duration_days") or 0) for s in services)
        cost = 0.0
        for s in services:
            est = s.get("estimate") or {}
            rate = float(s.get(daily_rate_field) or 0.0)
            chosen = int(est.get("chosen_days") or s.get("duration_days") or 0)
            cost += rate * chosen
        out_phases.append({
            "phase": ph,
            "services": services,
            "phase_days": int(days),
            "phase_cost_usd": float(cost),
        })

    totals_days = sum(p["phase_days"] for p in out_phases)
    totals_cost = float(sum(p["phase_cost_usd"] for p in out_phases))
    return {"phases": out_phases, "totals": {"days": int(totals_days), "cost_usd": float(totals_cost)}}

def _phase_backfill(phase: str) -> List[Dict[str, Any]]:
    fam_map = {
        "assessment": ["NCI", "NC2", "NDB"],
        "deployment": ["NC2", "NCI"],
        "migration":  ["NC2", "NDB", "NCI"],
        "database":   ["NDB", "NCI"],
        "dr":         ["NCI", "NC2"],
    }
    qmap = {
        "assessment": ["fitcheck", "assessment", "readiness"],
        "deployment": ["deployment", "infrastructure", "implementation"],
        "migration":  ["migration", "move", "conversion", "transition", "aws", "azure"],
        "database":   ["ndb", "database", "db", "sql", "postgres"],
        "dr":         ["dr", "disaster", "recovery", "protection", "metro", "replication", "leap"],
    }

    out = []
    for fam in fam_map.get(phase, []):
        for q in qmap.get(phase, []):
            try:
                out += suggest_services_repo(product_family=fam, platforms=None, limit=30, q=q)
            except Exception:
                continue
    return out

def _ensure_phase_coverage(
    ranked: List[Dict[str, Any]],
    base_items: List[Dict[str, Any]],
    needs: Dict[str, Any],
    required_phases_list: List[str],
    top_k: int
) -> List[Dict[str, Any]]:
    """Ensure exactly one top-ranked service per required phase."""
    required = required_phases_list or ["assessment", "deployment", "migration", "database", "dr"]

    phase_index = {p: [] for p in required}
    for r in ranked:
        ph = _infer_phase(r)
        if ph in phase_index:
            phase_index[ph].append(r)

    # Sort each bucket before selection
    for ph in phase_index:
        phase_index[ph].sort(key=lambda x: x["_scores"]["final"], reverse=True)

    out, seen = [], set()
    for ph in required:
        for cand in phase_index.get(ph, []):
            key = (
                (cand.get("service_name") or "").strip().lower(),
                (cand.get("product_family") or "").strip().lower(),
            )
            if key in seen:
                continue
            out.append(cand)
            seen.add(key)
            break

    if not out and ranked:
        out.append(ranked[0])

    for it in out:
        sc = (it.get("_scores") or {}).get("final")
        if not sc or sc == 0.0:
            prio = float(it.get("priority_score") or 0.0)
            pop = float(it.get("popularity_score") or 0.0)
            kw_bonus = 0.5 if any(k in (it.get("service_name","").lower())
                                   for k in ["fitcheck","migration","dr","database","deployment"]) else 0.0
            it["_scores"] = {
                "final": prio + pop + kw_bonus,
                "keyword": kw_bonus,
                "priority": prio,
                "popularity": pop,
            }
    return out



# ------------------ duration helpers ------------------

def _estimate_and_pick_days_for_service(svc, scope, industry, deployment_type) -> Dict[str, Any]:
    db_days = int(svc.get("duration_days") or 1)
    svc_name = svc.get("service_name", "").strip()

    task_text = svc_name
    if scope.get("target_platform"): task_text += f" on {scope['target_platform']}"
    if scope.get("source_platform"): task_text += f" from {scope['source_platform']}"

    hints = []
    ln = svc_name.lower()
    if "move" in ln: hints.append("Nutanix Move VM migration duration")
    if "fitcheck" in ln: hints.append("Nutanix FitCheck assessment duration")
    if "nc2" in ln: hints.append("NC2 on Azure/AWS deployment duration")
    if "database" in ln or "ndb" in ln: hints.append("Nutanix Database Service migration duration")
    if not hints: hints.append(f"{svc_name} Nutanix professional services duration")

    try:
        ai_days = estimate_days_from_web(
            task_text=task_text,
            industry=industry,
            deployment_type=deployment_type,
            source_platform=scope.get("source_platform"),
            target_platform=scope.get("target_platform"),
            vm_count=scope.get("vm_count"),
            node_count=scope.get("node_count"),
            search_hints=hints,
        )
    except Exception:
        ai_days = None

    if ai_days is not None:
        ai_days = int(max(db_days, min(ai_days, max(1, int(db_days * 2.5)))))

    chosen = pick_days_with_rule(db_days=db_days, ai_days=ai_days)
    provider = "ai" if (ai_days is not None and ai_days >= db_days) else "db"
    return {"db_days": db_days, "ai_days": ai_days, "chosen_days": chosen, "provider": provider}

# ------------------ main planner ------------------


def plan_suggestions(req: "SuggestPlanRequest"):
    """
    Returns: (items_serialized: List[Dict], debug: Dict, journey: Dict)
    Intelligent planner — selects one best-fit service per detected phase
    (assessment, deployment, migration, database, dr).
    """

    # ---------- Intent text ----------
    intent_text = " ".join([
        getattr(req, "requirements_text", "") or "",
        " ".join([getattr(b, "description", "") or "" for b in (getattr(req, "boq", None) or [])]),
        " ".join(list(getattr(req, "providers", None) or [])),
        " ".join(list(((getattr(req, "constraints", None) and getattr(req.constraints, "must_include", None)) or []) or [])),
    ]).strip()

    # ---------- Product family hint ----------
    try:
        pf_list = (getattr(req, "constraints", None) and getattr(req.constraints, "product_families", None)) or []
        pf_from_constraints = pf_list[0] if pf_list else None
    except Exception:
        pf_from_constraints = None
    product_family = getattr(req, "product_family", None) or pf_from_constraints

    # ---------- Keyword expansion ----------
    tokens = expand_keywords(intent_text, product_family)
    if getattr(req, "must_include", None):
        tokens["mandatory"] = sorted(list({*tokens.get("mandatory", []), *list(getattr(req, "must_include"))}))
    if getattr(req, "must_exclude", None):
        tokens["negative"] = sorted(list({*tokens.get("negative", []), *list(getattr(req, "must_exclude"))}))

    columns = ["service_name", "positioning", "product_family", "canonical_names.ov"]
    search_keys = (
        tokens.get("mandatory", []) +
        tokens.get("desirable", []) +
        tokens.get("synonyms", []) +
        tokens.get("tags", [])
    )

    # ---------- Optional repo OR filter ----------
    or_filter = None
    if repo is not None:
        try:
            or_filter = repo.build_or_ilike(search_keys, columns)
        except Exception:
            or_filter = None

    # ---------- Scope and AI scope ----------
    scope_details = _extract_scope_details(req)

    try:
        pf_hint = product_family or ""
        key_scope = f"{getattr(req,'requirements_text','')}|{pf_hint}|{getattr(req,'deployment_type','')}"
    except Exception:
        key_scope = f"{getattr(req,'requirements_text','')}|||{getattr(req,'deployment_type','')}"

    if key_scope in _ai_scope_cache:
        ai_scope = _ai_scope_cache[key_scope]
    else:
        ai_scope = _intelligent_scope_analysis(req)
        _ai_scope_cache[key_scope] = ai_scope

    needs = _detect_needs(req, scope_details, ai_scope)

    # ---------- Required phases detection ----------
    required_phases = set(ai_scope.get("_required_phases_list", []) or [])
    if scope_details.get("sites", 0) >= 2:
        required_phases.update({"assessment", "deployment", "dr"})
    if scope_details.get("has_database"):
        required_phases.add("database")
    if scope_details.get("wants_dr"):
        required_phases.add("dr")
    if scope_details.get("wants_k8s"):
        required_phases.add("deployment")
    if scope_details.get("wants_storage"):
        required_phases.update({"assessment", "migration"})
    if scope_details.get("vm_count", 0) > 0 or scope_details.get("source_platform"):
        required_phases.add("migration")
    if not required_phases:
        required_phases.update({"assessment", "deployment"})  # safe default

    prefer_migration_ready = bool(
        getattr(req, "constraints", None)
        and getattr(req.constraints, "prefer_migration_ready", False)
    )

    # ---------- Candidate fetch (phase- and platform-aware) ----------
    candidates: List[Dict[str, Any]] = []
    try:
        candidates = _dynamic_service_fetch(req, scope_details, ai_scope)
    except Exception:
        candidates = []

    # Fallback if dynamic fetch empty: broad fetch by buckets
    if not candidates:
        platforms = getattr(req, "target_platforms", None) or []
        buckets = [p for p in (platforms or []) if p in ("azure", "aws", "ahv")]
        if "null" not in buckets:
            buckets += ["null", "other"]
        for b in buckets or ["azure", "aws", "ahv", "null", "other"]:
            try:
                if repo is not None:
                    candidates += repo.fetch_candidates_smart(
                        SUPABASE_URL, SUPABASE_KEY, SERVICE_TABLE,
                        product_family, b, or_filter, limit=100
                    )
                else:
                    platforms_arg = [b] if b not in ("null", "other") else None
                    rows = suggest_services_repo(product_family=product_family, platforms=platforms_arg, limit=100, q=None)
                    candidates += rows
            except Exception:
                continue

    # ---------- Hard filters ----------
    neg = {w.lower() for w in tokens.get("negative", [])}
    mand = {w.lower() for w in tokens.get("mandatory", [])}
    scope_target = (scope_details.get("target_platform") or "").lower()

    filtered = []
    for r in candidates:
        blob = " ".join([
            str(r.get("service_name") or ""),
            str(r.get("positioning") or ""),
            str(r.get("category_name") or ""),
            " ".join(r.get("canonical_names") or []),
            str(r.get("product_family") or "")
        ]).lower()

        if any(n in blob for n in neg):
            continue
        if mand and not all(m in blob for m in mand):
            continue

        if scope_target:
            r_targets = [str(t).lower() for t in (r.get("target_platforms") or [])]
            if r_targets and scope_target not in r_targets:
                fam = str(r.get("product_family") or "").upper()
                if fam in {"NC2", "NCI", "NDB", "NUS", "NKP"}:
                    continue

        # constraint gate
        try:
            if not _honor_constraints(r, req, scope_details, ai_scope):
                continue
        except Exception:
            pass

        filtered.append(r)

    pool = filtered or candidates

    # ---------- Product family relevance gate ----------
    if ai_scope.get("product_families"):
        allowed_fams = {pf.upper() for pf in ai_scope["product_families"]}
        gated_pool = []
        for r in pool:
            fam = str(r.get("product_family") or "").upper()
            if not fam or fam in allowed_fams:
                gated_pool.append(r)
        pool = gated_pool or pool

    # ---------- Embedding for query ----------
    qtext = f"{product_family or ''} | {intent_text} | {' '.join(search_keys)}"
    qvec = _embed_cache.get(qtext)
    if qvec is None:
        qvec = _embed(qtext)
        if qvec is not None:
            _embed_cache[qtext] = qvec

    # ---------- Score and rank ----------
    ranked = []
    for r in pool:
        ks = _kw_score(r, tokens)
        pr = float(r.get("priority_score") or 0.0)
        pop = float(r.get("popularity_score") or 0.0)
        vs = 0.0

        if qvec is not None:
            rvec = _parse_emb(r)
            if rvec is None:
                txt = " ".join([
                    str(r.get("service_name") or ""),
                    str(r.get("positioning") or ""),
                    str(r.get("category_name") or ""),
                    " ".join(r.get("canonical_names") or []),
                    str(r.get("product_family") or ""),
                    " ".join([t or "" for t in (r.get("target_platforms") or [])]),
                    str(r.get("service_type") or ""),
                ])
                rvec = _embed(txt)
            if rvec is not None:
                vs = _cos(qvec, rvec)

        rel_score, rel_reasons = _calc_relevance(
            svc=r,
            needs=needs,
            scope=scope_details,
            query_text=intent_text,
            prefer_migration_ready=prefer_migration_ready,
        )

        # Phase keyword bias
        phase_bias = 0.0
        ql = intent_text.lower()
        if any(k in ql for k in ["fitcheck", "assessment", "readiness"]):
            phase_bias += 0.2
        if any(k in ql for k in ["deploy", "deployment", "infrastructure", "cluster", "expand"]):
            phase_bias += 0.2
        if any(k in ql for k in ["migrate", "migration", "move", "rehost"]):
            phase_bias += 0.25
        if any(k in ql for k in ["dr", "disaster", "recovery", "protection", "replication", "leap"]):
            phase_bias += 0.25
        if any(k in ql for k in ["database", "ndb", "sql", "postgres", "oracle"]):
            phase_bias += 0.25

        final = 0.55 * ks + 0.25 * vs + 0.15 * pr + 0.05 * pop + rel_score + phase_bias

        rr = dict(r)
        rr["_scores"] = {
            "final": float(final),
            "keyword": float(ks),
            "vector": float(vs),
            "priority": float(pr),
            "popularity": float(pop),
            "relevance": float(rel_score),
        }
        rr["_reasons"] = rel_reasons
        ranked.append(rr)

    ranked.sort(key=lambda x: x["_scores"]["final"], reverse=True)

    # ---------- Ensure 1 per phase ----------
    items = _ensure_phase_coverage(
        ranked=ranked,
        base_items=[],
        needs=needs,
        required_phases_list=sorted(list(required_phases)),
        top_k=len(required_phases),
    )

    # ---------- Duration + Cost Estimation ----------
    scope_for_est = {
        "vm_count": scope_details.get("vm_count", 0),
        "db_count": scope_details.get("db_count", 0),
        "node_count": scope_details.get("node_count", 0),
        "source_platform": scope_details.get("source_platform"),
        "target_platform": scope_details.get("target_platform"),
    }

    for r in items:
        try:
            est = _estimate_and_pick_days_for_service(
                svc=r,
                scope=scope_for_est,
                industry=getattr(req, "industry", None),
                deployment_type=getattr(req, "deployment_type", None),
            )
        except Exception as e:
            log.debug("Estimate failed for %s: %s", r.get("service_name"), e)
            est = {
                "db_days": int(r.get("duration_days") or 1),
                "ai_days": None,
                "chosen_days": int(r.get("duration_days") or 1),
                "provider": "db",
            }

        r["estimate"] = est
        try:
            p = float(r.get("price_man_day") or 0.0)
        except Exception:
            try:
                p = float(str(r.get("price_man_day")).replace(",", "").strip())
            except Exception:
                p = 0.0
        r["price_man_day"] = p
        r["duration_days"] = int(r.get("duration_days") or r["estimate"]["db_days"] or 1)
        r["cost_estimate"] = round(p * r["estimate"]["chosen_days"], 2)

    # ---------- Serialize ----------
    allowed_fields = [
        "id", "category_name", "service_name", "positioning",
        "duration_days", "price_man_day", "canonical_names",
        "service_type", "supports_db_migration", "target_platforms",
        "priority_score", "popularity_score", "product_family"
    ]

    items_serialized = []
    max_score = max((r.get("_scores", {}).get("final", 0.0) for r in items), default=0.0)

    for r in items:
        serialized = {fld: _to_py(r.get(fld)) for fld in allowed_fields}
        scores = _to_py(r.get("_scores") or {})
        serialized["_scores"] = {k: float(v) if v is not None else 0.0 for k, v in scores.items()}
        serialized["reasons"] = [str(x) for x in _to_py(r.get("_reasons") or [])][:8]
        serialized["reason"] = ", ".join(serialized["reasons"])
        est = _to_py(r.get("estimate") or {})
        serialized["estimate"] = {
            "db_days": int(est.get("db_days") or 0),
            "ai_days": (float(est.get("ai_days")) if est.get("ai_days") is not None else None),
            "chosen_days": int(est.get("chosen_days") or 0),
            "provider": str(est.get("provider") or "db"),
        }
        serialized["cost_estimate"] = float(round(_to_py(r.get("cost_estimate") or 0.0), 2))
        serialized["score"] = float(serialized["_scores"].get("final", 0.0))
        serialized["score_normalized"] = (
            round(serialized["score"] / max_score * 100.0, 2) if max_score > 0 else round(serialized["score"] * 10.0, 2)
        )
        items_serialized.append(serialized)

    # ---------- Journey ----------
    journey = _build_journey(items_serialized)

    # ---------- Debug ----------
    debug = {
        "query_text": intent_text,
        "scope": {"keywords": tokens, "ai_scope": ai_scope, "needs": needs},
        "candidates_fetched": len(candidates),
        "unique_services": len({
            (str(r.get('service_name') or '').lower(), str(r.get('product_family') or '').lower())
            for r in candidates
        }),
        "services_suggested": len(items_serialized),
        "required_phases": sorted(list(required_phases)),
    }

    return items_serialized, debug, journey
