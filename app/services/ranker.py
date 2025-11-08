# app/services/ranker.py — Intelligent planner (phase-covered, strict top_k, journey-fixed)
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Set
import json
import re
import logging
import os

import numpy as np
import httpx

from app.models.schemas import SuggestPlanRequest
from app.services.proposals_repo import suggest_services_repo
from app.services.duration_estimator import estimate_days_from_web, pick_days_with_rule, call_llm

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

# ------------------ keyword expansion ------------------

def expand_keywords(text: Optional[str], product_family: Optional[str] = None) -> Dict[str, List[str]]:
    tokens: Set[str] = set(_simple_tokenize(text or ""))
    if product_family:
        tokens.add(product_family.lower())

    synonym_map = {
        "ndb": ["ndb", "nutanix database service", "nutanix db service", "dbs", "database"],
        "nci": ["nci", "prism", "flow", "ncm", "ahv"],
        "nus": ["nus", "nutanix unified storage", "files"],
        "nkp": ["nkp", "nutanix kubernetes platform", "kubernetes"],
        "nc2": ["nc2", "nutanix cloud clusters", "nutanix cloud cluster"],
    }

    synonyms: Set[str] = set()
    pf = (product_family or "").strip().lower()
    if pf in synonym_map:
        synonyms.update(synonym_map[pf])
    for t in list(tokens):
        if t in synonym_map:
            synonyms.update(synonym_map[t])

    if any(k in tokens for k in ("migrate", "migration", "migrating", "move", "rehost")):
        synonyms.update(["migration", "migrate", "move"])
    if any(k in tokens for k in ("deploy", "deployment", "configure", "setup")):
        synonyms.update(["deployment", "deploy", "configure", "setup"])
    if any(k in tokens for k in ("dr", "disaster", "recovery", "protection", "replication", "metro", "near")):
        synonyms.update(["dr", "disaster", "recovery", "protection", "replication"])

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
    prompt = f"""
You are a Nutanix Professional Services planner. Read the project details and return ONLY compact JSON.

Return keys:
- source_platform: one of ["vmware","hyperv","aws","azure","gcp","cisco_hyperflex", null]
- target_platform: one of ["ahv","aws","azure","gcp", null]
- product_families: list of families to use (subset of ["NCI","NC2","NDB","NKP","NUS"])
- required_phases: ordered list from ["assessment","deployment","migration","database","dr"]
- estimated_vm_count: integer or 0
- estimated_db_count: integer or 0
- hard_constraints: {{"must_include":[], "must_exclude":[]}}

Project:
- Industry: {req.industry}
- Deployment type: {req.deployment_type}
- Proposal type: {req.proposal_type}
- Requirements: {req.requirements_text}
- Providers: {list(req.providers or [])}
- Selected vendor: {req.selected_vendor}
- BOQ: {json.dumps([b.model_dump() for b in (req.boq or [])], ensure_ascii=False)}

Respond with JSON only.
"""
    ai = call_llm(prompt, max_tokens=350, temperature=0.0) or ""
    data = _safe_json_loads(ai) or {}

    sp = (data.get("source_platform") or "")
    tp = (data.get("target_platform") or "")
    fams = data.get("product_families") or []
    phases = data.get("required_phases") or []
    hard = data.get("hard_constraints") or {}

    out = {
        "source_platform": sp.lower() if isinstance(sp, str) and sp else None,
        "target_platform": tp.lower() if isinstance(tp, str) and tp else None,
        "product_families": [f for f in fams if f in {"NCI","NC2","NDB","NKP","NUS"}],
        "required_phases": [p for p in phases if p in {"assessment","deployment","migration","database","dr"}],
        "estimated_vm_count": int(data.get("estimated_vm_count") or 0),
        "estimated_db_count": int(data.get("estimated_db_count") or 0),
        "hard_constraints": {
            "must_include": list(hard.get("must_include") or []),
            "must_exclude": list(hard.get("must_exclude") or []),
        },
    }

    phases_list = out.get("required_phases", []) or []
    out["_required_phases_list"] = list(phases_list)
    out["required_phases"] = {"phases": list(phases_list)}
    return out

# ------------------ rule scope extraction ------------------

def _extract_scope_details(req: SuggestPlanRequest) -> Dict[str, Any]:
    qtext = f"{req.requirements_text or ''} {' '.join([b.description or '' for b in (req.boq or [])])}".lower()

    vm_count = 0
    for pattern in [r'(\d+)\s*vms?\b', r'(\d+)\s*virtual\s*machines?\b']:
        m = re.search(pattern, qtext)
        if m: vm_count = max(vm_count, int(m.group(1)))
    db_count = 0
    for pattern in [r'(\d+)\s*databases?\b', r'(\d+)\s*dbs?\b']:
        m = re.search(pattern, qtext)
        if m: db_count = max(db_count, int(m.group(1)))

    node_count = 0
    for pattern in [r'(\d+)\s*nodes?\b', r'(\d+)\s*servers?\b', r'(\d+)\s*dell\s*nodes?\b']:
        m = re.search(pattern, qtext)
        if m: node_count = max(node_count, int(m.group(1)))

    source_platform = None
    if any(kw in qtext for kw in ['vmware','vsphere','vcenter','esxi']): source_platform = 'vmware'
    elif any(kw in qtext for kw in ['hyperflex','cisco ucs','ucs']): source_platform = 'cisco_hyperflex'
    elif 'hyper-v' in qtext or 'hyperv' in qtext: source_platform = 'hyperv'
    elif 'aws' in qtext: source_platform = source_platform or 'aws'

    target_platform = None
    if 'ahv' in qtext or 'acropolis' in qtext: target_platform = 'ahv'
    elif 'azure' in qtext: target_platform = 'azure'
    elif 'aws' in qtext: target_platform = target_platform or 'aws'

    has_database = any(kw in qtext for kw in ['database','oracle','sql server','postgres','mysql','mongodb','rds'])
    needs_cluster_config = node_count > 0 or any(kw in qtext for kw in ['configure','configuration','setup','deploy cluster'])

    return {
        "vm_count": vm_count,
        "db_count": db_count,
        "node_count": node_count,
        "has_database": has_database or (db_count > 0),
        "source_platform": source_platform,
        "target_platform": target_platform,
        "needs_cluster_config": needs_cluster_config,
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
            scope["node_count"] > 0 or scope["needs_cluster_config"] or any(t in tokens for t in ['deploy','setup','configure','infrastructure','expansion'])
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

# ------------------ fetch ------------------

def _dynamic_service_fetch(req: SuggestPlanRequest, scope: Dict[str, Any], ai_scope: Dict[str, Any]) -> List[Dict[str, Any]]:
    fam_from_req = list(getattr(req, "constraints", None) and (req.constraints.product_families or []) or [])
    fam_from_ai = list(ai_scope.get("product_families") or [])
    families = fam_from_req or fam_from_ai or ["NCI","NC2","NDB","NKP","NUS"]

    if scope.get("has_database") and "NDB" not in families:
        families = families + ["NDB"]
    if "NC2" in families and "NCI" not in families:
        families = families + ["NCI"]

    platforms = list(getattr(req, "constraints", None) and (req.constraints.target_platforms or []) or []) or ([ai_scope.get("target_platform")] if ai_scope.get("target_platform") else None)

    queries = [None, "fitcheck", "infrastructure", "migration", "database", "dr", "move"]

    seen: set[int] = set()
    out: List[Dict[str, Any]] = []
    for fam in families:
        for q in queries:
            try:
                rows = suggest_services_repo(product_family=fam, platforms=platforms, limit=50, service_type=None, q=q)
            except Exception:
                rows = []
            for r in rows:
                rid = r.get("id")
                if isinstance(rid, int) and rid not in seen:
                    seen.add(rid)
                    out.append(r)
    return out

# ------------------ relevance and scoring ------------------

def _calc_relevance(
    svc: Dict[str, Any],
    needs: Dict[str, Any],
    scope: Dict[str, Any],
    query_text: str,
    prefer_migration_ready: bool
) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    name = (svc.get("service_name") or "").lower()
    category = (svc.get("category_name") or "").lower()
    stype = (svc.get("service_type") or "").lower()
    family = (svc.get("product_family") or "").upper()
    targets = [t.lower() for t in (svc.get("target_platforms") or [])]

    if scope.get("target_platform"):
        if scope["target_platform"] in targets:
            score += 0.25; reasons.append(f"Targets {scope['target_platform']}")
        elif not targets:
            score += 0.05
        else:
            score -= 0.05

    if needs.get("needs_assessment") and (stype == "assessment" or "assessment" in category or "fitcheck" in name):
        score += 0.10; reasons.append("Phase alignment: assessment")
    if needs.get("needs_deployment") and (stype == "deployment" or "infrastructure" in name or "deployment" in name or "cluster" in name):
        score += 0.15; reasons.append("Phase alignment: deployment")
    if needs.get("needs_migration") and (stype == "migration") and not any(k in category for k in ["database"]) and "db" not in name and family != "NDB":
        score += 0.15; reasons.append("Phase alignment: migration")
    if needs.get("needs_dr") and any(k in name for k in [" dr", "disaster", "recovery", "protection", "near", "metro", "replication", "leap"]):
        score += 0.10; reasons.append("Phase alignment: DR")

    if ("fitcheck" in name) or ("assessment" in category) or (stype == "assessment"):
        if needs.get("needs_assessment"):
            score += 0.25; reasons.append("Assessment needed for sizing")
        else:
            score -= 0.10

    if stype == "deployment" or "infrastructure" in name or "deployment" in name or "cluster" in name:
        if needs.get("needs_deployment"):
            score += 0.30; reasons.append("Infrastructure deployment required")
            if scope.get("node_count", 0) > 0:
                score += 0.10; reasons.append(f"Configuring {scope['node_count']} nodes")
            if scope.get("needs_cluster_config"):
                score += 0.10; reasons.append("Cluster configuration needed")
        else:
            score -= 0.20

    if stype == "migration" and ("database" not in category and "db" not in name and family != "NDB"):
        if needs.get("needs_migration"):
            score += 0.35; reasons.append("VM migration required")
            if scope.get("vm_count", 0) > 0:
                score += 0.10; reasons.append(f"Migrating {scope['vm_count']} VMs")
            if "move" in name and scope.get("source_platform") in ["vmware","cisco_hyperflex","aws"]:
                score += 0.10; reasons.append(f"Nutanix Move for {scope.get('source_platform')}")
        else:
            score -= 0.25

    is_db_service = ("database" in category) or (family == "NDB") or (" db" in f" {name}") or ("ndb" in name)
    if is_db_service:
        if needs.get("needs_db_migration") or scope.get("has_database"):
            score += 0.25; reasons.append("Database scope present")
            if prefer_migration_ready and stype in ("migration","design"):
                score += 0.10; reasons.append("Migration-ready preference")
            if needs.get("needs_db_migration"):
                score += 0.10; reasons.append("Phase alignment: database")
        else:
            score -= 0.25

    dr_hit_name = any(k in name for k in [" dr", "disaster", "recovery", "protection", "metro", "nearsync", "replication", "leap"])
    dr_hit_cat  = any(k in category for k in ["dr","disaster","recovery","protection","metro","replication"])
    if dr_hit_name or dr_hit_cat:
        if needs.get("needs_dr"):
            score += 0.20; reasons.append("DR required")
        else:
            score -= 0.05

    qtok = set(_tokenize(query_text))
    stok = set(_tokenize(f"{name} {category} {family}"))
    ov = len(qtok & stok)
    if ov > 0:
        score += min(0.15, ov * 0.03)
        reasons.append(f"{ov} keyword matches")

    return score, reasons

# ------------------ AI shortlist (optional) ------------------

def _ai_rank_services(requirements: str, shortlist_rows: List[Dict[str, Any]], limit: int, scope: Dict[str, Any]) -> List[int]:
    brief = [
        {
            "id": r["id"],
            "name": r.get("service_name"),
            "family": r.get("product_family"),
            "type": r.get("service_type"),
            "category": r.get("category_name"),
            "targets": r.get("target_platforms"),
        }
        for r in shortlist_rows[:40]
    ]
    needs_flags = {k:v for k,v in scope.items() if k.startswith('needs_')}
    prompt = f"""
Select the best {limit} Nutanix Professional Services for this project. Return ONLY a JSON array of IDs.

Project scope:
- source_platform: {scope.get('source_platform')}
- target_platform: {scope.get('target_platform')}
- vm_count: {scope.get('vm_count')}
- db_count: {scope.get('db_count')}
- needs: {json.dumps(needs_flags, ensure_ascii=False)}
- requirements_text: {requirements[:1500]}

Shortlist:
{json.dumps(brief, ensure_ascii=False)}

Rules:
- Prefer services whose product_family and target_platform match the scope.
- Include assessment if sizing or discovery is implied.
- Include deployment for NC2/NCI cluster setup if moving to cloud/AHV.
- Include migration if VM move is needed; include database services if DB migration mentioned.
- Do not invent IDs. Only pick from shortlist.
Respond with JSON array like: [58, 70, 191]
"""
    out = call_llm(prompt, max_tokens=250, temperature=0.0) or "[]"
    ids = _safe_json_loads(out)
    return [int(i) for i in ids if isinstance(i, int)]

# ------------------ duration helpers ------------------

def _estimate_and_pick_days_for_service(
    svc: Dict[str, Any],
    scope: Dict[str, Any],
    industry: Optional[str],
    deployment_type: Optional[str]
) -> Dict[str, Any]:
    db_days = int(svc.get("duration_days") or 1)
    svc_name = svc.get("service_name", "").strip()

    task_text = svc_name
    if scope.get("target_platform"):
        task_text += f" on {scope['target_platform']}"
    if scope.get("source_platform"):
        task_text += f" from {scope['source_platform']}"

    hints: List[str] = []
    ln = svc_name.lower()
    if "move" in ln: hints.append("Nutanix Move VM migration duration")
    if "fitcheck" in ln: hints.append("Nutanix FitCheck assessment duration")
    if "infrastructure" in ln and "deployment" in ln: hints.append("Nutanix cluster deployment duration")
    if "infrastructure" in ln and "expansion" in ln: hints.append("Nutanix cluster expansion duration")
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
    except Exception as e:
        log.error(f"AI estimation failed for {svc_name}: {e}")
        ai_days = None

    chosen = pick_days_with_rule(db_days=db_days, ai_days=ai_days)
    provider = "ai" if (ai_days is not None and ai_days >= db_days) else "db"
    return {"db_days": db_days, "ai_days": ai_days, "chosen_days": chosen, "provider": provider}

# ------------------ embedding helpers ------------------

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _embed(text: str) -> np.ndarray | None:
    if not (AZOAI_ENDPOINT and AZOAI_KEY and AZOAI_DEPLOYMENT_EMBED):
        return None
    url = f"{AZOAI_ENDPOINT}/openai/deployments/{AZOAI_DEPLOYMENT_EMBED}/embeddings?api-version=2024-10-01-preview"
    payload = {"input": text}
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0, read=20.0)) as cx:
            r = cx.post(url, headers={"api-key": AZOAI_KEY, "Content-Type":"application/json"}, json=payload)
            r.raise_for_status()
            return np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
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

# ------------------ journey helpers ------------------

def _is_dr_service(r: Dict[str, Any]) -> bool:
    name = (r.get("service_name") or "").lower()
    cat  = (r.get("category_name") or "").lower()
    keys = [" dr", "disaster", "recovery", "protection", "metro", "nearsync", "replication", "leap"]
    return any(k in name for k in keys) or any(k in cat for k in ["dr","disaster","recovery","protection","metro","replication"])

def _infer_phase(r: Dict[str, Any]) -> str:
    stype = (r.get("service_type") or "").lower()
    name  = (r.get("service_name") or "").lower()
    cat   = (r.get("category_name") or "").lower()
    fam   = (r.get("product_family") or "").upper()
    if _is_dr_service(r):
        return "dr"
    if fam == "NDB" or "database" in cat or " ndb" in f" {name}" or " db" in f" {name}":
        return "database"
    if stype == "migration" or "migration" in name or "migrate" in name or "move" in name:
        return "migration"
    if stype == "assessment" or "assessment" in cat or "fitcheck" in name:
        return "assessment"
    return "deployment"

def _build_journey(items_serialized: list, daily_rate_field: str = "price_man_day") -> Dict[str, Any]:
    phases_order = ["assessment", "deployment", "migration", "database", "dr"]
    bucket = {p: [] for p in phases_order}

    seen = set()
    for it in items_serialized:
        key = ((it.get("service_name") or "").strip().lower(), (it.get("product_family") or "").strip().upper())
        if key in seen:
            continue
        seen.add(key)
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

# ------------------ selection helpers (phase coverage) ------------------

def _ensure_phase_coverage(
    ranked: List[Dict[str, Any]],
    base_items: List[Dict[str, Any]],
    needs: Dict[str, Any],
    required_phases_list: List[str],
    top_k: int
) -> List[Dict[str, Any]]:
    """Pad selection so at least one item exists for each required phase when possible."""
    phase_needed = set(required_phases_list or [])
    # derive phases already covered
    covered = { _infer_phase(it) for it in base_items }
    missing = [p for p in required_phases_list if p not in covered]

    if not missing:
        return base_items[:top_k]

    # build phase-to-candidates index in ranked order
    phase_index: Dict[str, List[Dict[str, Any]]] = {"assessment":[], "deployment":[], "migration":[], "database":[], "dr":[]}
    for r in ranked:
        phase_index[_infer_phase(r)].append(r)

    seen_keys = {( (it.get("service_name") or "").strip().lower(), (it.get("product_family") or "").strip().lower() ) for it in base_items}
    out = list(base_items)

    for ph in missing:
        for cand in phase_index.get(ph, []):
            key = (str(cand.get("service_name") or "").strip().lower(), str(cand.get("product_family") or "").strip().lower())
            if key in seen_keys:
                continue
            out.append(cand)
            seen_keys.add(key)
            break

    # final trim to top_k but prefer keeping one per phase first
    if len(out) > top_k:
        # keep at least one per required phase, then highest score for the rest
        must_keep = set()
        for ph in required_phases_list:
            for i, it in enumerate(out):
                if _infer_phase(it) == ph:
                    must_keep.add(i)
                    break
        keep_items = [it for i, it in enumerate(out) if i in must_keep]
        extra = [it for i, it in enumerate(out) if i not in must_keep]
        extra.sort(key=lambda x: float(((x.get("_scores") or {}).get("final")) or 0.0), reverse=True)
        out = (keep_items + extra)[:top_k]

    return out

# ------------------ main planner ------------------

def plan_suggestions(req: "SuggestPlanRequest"):
    """
    Returns: (items_serialized: List[Dict], debug: Dict, journey: Dict)
    """
    intent_text = getattr(req, "intent_text", None) or getattr(req, "requirements_text", None) or ""
    product_family = getattr(req, "product_family", None)
    if not product_family:
        pf_list = getattr(req, "constraints", None) and getattr(req.constraints, "product_families", None) or []
        product_family = pf_list[0] if pf_list else None

    tokens = expand_keywords(intent_text, product_family)

    if getattr(req, "must_include", None):
        tokens["mandatory"] = sorted(list({*tokens.get("mandatory", []), *list(req.must_include)}))
    if getattr(req, "must_exclude", None):
        tokens["negative"] = sorted(list({*tokens.get("negative", []), *list(req.must_exclude)}))

    columns = ["service_name","positioning","product_family","canonical_names.ov"]
    search_keys = (tokens.get("mandatory", []) +
                   tokens.get("desirable", []) +
                   tokens.get("synonyms", []) +
                   tokens.get("tags", []))

    or_filter = None
    if repo is not None:
        try:
            or_filter = repo.build_or_ilike(search_keys, columns)
        except Exception:
            or_filter = None

    platforms = getattr(req, "target_platforms", None) or []
    buckets = [p for p in (platforms or []) if p in ("azure","aws","ahv")]
    if "null" not in buckets:
        buckets += ["null","other"]

    candidates = []
    for b in buckets or ["azure","aws","ahv","null","other"]:
        try:
            if repo is not None:
                candidates += repo.fetch_candidates_smart(
                    SUPABASE_URL, SUPABASE_KEY, SERVICE_TABLE,
                    getattr(req, "product_family", None), b, or_filter, limit=100
                )
            else:
                pf = getattr(req, "product_family", None)
                platforms_arg = [b] if b not in ("null","other") else None
                rows = suggest_services_repo(product_family=pf, platforms=platforms_arg, limit=100, q=None)
                candidates += rows
        except Exception:
            continue

    qvec = _embed(f"{getattr(req, 'product_family', '')} | {intent_text} | {' '.join(search_keys)}")

    neg = {w.lower() for w in tokens.get("negative", [])}
    mand = {w.lower() for w in tokens.get("mandatory", [])}
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
        filtered.append(r)
    pool = filtered or candidates

    scope_details = _extract_scope_details(req)
    ai_scope = _intelligent_scope_analysis(req)
    needs = _detect_needs(req, scope_details, ai_scope)
    prefer_migration_ready = bool(getattr(req, "constraints", None) and getattr(req.constraints, "prefer_migration_ready", False))

    ranked = []
    for r in pool:
        ks = _kw_score(r, tokens)
        try:
            pr = float(r.get("priority_score") or 0.0)
        except Exception:
            pr = 0.0
        try:
            pop = float(r.get("popularity_score") or 0.0)
        except Exception:
            pop = 0.0

        vs = 0.0
        if qvec is not None:
            rvec = _parse_emb(r)
            if rvec is not None:
                vs = _cos(qvec, rvec)

        rel_score, rel_reasons = _calc_relevance(
            svc=r, needs=needs, scope=scope_details,
            query_text=intent_text, prefer_migration_ready=prefer_migration_ready
        )

        final = 0.55 * ks + 0.25 * vs + 0.15 * pr + 0.05 * pop + rel_score

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

    req_top_k = int(getattr(req, "top_k", 8) or 8)
    # ensure at least one per required phase if present and if req.top_k is too small
    min_k = max(req_top_k, len(ai_scope.get("_required_phases_list", [])))
    top_k = max(1, min(min_k, 50))

    # first pass unique pick by name+family
    seen, items = set(), []
    for r in ranked:
        key = (str(r.get("service_name") or "").strip().lower(),
               str(r.get("product_family") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        items.append(r)
        if len(items) >= top_k:
            break

    # phase coverage padding
    items = _ensure_phase_coverage(
        ranked=ranked,
        base_items=items,
        needs=needs,
        required_phases_list=ai_scope.get("_required_phases_list", []),
        top_k=top_k,
    )

    # durations and costs
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

        r["estimate"] = {
            "db_days": int(est.get("db_days") or 0),
            "ai_days": (float(est.get("ai_days")) if est.get("ai_days") is not None else None),
            "chosen_days": int(est.get("chosen_days") or 0),
            "provider": str(est.get("provider") or "db"),
        }

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

    # serialize
    allowed_fields = [
        "id", "category_name", "service_name", "positioning",
        "duration_days", "price_man_day", "canonical_names",
        "service_type", "supports_db_migration", "target_platforms",
        "priority_score", "popularity_score", "product_family"
    ]

    items_serialized = []
    max_score = max((r.get("_scores", {}).get("final", 0.0) for r in items), default=0.0)

    for r in items:
        serialized: Dict[str, Any] = {}
        for fld in allowed_fields:
            v = r.get(fld)
            if v is None:
                if fld in ("canonical_names", "target_platforms"):
                    serialized[fld] = []
                elif fld in ("supports_db_migration",):
                    serialized[fld] = bool(r.get(fld, False))
                else:
                    serialized[fld] = _to_py(v)
            else:
                serialized[fld] = _to_py(v)

        scores = _to_py(r.get("_scores") or {})
        if scores:
            serialized["_scores"] = {k: (float(v) if v is not None else None) for k, v in scores.items()}
        else:
            serialized["_scores"] = {"final": 0.0, "keyword": 0.0, "vector": 0.0, "priority": 0.0, "popularity": 0.0, "relevance": 0.0}

        reasons = _to_py(r.get("_reasons") or [])
        if reasons:
            serialized["reasons"] = [str(x) for x in reasons][:8]
            serialized["reason"] = ", ".join(serialized["reasons"])
        else:
            kw = serialized["_scores"].get("keyword", 0.0)
            pr = serialized["_scores"].get("priority", 0.0)
            vec = serialized["_scores"].get("vector", 0.0)
            serialized["reasons"] = [f"kw={kw:.2f}", f"prio={pr:.2f}", f"vec={vec:.2f}"]
            serialized["reason"] = ", ".join(serialized["reasons"])

        est = _to_py(r.get("estimate") or {})
        serialized["estimate"] = {
            "db_days": int(est.get("db_days") or 0),
            "ai_days": (float(est.get("ai_days")) if est.get("ai_days") is not None else None),
            "chosen_days": int(est.get("chosen_days") or 0),
            "provider": str(est.get("provider") or "db"),
        }
        serialized["cost_estimate"] = float(round(_to_py(r.get("cost_estimate") or 0.0), 2))

        try:
            serialized["score"] = float(serialized["_scores"].get("final") or 0.0)
        except Exception:
            serialized["score"] = 0.0

        try:
            if max_score and max_score > 0:
                serialized["score_normalized"] = round(float(serialized["score"]) / float(max_score) * 100.0, 2)
            else:
                serialized["score_normalized"] = round(float(serialized["score"]) * 10.0, 2)
        except Exception:
            serialized["score_normalized"] = float(serialized["score"])

        items_serialized.append(serialized)

    journey = _build_journey(items_serialized)

    debug = {
        "query_text": intent_text,
        "scope": {"ai": tokens, "ai_scope": ai_scope},
        "candidates_fetched": len(candidates),
        "unique_services": len({(str(r.get('service_name') or '').lower(), str(r.get('product_family') or '').lower()) for r in candidates}),
        "services_suggested": len(items_serialized),
    }

    return items_serialized, debug, journey
