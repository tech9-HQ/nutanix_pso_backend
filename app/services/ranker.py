from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import re, logging

from app.models.schemas import SuggestPlanRequest, RankedService
from app.services.proposals_repo import suggest_services_repo
from app.services.llm_selector import llm_pick_services
from app.services.duration_estimator import estimate_days_from_web, pick_days_with_rule

log = logging.getLogger("ranker")

# ---------------- token utils ----------------
def _tokenize(s: str) -> List[str]:
    if not s:
        return []
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return s.split()

# ---------------- scope + need detection ----------------
def _extract_scope_details(req: SuggestPlanRequest) -> Dict[str, Any]:
    qtext = f"{req.requirements_text or ''} {' '.join([b.description or '' for b in (req.boq or [])])}".lower()

    vm_count = 0
    for pattern in [r'(\d+)\s*vms?\b', r'(\d+)\s*virtual\s*machines?\b']:
        m = re.search(pattern, qtext)
        if m:
            vm_count = max(vm_count, int(m.group(1)))

    node_count = 0
    for pattern in [r'(\d+)\s*nodes?\b', r'(\d+)\s*servers?\b', r'(\d+)\s*dell\s*nodes?\b']:
        m = re.search(pattern, qtext)
        if m:
            node_count = max(node_count, int(m.group(1)))

    db_keywords = ['database', 'oracle', 'sql server', 'postgres', 'mysql', 'mongodb', 'db migration']
    has_database = any(kw in qtext for kw in db_keywords)

    source_platform = None
    if any(kw in qtext for kw in ['vmware', 'vsphere', 'vcenter', 'esxi']):
        source_platform = 'vmware'
    elif any(kw in qtext for kw in ['hyperflex', 'cisco', 'ucs']):
        source_platform = 'cisco_hyperflex'
    elif any(kw in qtext for kw in ['hyper-v', 'hyperv', 'microsoft']):
        source_platform = 'hyperv'

    target_platform = None
    if any(kw in qtext for kw in ['ahv', 'acropolis']):
        target_platform = 'ahv'
    elif 'aws' in qtext:
        target_platform = 'aws'
    elif 'azure' in qtext:
        target_platform = 'azure'

    needs_cluster_config = node_count > 0 or any(kw in qtext for kw in [
        'configure', 'configuration', 'setup', 'deploy cluster', 'rack and stack'
    ])

    return {
        'vm_count': vm_count,
        'node_count': node_count,
        'has_database': has_database,
        'source_platform': source_platform,
        'target_platform': target_platform,
        'needs_cluster_config': needs_cluster_config,
        'has_boq': bool(req.boq and len(req.boq) >= 2),
    }

def _detect_service_needs(req: SuggestPlanRequest, scope: Dict[str, Any]) -> Dict[str, Any]:
    qtext = f"{req.requirements_text or ''} {' '.join([b.description or '' for b in (req.boq or [])])}".lower()
    tokens = set(_tokenize(qtext))
    return {
        'needs_assessment': (
            not scope['has_boq'] or
            scope['vm_count'] > 0 or
            any(t in tokens for t in ['assess', 'fitcheck', 'sizing', 'evaluation', 'poc'])
        ),
        'needs_deployment': (
            scope['node_count'] > 0 or
            scope['needs_cluster_config'] or
            any(t in tokens for t in ['deploy', 'setup', 'configure', 'infrastructure', 'expansion'])
        ),
        'needs_migration': (
            (scope['vm_count'] > 0 and scope['source_platform'] is not None) or
            any(t in tokens for t in ['migrate', 'migration', 'move', 'consolidate', 'move', 'move out'])
        ),
        'needs_db_migration': scope['has_database'] and any(t in tokens for t in ['migrate', 'migration']),
        'needs_development': any(t in tokens for t in ['custom', 'automation', 'integration', 'api', 'script']),
        'needs_ai_analytics': any(t in tokens for t in ['ai', 'machine', 'analytics', 'ml', 'data science']),
        'needs_optimization': any(t in tokens for t in ['optimize', 'health', 'performance', 'tuning']),
    }

# ---------------- relevance scoring ----------------
def _calculate_relevance_score(
    service: Dict[str, Any],
    needs: Dict[str, Any],
    scope: Dict[str, Any],
    query_text: str
) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    svc_name = (service.get('service_name') or '').lower()
    svc_type = (service.get('service_type') or '').lower()
    category = (service.get('category_name') or '').lower()

    if not scope['has_database']:
        if 'database' in svc_name or 'database' in category:
            score -= 0.25
            reasons.append('No DB scope; deprioritize database services')

    if 'assessment' in category or 'fitcheck' in svc_name:
        if needs['needs_assessment']:
            score += 0.20
            reasons.append('Assessment needed for sizing')
            if scope['vm_count'] > 0:
                score += 0.10
                reasons.append(f"Sizing for {scope['vm_count']} VMs")
        else:
            score -= 0.15

    if svc_type == 'deployment' or any(k in svc_name for k in ['infrastructure deployment', 'infrastructure expansion']):
        if needs['needs_deployment']:
            score += 0.25
            reasons.append('Infrastructure deployment required')
            if scope['node_count'] > 0:
                score += 0.15
                reasons.append(f"Configuring {scope['node_count']} nodes")
            if scope['needs_cluster_config']:
                score += 0.10
                reasons.append('Cluster configuration needed')
        else:
            score -= 0.30

    if svc_type == 'migration' and 'database' not in category:
        if needs['needs_migration']:
            score += 0.30
            reasons.append('VM migration required')
            if scope['vm_count'] > 0:
                score += 0.15
                reasons.append(f"Migrating {scope['vm_count']} VMs")
            if 'move' in svc_name and scope['source_platform'] in ['vmware', 'cisco_hyperflex']:
                score += 0.20
                reasons.append(f"Nutanix Move for {scope['source_platform']}")
            if 'hyperflex' in svc_name and scope['source_platform'] == 'cisco_hyperflex':
                score += 0.25
                reasons.append('HyperFlex to AHV migration specialist')
        else:
            score -= 0.40

    if 'database' in category and 'migration' in svc_name:
        if needs['needs_db_migration']:
            score += 0.30
            reasons.append('Database migration required')
        else:
            score -= 0.35

    if 'development' in category:
        if needs['needs_development']:
            score += 0.20
            reasons.append('Custom development needed')
        else:
            score -= 0.25

    if 'ai' in category or 'analytics' in category:
        if needs['needs_ai_analytics']:
            score += 0.25
            reasons.append('AI/Analytics workload')
        else:
            score -= 0.30

    if any(k in svc_name for k in ['health check', 'optimization', 'performance']):
        if needs['needs_optimization']:
            score += 0.15
            reasons.append('Optimization services')
        else:
            score -= 0.10

    query_tokens = set(_tokenize(query_text))
    service_tokens = set(_tokenize(f"{svc_name} {category}"))
    overlap = len(query_tokens & service_tokens)
    if overlap > 0:
        kw_score = min(0.15, overlap * 0.03)
        score += kw_score
        reasons.append(f"{overlap} keyword matches")

    return score, reasons

# ---------------- minimal set selection ----------------
def _select_core_services(
    all_services: List[Dict[str, Any]],
    needs: Dict[str, Any],
    scope: Dict[str, Any],
    limit: int
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    selected_ids = set()

    def pick_best(predicate, max_picks=1):
        cand = [s for s in all_services if s['id'] not in selected_ids and predicate(s)]
        if not cand:
            return
        cand.sort(
            key=lambda x: (x.get('priority_score', 0) * 0.6 + x.get('popularity_score', 0) * 0.4),
            reverse=True
        )
        for i in range(min(max_picks, len(cand))):
            selected.append(cand[i])
            selected_ids.add(cand[i]['id'])

    # Assessment
    if needs['needs_assessment']:
        pick_best(lambda s: 'fitcheck' in s.get('service_name','').lower()
                           and s.get('product_family') == 'NCI'
                           and 'starter' in s.get('service_name','').lower(), 1)
        if not selected:
            pick_best(lambda s: 'fitcheck' in s.get('service_name','').lower()
                               and s.get('product_family') == 'NCI', 1)
        if not selected:
            pick_best(lambda s: 'fitcheck' in s.get('service_name','').lower(), 1)

    # Deployment
    if needs['needs_deployment']:
        picked = len(selected)
        pick_best(lambda s: s.get('product_family') == 'NCI'
                           and 'infrastructure deployment' in s.get('service_name','').lower(), 1)
        if len(selected) == picked:
            pick_best(lambda s: s.get('product_family') == 'NCI'
                               and 'infrastructure expansion' in s.get('service_name','').lower(), 1)

    # Migration
    if needs['needs_migration']:
        pick_best(lambda s: s.get('service_type') == 'migration'
                           and 'move' in s.get('service_name','').lower(), 1)
        if scope.get('source_platform') == 'cisco_hyperflex':
            pick_best(lambda s: 'hyperflex' in s.get('service_name','').lower(), 1)

    return selected[:max(1, limit)]

# ---------------- days estimator glue ----------------
def _estimate_and_pick_days_for_service(
    svc: Dict[str, Any],
    scope: Dict[str, Any],
    industry: Optional[str],
    deployment_type: Optional[str]
) -> Dict[str, Any]:
    """
    Returns {"db_days": int, "ai_days": Optional[int], "chosen_days": int, "provider": "db"|"ai"}
    and does not mutate svc.
    """
    db_days = int(svc.get("duration_days") or 0)
    task_text = f"{svc.get('service_name','')} on {scope.get('target_platform') or ''}".strip()
    # Strong hints help the web search
    hints: List[str] = []
    n = (svc.get("service_name") or "").lower()
    if "move" in n:
        hints.append("Nutanix Move VM migration typical duration days estimate")
    if "fitcheck" in n:
        hints.append("Nutanix FitCheck assessment typical duration days")
    if "infrastructure" in n:
        hints.append("Nutanix AHV cluster deployment typical duration days")

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

    chosen = pick_days_with_rule(db_days=db_days, ai_days=ai_days)
    provider = "ai" if (ai_days is not None and ai_days >= db_days) else "db"

    return {
        "db_days": db_days,
        "ai_days": ai_days,
        "chosen_days": chosen,
        "provider": provider,
    }

# ---------------- planner API ----------------
def plan_suggestions(req: SuggestPlanRequest):
    scope = _extract_scope_details(req)
    needs = _detect_service_needs(req, scope)

    providers_txt = " ".join(req.providers or [])
    meta_txt = f"{req.deployment_type or ''} {req.industry or ''} {req.proposal_type or ''}".strip()
    boq_txt = " ".join([f"{b.sku or ''} {b.description or ''} {b.vendor or ''}" for b in (req.boq or [])])
    query_text = " ".join([
        req.requirements_text or "",
        boq_txt,
        providers_txt,
        req.selected_vendor or "",
        meta_txt
    ]).strip()

    platforms = req.constraints.target_platforms or None
    families = req.constraints.product_families or ["NDB", "NCI", "NC2", "NKP", "NUS"]

    candidates: List[Dict[str, Any]] = []

    def _pull(fam: str, service_type: str | None, q: str | None):
        rows = suggest_services_repo(
            product_family=fam,
            platforms=platforms,
            limit=30,
            service_type=service_type,
            q=q,
        )
        candidates.extend(rows)

    for fam in families:
        if needs['needs_assessment']:
            _pull(fam, None, 'fitcheck')
        if needs['needs_deployment']:
            _pull(fam, 'deployment', 'infrastructure')
        if needs['needs_migration']:
            _pull(fam, 'migration', None)
            if scope['source_platform'] == 'cisco_hyperflex':
                _pull(fam, 'migration', 'hyperflex')
        if needs['needs_db_migration']:
            _pull('NDB', 'migration', 'database')

    uniq: Dict[int, Dict[str, Any]] = {}
    for r in candidates:
        uniq[r["id"]] = r
    all_services = list(uniq.values())

    scored: List[Tuple[float, List[str], Dict[str, Any]]] = []
    for svc in all_services:
        rel_score, reasons = _calculate_relevance_score(svc, needs, scope, query_text)
        base_score = svc.get('priority_score', 0) * 0.3 + svc.get('popularity_score', 0) * 0.2
        final_score = base_score + rel_score
        scored.append((final_score, reasons, svc))
    scored.sort(key=lambda x: x[0], reverse=True)

    must_inc = req.constraints.must_include or []
    must_exc = req.constraints.must_exclude or []
    filtered: List[Tuple[float, List[str], Dict[str, Any]]] = []
    for score, reasons, svc in scored:
        namecat = f"{svc.get('service_name','')} {svc.get('category_name','')}".lower()
        if must_inc and not any(inc.lower() in namecat for inc in must_inc):
            continue
        if must_exc and any(exc.lower() in namecat for exc in must_exc):
            continue
        filtered.append((score, reasons, svc))

    core_services = _select_core_services([s[2] for s in filtered], needs, scope, limit=max(3, req.limit))

    llm_ids = llm_pick_services(
        query_text=query_text,
        providers=list(req.providers or []),
        shortlist_rows=[s[2] for s in filtered[:40]],
        limit=max(3, req.limit),
        required_platforms=(req.constraints.target_platforms or []),
        allowed_families=families,
        scope_context={
            'vm_count': scope['vm_count'],
            'node_count': scope['node_count'],
            'source_platform': scope['source_platform'],
            'needs_migration': needs['needs_migration'],
            'needs_deployment': needs['needs_deployment'],
        }
    ) or []

    chosen_ids = [svc['id'] for svc in core_services]
    if llm_ids:
        id_to_svc = {s[2]['id']: s[2] for s in filtered}
        chosen_ids = [i for i in llm_ids if i in id_to_svc]
        if len(chosen_ids) < 3:
            for svc in core_services:
                if svc['id'] not in chosen_ids:
                    chosen_ids.append(svc['id'])
                if len(chosen_ids) >= 3:
                    break

    final_ids = chosen_ids[:3]
    id_to_svc = {s[2]['id']: s[2] for s in filtered}
    selected_svcs: List[Dict[str, Any]] = [id_to_svc[i] for i in final_ids if i in id_to_svc]

    # --------- estimate days and overwrite duration_days ----------
    duration_sources: Dict[str, Dict[str, Any]] = {}
    for svc in selected_svcs:
        meta = _estimate_and_pick_days_for_service(
            svc=svc,
            scope=scope,
            industry=req.industry,
            deployment_type=req.deployment_type
        )
        # overwrite duration in-memory for downstream journey + response
        svc["duration_days"] = int(meta["chosen_days"])
        duration_sources[str(svc["id"])] = {
            "db_days": meta["db_days"],
            "ai_days": meta["ai_days"],
            "chosen_days": meta["chosen_days"],
            "provider": meta["provider"],
        }

    # --------- build response items ----------
    # recompute scores mapping for chosen IDs
    score_map: Dict[int, Tuple[float, List[str]]] = { s[2]['id']: (s[0], s[1]) for s in filtered }
    items: List[RankedService] = []
    for svc in selected_svcs:
        sc, reasons = score_map.get(svc["id"], (0.0, []))
        phase_tag = ''
        if 'fitcheck' in (svc.get('service_name','').lower()):
            phase_tag = 'Kickoff / Assessment & Planning'
        elif svc.get('service_type') == 'deployment':
            phase_tag = 'Infrastructure Setup'
        elif svc.get('service_type') == 'migration':
            phase_tag = 'Data Migration'

        reason_str = " | ".join(reasons) if reasons else "Relevant to scope"
        if phase_tag:
            reason_str = f"[{phase_tag}] " + reason_str

        items.append(RankedService(
            id=svc["id"],
            service_name=svc["service_name"],
            category_name=svc["category_name"],
            product_family=svc["product_family"],
            score=round(float(sc), 3),
            reason=reason_str,
            duration_days=int(svc["duration_days"]),  # already overwritten
            price_man_day=float(svc["price_man_day"]),
            service_type=svc.get("service_type"),
            supports_db_migration=bool(svc.get("supports_db_migration") or False),
            target_platforms=list(svc.get("target_platforms") or []),
            canonical_names=list(svc.get("canonical_names") or []),
            popularity_score=float(svc.get("popularity_score") or 0.0),
            priority_score=float(svc.get("priority_score") or 0.0),
        ))

    debug = {
        "query_text": query_text,
        "scope": scope,
        "needs": needs,
        "candidates_fetched": len(candidates),
        "unique_services": len(all_services),
        "services_suggested": len(items),
        "duration_sources": duration_sources,
    }

    return items, debug
