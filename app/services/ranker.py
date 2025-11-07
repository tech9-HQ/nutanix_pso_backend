# app/services/ranker.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import re

from app.models.schemas import SuggestPlanRequest, RankedService
from app.services.proposals_repo import suggest_services_repo
from app.services.llm_selector import llm_pick_services


# ============== token utils ==============

def _tokenize(s: str) -> List[str]:
    if not s:
        return []
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return s.split()


# ============== scope + need detection ==============

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


# ============== relevance scoring ==============

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

    # hard de-bias: no DB scope => penalize "database" services
    if not scope['has_database']:
        if 'database' in svc_name or 'database' in category:
            score -= 0.25
            reasons.append('No DB scope; deprioritize database services')

    # assessment / fitcheck
    if 'assessment' in category or 'fitcheck' in svc_name:
        if needs['needs_assessment']:
            score += 0.20
            reasons.append('Assessment needed for sizing')
            if scope['vm_count'] > 0:
                score += 0.10
                reasons.append(f"Sizing for {scope['vm_count']} VMs")
        else:
            score -= 0.15

    # deployment
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

    # vm migration (non-db)
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

    # db migration services
    if 'database' in category and 'migration' in svc_name:
        if needs['needs_db_migration']:
            score += 0.30
            reasons.append('Database migration required')
        else:
            score -= 0.35

    # dev services
    if 'development' in category:
        if needs['needs_development']:
            score += 0.20
            reasons.append('Custom development needed')
        else:
            score -= 0.25

    # ai/analytics
    if 'ai' in category or 'analytics' in category:
        if needs['needs_ai_analytics']:
            score += 0.25
            reasons.append('AI/Analytics workload')
        else:
            score -= 0.30

    # optimization
    if any(k in svc_name for k in ['health check', 'optimization', 'performance']):
        if needs['needs_optimization']:
            score += 0.15
            reasons.append('Optimization services')
        else:
            score -= 0.10

    # keyword overlap
    query_tokens = set(_tokenize(query_text))
    service_tokens = set(_tokenize(f"{svc_name} {category}"))
    overlap = len(query_tokens & service_tokens)
    if overlap > 0:
        kw_score = min(0.15, overlap * 0.03)
        score += kw_score
        reasons.append(f"{overlap} keyword matches")

    return score, reasons


# ============== minimal service set selection ==============

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

    # Phase 1: Assessment — prefer NCI FitCheck Starter, then other NCI FitCheck, then any FitCheck
    if needs['needs_assessment']:
        pick_best(lambda s: 'fitcheck' in s.get('service_name','').lower()
                           and s.get('product_family') == 'NCI'
                           and 'starter' in s.get('service_name','').lower(), 1)
        if not selected:
            pick_best(lambda s: 'fitcheck' in s.get('service_name','').lower()
                               and s.get('product_family') == 'NCI', 1)
        if not selected:
            pick_best(lambda s: 'fitcheck' in s.get('service_name','').lower(), 1)

    # Phase 2: Deployment — prefer "Infrastructure Deployment" then "Infrastructure Expansion"
    if needs['needs_deployment']:
        picked_before = len(selected)
        pick_best(lambda s: s.get('product_family') == 'NCI'
                           and 'infrastructure deployment' in s.get('service_name','').lower(), 1)
        if len(selected) == picked_before:
            pick_best(lambda s: s.get('product_family') == 'NCI'
                               and 'infrastructure expansion' in s.get('service_name','').lower(), 1)

    # Phase 3: Migration — prefer Nutanix Move
    if needs['needs_migration']:
        pick_best(lambda s: s.get('service_type') == 'migration'
                           and 'move' in s.get('service_name','').lower(), 1)
        if scope.get('source_platform') == 'cisco_hyperflex':
            pick_best(lambda s: 'hyperflex' in s.get('service_name','').lower(), 1)

    return selected[:max(1, limit)]


# ============== planner API ==============

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

    # targeted pulls based on detected needs
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

    # dedupe
    uniq: Dict[int, Dict[str, Any]] = {}
    for r in candidates:
        uniq[r["id"]] = r
    all_services = list(uniq.values())

    # score
    scored: List[Tuple[float, List[str], Dict[str, Any]]] = []
    for svc in all_services:
        rel_score, reasons = _calculate_relevance_score(svc, needs, scope, query_text)
        base_score = svc.get('priority_score', 0) * 0.3 + svc.get('popularity_score', 0) * 0.2
        final_score = base_score + rel_score
        scored.append((final_score, reasons, svc))
    scored.sort(key=lambda x: x[0], reverse=True)

    # include/exclude filters
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

    # minimal set via phase chooser
    core_services = _select_core_services([s[2] for s in filtered], needs, scope, limit=max(3, req.limit))

    # LLM refinement (keep within shortlist)
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

    # if LLM returns something, respect its order
    chosen_ids = [svc['id'] for svc in core_services]
    if llm_ids:
        id_to_svc = {svc['id']: svc for svc in [s[2] for s in filtered]}
        chosen_ids = [i for i in llm_ids if i in id_to_svc]
        # ensure at least minimal set if LLM returned fewer than 3
        if len(chosen_ids) < 3:
            for svc in core_services:
                if svc['id'] not in chosen_ids:
                    chosen_ids.append(svc['id'])
                if len(chosen_ids) >= 3:
                    break

    # final assembly, cap to exactly 3 items
    final_ids = chosen_ids[:3]
    id_to_scored = {s[2]['id']: (s[0], s[1], s[2]) for s in filtered}
    final_triples: List[Tuple[float, List[str], Dict[str, Any]]] = []
    for sid in final_ids:
        if sid in id_to_scored:
            final_triples.append(id_to_scored[sid])

    # map to response
    items: List[RankedService] = []
    for score, reasons, svc in final_triples:
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
            score=round(float(score), 3),
            reason=reason_str,
            duration_days=int(svc["duration_days"]),
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
    }

    return items, debug
