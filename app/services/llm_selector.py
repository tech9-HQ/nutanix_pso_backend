# app/services/llm_selector.py
from __future__ import annotations
import os, json, re, logging, requests
from typing import List, Dict, Any, Optional

log = logging.getLogger("llm_selector")

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
AZURE_ENDPOINT    = (os.getenv("AZURE_ENDPOINT", "") or "").rstrip("/")
AZURE_DEPLOYMENT  = os.getenv("AZURE_DEPLOYMENT", "") or ""
AZURE_API_KEY     = os.getenv("AZURE_API_KEY", "") or ""
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-10-21")  # used for Azure OpenAI style endpoints

# Supabase keys. Prefer explicit SUPABASE_KEY, fall back to anon key name.
SUPABASE_URL = os.getenv("SUPABASE_URL", "") or ""
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "") or ""

# -----------------------------------------------------------------------------
# Supabase helper
# -----------------------------------------------------------------------------
def _fetch_supabase(url: str, params: Dict[str, str]) -> List[Dict[str, Any]]:
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Supabase fetch failed: {e}")
        return []

# -----------------------------------------------------------------------------
# Azure Chat helper (supports proxy-style and Azure OpenAI-style endpoints)
# -----------------------------------------------------------------------------
def _azure_chat(prompt: str, max_tokens: int = 400) -> Optional[str]:
    if not (AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_API_KEY):
        log.warning("Azure LLM not configured.")
        return None

    try:
        headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}

        # Two common styles:
        # 1) Proxy/gateway:  {AZURE_ENDPOINT}/chat/completions   body: {"model": AZURE_DEPLOYMENT, ...}
        # 2) Azure OpenAI:   {AZURE_ENDPOINT}/openai/deployments/{DEP}/chat/completions?api-version=...
        #    body: {"messages":[...], "max_tokens":...}
        if "/openai/deployments/" in AZURE_ENDPOINT:
            # Caller passed fully qualified deployments URL in AZURE_ENDPOINT
            url = f"{AZURE_ENDPOINT}/chat/completions?api-version={AZURE_API_VERSION}"
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
        else:
            # Gateway style (what your logs showed earlier)
            url = f"{AZURE_ENDPOINT}/chat/completions"
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

# -----------------------------------------------------------------------------
# Lightweight scoring (fallback)
# -----------------------------------------------------------------------------
def _score_match(text: str, row: Dict[str, Any]) -> float:
    t = (text or "").lower()
    name = (row.get("service_name") or "").lower()
    fam = (row.get("product_family") or "").lower()
    stype = (row.get("service_type") or "").lower()
    targets = " ".join([p or "" for p in (row.get("target_platforms") or [])]).lower()
    score = 0.0
    if "nc2" in t and ("nc2" in name or fam == "nc2"): score += 2.0
    if "azure" in t and "azure" in targets: score += 2.0
    if "aws" in t and "aws" in targets: score += 1.0
    if ("db" in t or "database" in t) and (("ndb" in name) or fam == "ndb"): score += 2.0
    if "migration" in t and "migration" in stype: score += 1.5
    if "deployment" in t and "deployment" in stype: score += 1.0
    if "assessment" in t and "assessment" in stype: score += 0.8
    if ("dr" in t or "disaster" in t) and ("dr" in name or "recovery" in name): score += 1.2
    if fam in {"other"} and not any(k in t for k in ["euc","ai ","naigpt"]): score -= 0.6
    return score

# -----------------------------------------------------------------------------
# Public: LLM-driven service picking
# -----------------------------------------------------------------------------
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
    log.info("Starting LLM-based service selection")

    shortlist_rows = shortlist_rows or []
    if not shortlist_rows and SUPABASE_URL and SUPABASE_KEY:
        base = f"{SUPABASE_URL}/rest/v1/proposals_updated"
        fams = (allowed_families or ["NC2","NDB","NCI","NUS","NKP"])
        plats = (required_platforms or ["azure"])
        for fam in fams:
            params_base = {
                "select": "id,category_name,service_name,service_type,product_family,target_platforms,priority_score,popularity_score",
                "product_family": f"eq.{fam}",
                "order": "priority_score.desc.nullslast,popularity_score.desc.nullslast",
                "limit": "120",
            }
            for tp in plats:
                params = dict(params_base)
                params["target_platforms"] = f"ov.{{{tp}}}"
                shortlist_rows += _fetch_supabase(base, params)
            # generic rows
            params_null = dict(params_base)
            shortlist_rows += _fetch_supabase(base, params_null)

    db_preview = "\n".join(
        [f"- {r['id']}: {r['service_name']} ({r['product_family']}, {r['service_type']})"
         for r in shortlist_rows[:30]]
    )
    provider_txt = ", ".join(providers or [])
    scope_json = json.dumps(scope_context or {}, ensure_ascii=False)
    platform_txt = ", ".join(required_platforms or []) if required_platforms else ""
    family_txt = ", ".join(allowed_families or []) if allowed_families else ""

    prompt = f"""
You are a Nutanix Professional Services recommender.
Pick the best {limit} service IDs for the project. Use only IDs from the list.

Available services (sample):
{db_preview}

Project scope:
{requirements_text}

Providers: {provider_txt}
Target Platforms: {platform_txt}
Product Families: {family_txt}
Scope Context: {scope_json}

Rules:
- Prefer matching product_family and target_platform.
- Include assessment if sizing/discovery implied.
- Include deployment for NC2/NCI when moving to cloud/AHV.
- Include migration for VMs; include database services for DB move.
- Avoid EUC/OTHER unless EUC/AI is in scope.
Return JSON only: {{"ids":[<ids>]}}.
"""
    llm_raw = _azure_chat(prompt, max_tokens=600) or ""
    ids: List[int] = []
    if llm_raw:
        try:
            parsed = json.loads(llm_raw)
            ids = [int(x) for x in parsed.get("ids", [])]
        except Exception:
            ids = [int(x) for x in re.findall(r"\b\d{1,5}\b", llm_raw)]

    if not ids:
        hits = []
        for r in shortlist_rows:
            score = _score_match(requirements_text, r)
            if score > 0:
                hits.append((score, r["id"]))
        hits.sort(reverse=True)
        ids = [hid for _, hid in hits[:limit]]

    if not ids:
        ids = [r["id"] for r in shortlist_rows[:limit]]

    log.info(f"LLM selected service IDs: {ids}")
    return ids

# -----------------------------------------------------------------------------
# Internal util for LLM with fallback
# -----------------------------------------------------------------------------
def _coalesce_llm(prompt: str, fallback: str, max_tokens: int = 600) -> str:
    txt = _azure_chat(prompt, max_tokens=max_tokens)
    if txt and txt.strip():
        return txt.strip()
    return fallback.strip()

# -----------------------------------------------------------------------------
# Section generators used by router
# -----------------------------------------------------------------------------
def _llm_generate_introduction(company_name: str,
                               industry: Optional[str],
                               combined_requirements: Optional[str],
                               deployment_type: Optional[str]) -> str:
    fb = []
    fb.append("Introduction")
    fb.append(f"{company_name} is engaging Professional Services to achieve defined outcomes.")
    if industry:
        fb.append(f"Industry context: {industry}.")
    if deployment_type:
        fb.append(f"Deployment profile: {deployment_type}.")
    if combined_requirements:
        fb.append(f"High-level requirements: {combined_requirements[:1200]}")
    fallback = "\n\n".join(fb)
    prompt = f"""Write a crisp Introduction section for a Professional Services SOW.
Company: {company_name}
Industry: {industry or 'N/A'}
Deployment type: {deployment_type or 'N/A'}
Key requirements: {combined_requirements or 'N/A'}
No headings besides 'Introduction'. No bullets unless essential."""
    return _coalesce_llm(prompt, fallback)

def _llm_generate_executive_summary(company_name: str,
                                    combined_requirements: Optional[str],
                                    deployment_type: Optional[str]) -> str:
    fallback = "\n\n".join([
        "Executive Summary",
        f"This Statement of Work describes services to support {company_name}.",
        "Client Overview",
        f"{company_name} seeks predictable delivery and measurable outcomes.",
        "Requirement Summary",
        combined_requirements or "Requirements will be finalized during discovery.",
        "Assumptions",
        "- Work proceeds in agreed sprints\n- Stakeholders available for decisions\n- Environment access provided",
    ])
    prompt = f"""Write an Executive Summary for a PS SOW.
Company: {company_name}
Deployment: {deployment_type or 'N/A'}
Requirements:
{combined_requirements or 'N/A'}
Structure with short subsections: Client Overview, Requirement Summary, Assumptions."""
    return _coalesce_llm(prompt, fallback)

def _llm_generate_solution_summary(company_name: str,
                                   combined_requirements: Optional[str],
                                   services: List[Dict[str, Any]],
                                   industry: Optional[str],
                                   deployment_type: Optional[str]) -> str:
    base_rows = [
        "| Client Requirement | Proposed Solution / Approach | Expected Outcome |",
        "| --- | --- | --- |",
    ]
    if combined_requirements:
        base_rows.append("| Address documented needs | Align services to scope and constraints | Traceable deliverables |")
    if industry:
        base_rows.append(f"| Industry alignment | Apply {industry} reference practices | Faster adoption |")
    if deployment_type:
        base_rows.append(f"| Deployment model | Execute for {deployment_type} | Stable, supportable operations |")
    for s in services[:8]:
        name = (s.get("service_name") if isinstance(s, dict) else getattr(s, "service_name", "")) or "Service"
        base_rows.append(f"| Service coverage | Deliver {name} | Completed as per acceptance |")
    fallback = "\n".join(base_rows)
    prompt = f"""Create a 3-column Markdown table mapping requirements to solution and expected outcome.
Company: {company_name}
Industry: {industry or 'N/A'}
Deployment: {deployment_type or 'N/A'}
Key requirements: {combined_requirements or 'N/A'}
Services: {[ (s.get('service_name') if isinstance(s, dict) else getattr(s,'service_name', '')) for s in (services or []) ]}
Only the table. No extra headings."""
    return _coalesce_llm(prompt, fallback)

def _llm_generate_milestone_plan(services: List[Dict[str, Any]],
                                 duration_summary: str,
                                 deployment_type: Optional[str]) -> str:
    fb = [
        "| Phase | Key Activities | Duration (days) |",
        "| --- | --- | --- |",
        "| Discovery & Planning | Scope confirmation, access, success criteria | 2 |",
        "| Build & Configure | Implement baseline, integrations | 3 |",
        "| Validate & Handover | UAT, documentation, sign-off | 1 |",
    ]
    fallback = "\n".join(fb) + f"\n\nOverall timeline: {duration_summary}"
    prompt = f"""Produce a 3-column Markdown table for a milestone plan: Phase, Key Activities, Duration (days).
Consider deployment type: {deployment_type or 'N/A'}.
Keep 3–5 concise phases. No prose outside the table."""
    return _coalesce_llm(prompt, fallback)

def _llm_generate_acceptance_criteria(company_name: str,
                                      services: List[Dict[str, Any]],
                                      deployment_type: Optional[str]) -> str:
    items = [
        "Acceptance Criteria",
        "- All deliverables submitted and reviewed",
        "- Environment access and credentials handed over",
        "- Knowledge transfer session completed",
        "- Sign-off received from authorized stakeholder",
    ]
    if deployment_type:
        items.append(f"- Solution deployed for {deployment_type} as scoped")
    if services:
        items.append("- Service outputs match the defined scope for selected items")
    fallback = "\n".join(items)
    prompt = f"""List 6–10 clear, objective acceptance criteria for a PS SOW.
Company: {company_name}
Deployment: {deployment_type or 'N/A'}
Selected services: {[ (s.get('service_name') if isinstance(s, dict) else getattr(s,'service_name','')) for s in (services or []) ]}
Return as a simple bullet list with a single heading 'Acceptance Criteria'."""
    return _coalesce_llm(prompt, fallback)

# -----------------------------------------------------------------------------
# Service content generation
# -----------------------------------------------------------------------------
def generate_service_content(service_name: str,
                             service_info: Optional[Dict[str, Any]],
                             kb_pointers: Optional[List[Dict[str, Any]]],
                             duration_days: int,
                             *,
                             industry: Optional[str] = None,
                             service_comment: Optional[str] = None,
                             deployment_type: Optional[str] = None) -> str:
    duration_days = max(1, int(duration_days or 1))

    # Fallback deterministic content
    fb_lines = [
        f"{service_name}",
        "Objective",
        f"- Deliver the scoped outcomes for {service_name}",
        "Scope of Work",
        f"- Plan and execute core tasks over ~{duration_days} day(s)",
    ]
    if industry:
        fb_lines.append(f"- Consider {industry} constraints and best practices")
    if deployment_type:
        fb_lines.append(f"- Align to {deployment_type} deployment")
    if service_comment:
        fb_lines.append(service_comment)
    fb_lines += [
        "Deliverables",
        "- Configured components, runbooks, and documentation",
        "Assumptions & Dependencies",
        "- Timely access, change approvals, and SME availability",
    ]
    fallback = "\n".join(fb_lines)

    # Grounding text from KB pointers if any
    kb_text = ""
    if kb_pointers:
        pieces = []
        for kp in kb_pointers[:8]:
            t = (kp.get("text") or "").strip()
            if t:
                pieces.append(t[:600])
        kb_text = "\n\n".join(pieces)

    prompt = f"""Write a concise service section for a PS SOW.
Service: {service_name}
Duration (days): {duration_days}
Industry: {industry or 'N/A'}
Deployment: {deployment_type or 'N/A'}
Comment: {service_comment or ''}
Grounding excerpts:
{kb_text}

Sections to include as plain text in this order:
- Objective
- Scope of Work
- Deliverables
- Assumptions & Dependencies

Keep it brief and specific. No markdown headings, just plain lines."""
    return _coalesce_llm(prompt, fallback, max_tokens=800)

def create_comprehensive_fallback_content(service_name: str,
                                          positioning: Optional[str],
                                          duration_days: int) -> str:
    return generate_service_content(service_name, None, None, duration_days)

def generate_services_parallel(services: List[Dict[str, Any]], req_obj: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for s in services or []:
        if isinstance(s, dict):
            name = (s.get("service_name") or "Service").strip()
            dur = int(s.get("duration") or 1)
            cmt = s.get("comment") or ""
        else:
            name = (getattr(s, "service_name", None) or "Service").strip()
            dur = int(getattr(s, "duration", None) or 1)
            cmt = getattr(s, "comment", None) or ""
        out[name] = generate_service_content(
            name,
            None,
            None,
            dur,
            industry=getattr(req_obj, "industry", None),
            service_comment=cmt,
            deployment_type=getattr(req_obj, "deployment_type", None),
        )
    return out

# -----------------------------------------------------------------------------
# Project Governance (single authoritative function with LLM + fallback)
# -----------------------------------------------------------------------------
def generate_project_governance(
    client_name: str,
    industry: Optional[str] = None,
    services_overview: Optional[str] = None,
    deployment_type: Optional[str] = None,
) -> str:
    """
    Returns a governance section. Uses Azure LLM if configured,
    otherwise a deterministic fallback.
    """
    client = (client_name or "Client").strip()
    industry = industry or ""
    services_overview = services_overview or ""
    deployment_type = deployment_type or ""

    prompt = f"""
You are writing the "Project Governance" section of a professional services SOW.

Client: {client}
Industry: {industry}
Deployment type: {deployment_type}
Services overview: {services_overview}

Write a concise, practical governance section with headings:
- Governance Model
- Roles and Responsibilities
- Cadence and Reporting
- Risk and Change Management
- Escalation Matrix
- Communication Channels

Keep it vendor-neutral, use clear bullets, no marketing fluff, no duplicate headings.
Return plain text only.
"""
    txt = _azure_chat(prompt, max_tokens=700)
    if txt and txt.strip():
        return txt.strip()

    # Fallback if LLM not configured or failed
    return (
        "Governance Model\n"
        f"- A joint steering structure ensures scope alignment and timely decisions for {client}.\n"
        "- Weekly working reviews and a monthly steering review track progress and risks.\n\n"
        "Roles and Responsibilities\n"
        f"- {client}: Business ownership, SMEs, approvals, environment access.\n"
        "- Vendor: Project management, technical delivery, status reporting, risk management.\n\n"
        "Cadence and Reporting\n"
        "- Daily standup during active build/cutover phases.\n"
        "- Weekly status report covering scope, schedule, risks, issues, actions.\n"
        "- Monthly steering committee with milestones and decisions.\n\n"
        "Risk and Change Management\n"
        "- Risks/Issues logged with owner, impact, mitigation, and ETA.\n"
        "- Changes raised via formal Change Request with scope, effort, and timeline impact.\n\n"
        "Escalation Matrix\n"
        "- L1: Project Managers → L2: Practice/Account Leads → L3: Executive Sponsors.\n\n"
        "Communication Channels\n"
        "- Email for formal decisions, shared tracker for actions, approved chat for day-to-day.\n"
    )

# -----------------------------------------------------------------------------
# Explicit exports
# -----------------------------------------------------------------------------
__all__ = [
    "llm_pick_services",
    "_llm_generate_introduction",
    "_llm_generate_executive_summary",
    "_llm_generate_solution_summary",
    "_llm_generate_milestone_plan",
    "_llm_generate_acceptance_criteria",
    "generate_service_content",
    "create_comprehensive_fallback_content",
    "generate_services_parallel",
    "generate_project_governance",
]
