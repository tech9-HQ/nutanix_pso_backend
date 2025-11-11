# app/routers/suggest.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Form
from typing import Any, Dict, List, Tuple
import logging
import numpy as np
from decimal import Decimal
import itertools, math
from app.models.schemas import SuggestPlanRequest, SuggestPlanResponse
from app.services.ranker import plan_suggestions 

router = APIRouter(prefix="/suggest", tags=["suggest"])
log = logging.getLogger("suggest")

# ------------------ utils ------------------

def _to_python_native(val):
    if val is None:
        return None
    try:
        if isinstance(val, np.generic):
            return val.item()
        if isinstance(val, np.ndarray):
            return val.tolist()
    except Exception:
        pass
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return str(val)
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, (list, tuple, set)):
        return [_to_python_native(v) for v in list(val)]
    if isinstance(val, dict):
        return {str(k): _to_python_native(v) for k, v in val.items()}
    return val

def _serialize_item(item: Any) -> Dict[str, Any]:
    raw = None
    try:
        if hasattr(item, "model_dump"):
            raw = item.model_dump()
        elif hasattr(item, "dict"):
            raw = item.dict()
    except Exception:
        raw = None

    if raw is None:
        try:
            if hasattr(item, "__dict__"):
                raw = dict(item.__dict__)
        except Exception:
            raw = None

    if raw is None:
        try:
            raw = vars(item)
        except Exception:
            raw = None

    if raw is None:
        raw = dict(item) if isinstance(item, dict) else {"value": str(item)}

    normalized = {}
    for k, v in raw.items():
        if callable(v):
            continue
        normalized[str(k)] = _to_python_native(v)

    if "id" in normalized:
        try:
            normalized["id"] = int(normalized["id"])
        except Exception:
            pass

    return normalized

def _derive_score_and_reason(row: Dict[str, Any]) -> Tuple[float, str, List[str]]:
    score_val = 0.0
    reason_parts: List[str] = []

    s = row.get("_scores") or {}
    if isinstance(s, dict):
        try:
            score_val = float(s.get("final", 0.0) or 0.0)
        except Exception:
            score_val = 0.0
        if s.get("reason"):
            reason_parts.append(str(s.get("reason")))
        for key, label in (("keyword", "kw"), ("priority", "prio"), ("vector", "vec")):
            if key in s:
                try:
                    reason_parts.append(f"{label}={float(s.get(key, 0.0)):.2f}")
                except Exception:
                    pass

    explicit_reasons = row.get("_reasons")
    if explicit_reasons and isinstance(explicit_reasons, list):
        reason_parts = [str(x) for x in explicit_reasons]

    if not reason_parts:
        fam = row.get("product_family") or row.get("family") or ""
        targets = row.get("target_platforms") or row.get("targets") or []
        cat = row.get("category_name") or row.get("category") or ""
        if fam:
            reason_parts.append(str(fam))
        if targets:
            if isinstance(targets, (list, tuple)):
                reason_parts.append("targets:" + ",".join([str(t) for t in targets]))
            else:
                reason_parts.append(f"targets:{targets}")
        if cat:
            reason_parts.append(str(cat))

    return score_val, (", ".join(reason_parts) if reason_parts else "matched"), reason_parts

def _phase_order_from_debug(debug: dict) -> List[str]:
    ai = debug.get("scope", {}).get("ai", {})
    des = set(ai.get("desirable", []) or [])
    order = []
    if des.intersection({"assess", "fitcheck", "sizing", "poc"}):
        order.append("assessment")
    if des.intersection({"deploy", "deployment", "cluster", "infrastructure", "configure", "setup"}):
        order.append("deployment")
    if des.intersection({"migrate", "migration", "move", "rehost", "ec2"}):
        order.append("migration")
    if des.intersection({"database", "ndb", "dbs", "databases"}):
        order.append("database")
    if des.intersection({"dr", "disaster", "recovery", "replication", "protection", "metro", "near"}):
        order.append("dr")
    return order

def select_optimal_services(items: List[Dict[str, Any]], req: SuggestPlanRequest) -> Dict[str, Any]:
    """
    Returns a compact selection object without changing your response schema.
    The result is placed under debug['selection'] so SuggestPlanResponse stays valid.
    """
    constraints = getattr(req, "constraints", None)
    limit = int(getattr(req, "limit", 8) or 8)

    must_include_names = set()
    must_exclude_tokens = set()
    required_families = set()
    required_targets = set()

    if constraints:
        must_include_names = set([s.lower() for s in (constraints.must_include or [])])
        must_exclude_tokens = set([s.lower() for s in (constraints.must_exclude or [])])
        required_families = set([s.upper() for s in (constraints.product_families or [])])
        required_targets = set([s.lower() for s in (constraints.target_platforms or [])])

    def norm_name(x: str) -> str:
        return (x or "").strip().lower()

    def price_float(v) -> float:
        try:
            return float(v)
        except Exception:
            try:
                return float(str(v).replace(",", ""))
            except Exception:
                return 0.0

    def duration_int(v) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    # 1) Build candidate pool honoring excludes and filters
    pool: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for r in items:
        name = norm_name(r.get("service_name"))
        fam = (r.get("product_family") or "").upper()
        targets = [str(t).lower() for t in (r.get("target_platforms") or [])]

        # Exclude if any must_exclude token hits family or name
        if must_exclude_tokens:
            if fam.lower() in must_exclude_tokens or any(tok in name for tok in must_exclude_tokens):
                continue

        # If product_families specified, require membership
        if required_families and fam not in required_families:
            continue

        # If target_platforms specified, require overlap or empty targets allowed
        if required_targets:
            if targets and not (required_targets.intersection(targets)):
                continue

        pool.append(r)

    # 2) Always include must_include items, even if filtered out above, with notes
    forced: List[Dict[str, Any]] = []
    for minc in must_include_names:
        # find highest scored match by name contains or exact
        matches = [
            r for r in items
            if minc in norm_name(r.get("service_name")) or norm_name(r.get("service_name")) == minc
        ]
        if matches:
            # pick max by score
            pick = max(matches, key=lambda x: float((x.get("score") or 0.0)))
            # annotate conflict if target mismatch
            fam = (pick.get("product_family") or "").upper()
            targets = [str(t).lower() for t in (pick.get("target_platforms") or [])]
            conflict = False
            if required_targets and targets and not (required_targets.intersection(targets)):
                conflict = True
                warnings.append(
                    f"Service '{pick.get('service_name')}' was required by must_include but its "
                    f"target_platforms={targets} do not match constraints.target_platforms={list(required_targets)}. "
                    f"Included anyway."
                )
            if required_families and fam not in required_families:
                conflict = True
                warnings.append(
                    f"Service '{pick.get('service_name')}' was required by must_include but product_family='{fam}' "
                    f"not in constraints.product_families={list(required_families)}. Included anyway."
                )
            forced.append((pick, conflict))

    # 3) Rank remaining pool by your computed score
    ranked = sorted(pool, key=lambda x: float((x.get("score") or 0.0)), reverse=True)

    # 4) Compose the selection: forced first (dedup), then ranked
    seen_keys = set()
    selected: List[Dict[str, Any]] = []

    def add_row(r: Dict[str, Any], note: str | None = None):
        key = (norm_name(r.get("service_name")), (r.get("product_family") or "").upper())
        if key in seen_keys:
            return
        seen_keys.add(key)
        price = price_float(r.get("price_man_day"))
        dur = duration_int(r.get("estimate", {}).get("chosen_days") or r.get("duration_days") or 0)
        entry = {
            "id": r.get("id"),
            "service_name": r.get("service_name"),
            "product_family": (r.get("product_family") or "").upper(),
            "duration_days": dur,
            "price_man_day": f"{price:.2f}",
            "cost": round(price * dur, 2),
        }
        if note:
            entry["note"] = note
        selected.append(entry)

    for pick, conflict in forced:
        add_row(
            pick,
            note="Included due to must_include" + (" (constraint conflict)" if conflict else "")
        )
        if len(selected) >= limit:
            break

    for r in ranked:
        if len(selected) >= limit:
            break
        add_row(r)

    # If nothing selected, return empty with suggestion
    if not selected:
        return {
            "selected_services": [],
            "total_duration_days": 0,
            "total_cost": 0,
            "warnings": warnings or ["No services matched the provided constraints. Check product_families, target_platforms, or must_exclude settings."],
            "suggestions": [
                "Relax must_exclude / product_families or add must_include services.",
                "Try leaving product_families empty to allow any family.",
            ],
        }
    
    total_days = sum(int(x["duration_days"]) for x in selected)
    total_cost = round(sum(float(x["cost"]) for x in selected), 2)

    # --- Concise proposal mode trim ---
    if getattr(req, "proposal_type", None) == "concise":
        ndp = [
            r for r in items
            if (r.get("product_family") or "").upper() == "NDB"
            and "ahv" in [t.lower() for t in (r.get("target_platforms") or [])]
        ]
        ndp.sort(
            key=lambda x: (
                float(x.get("score", 0.0)),
                float(x.get("priority_score", 0.0)),
                float(x.get("popularity_score", 0.0)),
            ),
            reverse=True,
        )
        limit = int(getattr(req, "limit", 5) or 5)
        top5 = ndp[:limit]

        concise_selected = []
        concise_days = concise_cost = 0
        for r in top5:
            dur = int(r.get("duration_days") or 0)
            price = float(r.get("price_man_day") or 0)
            cost = round(dur * price, 2)
            concise_selected.append({
                "id": r.get("id"),
                "service_name": r.get("service_name"),
                "duration_days": dur,
                "price_man_day": f"{price:.2f}",
                "cost": cost,
            })
            concise_days += dur
            concise_cost += cost

        return {
            "selected_services": concise_selected,
            "total_duration_days": concise_days,
            "total_cost": concise_cost,
            "notes": [
                "Selected top 5 NDB services targeting AHV based on scoring (priority/popularity).",
                "If you want per-DB scaling (3 DBs), enable DB-scaling logic; current cost model is per-service.",
            ],
        }

    # --- Default full mode ---
    return {
        "selected_services": selected,
        "total_duration_days": total_days,
        "total_cost": total_cost,
        "warnings": warnings,
        "notes": [
            "Selection ranked by model score. Forced inclusions appear first.",
            "Costs use duration_days * price_man_day as provided by ranking output.",
        ],
    }


# ------------------ route ------------------

@router.post("/plan", response_model=SuggestPlanResponse)
def suggest_plan(req: SuggestPlanRequest):
    try:
        out = plan_suggestions(req)
    except Exception as e:
        log.exception("plan_suggestions failed")
        raise HTTPException(status_code=500, detail=f"plan_suggestions error: {e}")

    if isinstance(out, tuple) and len(out) >= 2:
        items_raw, debug = out[0], out[1]
    else:
        items_raw, debug = out, {}

    # serialize items
    items_serialized: List[Dict[str, Any]] = []
    for itm in items_raw or []:
        row = _serialize_item(itm) if not isinstance(itm, dict) else {k: _to_python_native(v) for k, v in itm.items()}

        # score + reasons
        score_val, reason_text, reason_list = _derive_score_and_reason(row)
        row["score"] = float(row.get("score") or score_val or 0.0)
        row["reasons"] = reason_list or row.get("reasons") or []
        row["reason"] = ", ".join(str(x) for x in (row["reasons"][:8] if row.get("reasons") else [reason_text]))

        # ensure estimate
        if "estimate" not in row or not isinstance(row["estimate"], dict):
            try:
                db_days = int(row.get("duration_days") or 1)
            except Exception:
                db_days = 1
            row["estimate"] = {"db_days": db_days, "ai_days": None, "chosen_days": db_days, "provider": "db"}
        else:
            # type safety
            try:
                row["estimate"]["db_days"] = int(row["estimate"].get("db_days") or 0)
                row["estimate"]["chosen_days"] = int(row["estimate"].get("chosen_days") or 0)
                if row["estimate"].get("ai_days") is not None:
                    row["estimate"]["ai_days"] = float(row["estimate"]["ai_days"])
            except Exception:
                pass

        # price coercion
        if "price_man_day" in row:
            try:
                row["price_man_day"] = float(row["price_man_day"]) if row["price_man_day"] is not None else None
            except Exception:
                pass

        # cost fallback
        if "cost_estimate" not in row:
            try:
                price = float(row.get("price_man_day")) if row.get("price_man_day") is not None else 0.0
            except Exception:
                try:
                    price = float(str(row.get("price_man_day")).replace(",", ""))
                except Exception:
                    price = 0.0
            chosen = int(row["estimate"].get("chosen_days") or row.get("duration_days") or 1)
            row["cost_estimate"] = round(price * chosen, 2)

        items_serialized.append(row)

    # normalize scores
    max_score = max((it.get("score") or 0.0) for it in items_serialized) if items_serialized else 0.0
    for it in items_serialized:
        try:
            it["score_normalized"] = round(float(it.get("score", 0.0)) / float(max_score) * 100.0, 2) if max_score > 0 else round(float(it.get("score", 0.0)) * 10.0, 2)
        except Exception:
            it["score_normalized"] = float(it.get("score", 0.0))

    order = _phase_order_from_debug(_to_python_native(debug or {}))
    # Build structured journey summary
    journey = {
        "phases": [
            {
                "phase": "assessment",
                "services": [s for s in items_serialized if "assessment" in (s.get("service_type") or "").lower()],
                "phase_days": sum(
                    int(s.get("estimate", {}).get("chosen_days") or s.get("duration_days") or 0)
                    for s in items_serialized
                    if "assessment" in (s.get("service_type") or "").lower()
                ),
                "phase_cost_usd": sum(
                    float(s.get("cost_estimate") or 0.0)
                    for s in items_serialized
                    if "assessment" in (s.get("service_type") or "").lower()
                ),
            },
            {
                "phase": "deployment",
                "services": [s for s in items_serialized if "deployment" in (s.get("service_type") or "").lower()],
                "phase_days": sum(
                    int(s.get("estimate", {}).get("chosen_days") or s.get("duration_days") or 0)
                    for s in items_serialized
                    if "deployment" in (s.get("service_type") or "").lower()
                ),
                "phase_cost_usd": sum(
                    float(s.get("cost_estimate") or 0.0)
                    for s in items_serialized
                    if "deployment" in (s.get("service_type") or "").lower()
                ),
            },
            {
                "phase": "migration",
                "services": [s for s in items_serialized if "migration" in (s.get("service_type") or "").lower()],
                "phase_days": sum(
                    int(s.get("estimate", {}).get("chosen_days") or s.get("duration_days") or 0)
                    for s in items_serialized
                    if "migration" in (s.get("service_type") or "").lower()
                ),
                "phase_cost_usd": sum(
                    float(s.get("cost_estimate") or 0.0)
                    for s in items_serialized
                    if "migration" in (s.get("service_type") or "").lower()
                ),
            },
            {
                "phase": "database",
                "services": [
                    s for s in items_serialized
                    if "database" in (s.get("category_name") or "").lower()
                    or s.get("product_family") == "NDB"
                ],
                "phase_days": sum(
                    int(s.get("estimate", {}).get("chosen_days") or s.get("duration_days") or 0)
                    for s in items_serialized
                    if "database" in (s.get("category_name") or "").lower()
                    or s.get("product_family") == "NDB"
                ),
                "phase_cost_usd": sum(
                    float(s.get("cost_estimate") or 0.0)
                    for s in items_serialized
                    if "database" in (s.get("category_name") or "").lower()
                    or s.get("product_family") == "NDB"
                ),
            },
            {
                "phase": "dr",
                "services": [
                    s for s in items_serialized
                    if any(k in (s.get("service_name") or "").lower() for k in ["dr", "disaster", "recovery", "protection", "metro"])
                ],
                "phase_days": sum(
                    int(s.get("estimate", {}).get("chosen_days") or s.get("duration_days") or 0)
                    for s in items_serialized
                    if any(k in (s.get("service_name") or "").lower() for k in ["dr", "disaster", "recovery", "protection", "metro"])
                ),
                "phase_cost_usd": sum(
                    float(s.get("cost_estimate") or 0.0)
                    for s in items_serialized
                    if any(k in (s.get("service_name") or "").lower() for k in ["dr", "disaster", "recovery", "protection", "metro"])
                ),
            },
        ],
        "totals": {
            "days": sum(
                int(s.get("estimate", {}).get("chosen_days") or s.get("duration_days") or 0)
                for s in items_serialized
            ),
            "cost_usd": sum(
                float(s.get("cost_estimate") or 0.0)
                for s in items_serialized
            ),
        },
    }

    # ---- optimal selection (smart combinatorial pick) ----
    selection = {}
    try:
        selection = select_optimal_services(items_serialized, req)
    except Exception as e:
        log.exception("select_optimal_services failed: %s", e)
        selection = {"error": f"selection_failed: {str(e)}"}


    resp_payload = {
        "items": items_serialized,
        "count": len(items_serialized),
        "journey": journey,
        "debug": {**_to_python_native(debug or {}), "selection": selection},
    }


    try:
        return SuggestPlanResponse(**resp_payload)
    except Exception as e:
        log.exception("Failed to validate SuggestPlanResponse: %s", e)
        raise HTTPException(status_code=500, detail=f"Response validation error: {e}")


@router.post("", response_model=SuggestPlanResponse)
def suggest_plan_form(
    client_name: str = Form("Client"),
    industry: str | None = Form(None),
    requirements_text: str = Form(...),
    top_k: int = Form(6),
    # optional FE extras:
    hardware_providers: str | None = Form(None),  # e.g. "NUTANIX, DELL"
    boq_text: str | None = Form(None),
    limit: int = Form(8),
    proposal_type: str | None = Form(None),       # e.g. "concise"
):
    """
    Accepts FormData from the frontend and forwards it to the JSON /suggest/plan
    by constructing a SuggestPlanRequest payload.
    """
    # Parse vendors string into tokens for constraints.must_exclude (example policy).
    # Adjust to your needs (e.g., must_include or product_families) if desired.
    vendor_tokens = []
    if hardware_providers:
        vendor_tokens = [v.strip().lower() for v in hardware_providers.split(",") if v.strip()]

    payload = {
        "client_name": client_name,
        "industry": industry,
        "requirements_text": requirements_text,
        "top_k": top_k,
        "limit": limit,
        "proposal_type": proposal_type,
        # You can pass any extra context your planner uses:
        "context": {
            "hardware_providers": vendor_tokens,
            "boq_text": boq_text or "",
        },
        # Basic constraints example: exclude services containing vendor tokens
        "constraints": {
            "must_exclude": vendor_tokens,
            # leave others empty so your selector stays permissive by default
            "must_include": [],
            "product_families": [],
            "target_platforms": [],
        },
    }

    try:
        req_obj = SuggestPlanRequest(**payload)
    except Exception as e:
        # If your Pydantic schema names differ, inspect and adapt the keys above.
        raise HTTPException(status_code=422, detail=f"Invalid suggest payload: {e}")

    return suggest_plan(req_obj)