# app/routers/suggest.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Any, Dict, List, Tuple
import logging
import numpy as np
from decimal import Decimal

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

    resp_payload = {
        "items": items_serialized,
        "count": len(items_serialized),
        "journey": journey,
        "debug": _to_python_native(debug or {})
    }

    try:
        return SuggestPlanResponse(**resp_payload)
    except Exception as e:
        log.exception("Failed to validate SuggestPlanResponse: %s", e)
        raise HTTPException(status_code=500, detail=f"Response validation error: {e}")
