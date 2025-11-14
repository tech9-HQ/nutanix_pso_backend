# app/services/ranker_feedback.py
from __future__ import annotations

from datetime import datetime, timezone
from math import sqrt
from typing import Dict, Tuple, List

from app.utils import supabase_ro, supabase_rw

ALPHA = 1.0
BETA = 1.0
USER_WEIGHT = 0.35
BLACKLIST_K = 3
MIN_SCORE = -1e9


# ---------- helpers you already had ---------- #

def _beta_centered(pos: int, neg: int) -> float:
    return ((pos + ALPHA) / (pos + neg + ALPHA + BETA)) - 0.5


def _shrink(n: int) -> float:
    return min(1.0, sqrt(n / 20.0))  # ~20 votes to full weight


def _adj(pos: int, neg: int) -> float:
    return _beta_centered(pos, neg) * _shrink(pos + neg)


def fetch_user_stats(user_id: str, service_ids: List[int]) -> Dict[int, Tuple[int, int]]:
    if not service_ids:
        return {}
    r = (
        supabase_ro.table("user_service_stats")
        .select("service_id,pos,neg")
        .eq("user_id", user_id)
        .in_("service_id", service_ids)
        .execute()
    )
    rows = getattr(r, "data", []) or []
    return {x["service_id"]: (x["pos"], x["neg"]) for x in rows}


def apply_user_feedback(base_scores: Dict[int, float], user_id: str) -> Dict[int, float]:
    """
    base_scores: {service_id: base_score}
    returns:     {service_id: adjusted_score}
    """
    stats = fetch_user_stats(user_id, list(base_scores.keys()))
    out: Dict[int, float] = {}
    for sid, base in base_scores.items():
        pos, neg = stats.get(sid, (0, 0))
        if neg >= BLACKLIST_K and pos == 0:
            out[sid] = MIN_SCORE
        else:
            out[sid] = base + USER_WEIGHT * _adj(pos, neg)
    return out


# ---------- mutation: record like / dislike ---------- #

def record_feedback(user_id: str, service_id: int, vote: int) -> dict:
    """
    vote: 1 = like, -1 = dislike
    Updates user_service_stats and returns new stats + adjustment.
    """
    if vote not in (1, -1):
        return {"ok": False, "error": "vote must be 1 or -1"}

    # read current stats (if any)
    res = (
        supabase_rw.table("user_service_stats")
        .select("pos,neg")
        .eq("user_id", user_id)
        .eq("service_id", service_id)
        .limit(1)
        .execute()
    )
    rows = getattr(res, "data", []) or []
    if rows:
        pos = rows[0]["pos"]
        neg = rows[0]["neg"]
    else:
        pos = 0
        neg = 0

    if vote == 1:
        pos += 1
    else:
        neg += 1

    now_utc = datetime.now(timezone.utc).isoformat()

    row = {
        "user_id": user_id,
        "service_id": service_id,
        "pos": pos,
        "neg": neg,
        "last_update": now_utc,
    }

    # IMPORTANT: user_id + service_id should be the PK / unique index
    upsert_res = (
        supabase_rw.table("user_service_stats")
        .upsert(row, on_conflict="user_id,service_id")
        .execute()
    )

    if getattr(upsert_res, "error", None):
        return {"ok": False, "error": str(upsert_res.error)}

    adj = _adj(pos, neg)

    return {
        "ok": True,
        "user_id": user_id,
        "service_id": service_id,
        "pos": pos,
        "neg": neg,
        "adj": adj,
    }
