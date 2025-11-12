# app/routers/ranker_feedback.py
from math import sqrt
from typing import Dict, Tuple
from app.utils import supabase_ro, supabase_rw

ALPHA = 1.0
BETA = 1.0
USER_WEIGHT = 0.35
BLACKLIST_K = 3
MIN_SCORE = -1e9

def _beta_centered(pos:int, neg:int)->float:
    return ((pos + ALPHA) / (pos + neg + ALPHA + BETA)) - 0.5

def _shrink(n:int)->float:
    return min(1.0, sqrt(n / 20.0))  # ~20 votes to full weight

def _adj(pos:int, neg:int)->float:
    return _beta_centered(pos, neg) * _shrink(pos + neg)

def fetch_user_stats(user_id:str, service_ids:list[int]) -> dict[int, Tuple[int,int]]:
    if not service_ids: return {}
    r = supabase_ro.table("user_service_stats")\
        .select("service_id,pos,neg")\
        .eq("user_id", user_id)\
        .in_("service_id", service_ids).execute()
    rows = getattr(r, "data", []) or []
    return {x["service_id"]:(x["pos"], x["neg"]) for x in rows}

def apply_user_feedback(base_scores:Dict[int,float], user_id:str)->Dict[int,float]:
    stats = fetch_user_stats(user_id, list(base_scores.keys()))
    out = {}
    for sid, base in base_scores.items():
        pos, neg = stats.get(sid, (0,0))
        if neg >= BLACKLIST_K and pos == 0:
            out[sid] = MIN_SCORE
        else:
            out[sid] = base + USER_WEIGHT * _adj(pos, neg)
    return out
