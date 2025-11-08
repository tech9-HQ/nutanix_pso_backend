# app/routers/estimate.py
from __future__ import annotations
import os, re, math, time, statistics, logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
import requests

from app.services.duration_estimator import estimate_days_from_web, pick_days_with_rule

log = logging.getLogger("estimate_days")
router = APIRouter(prefix="", tags=["estimate"])

# ---------------- Supabase REST ----------------
_SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
_SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

def _fetch_proposals_rows(table: str = "proposals_updated") -> List[Dict[str, Any]]:
    if not (_SUPABASE_URL and _SUPABASE_ANON_KEY):
        log.error("SUPABASE_URL or SUPABASE_ANON_KEY missing")
        return []

    sel = ",".join([
        "id","category_name","service_name","positioning",
        "duration_days","price_man_day","priority_score","popularity_score"
    ])
    url = f"{_SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": _SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {_SUPABASE_ANON_KEY}",
        "Accept": "application/json",
    }
    params = {
        "select": sel,
        "limit": 1000,
        "order": "priority_score.desc.nullslast,popularity_score.desc.nullslast,duration_days.asc.nullsfirst"
    }
    r = requests.get(url, headers=headers, params=params, timeout=20)
    if not r.ok:
        log.error("Supabase REST error %s: %s", r.status_code, r.text[:300])
        return []
    return r.json() or []

# ---------------- Optional local embeddings ----------------
_MODEL = None
_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
try:
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer(_MODEL_NAME)
    log.info(f"Embeddings model loaded: {_MODEL_NAME}")
except Exception as _e:
    _MODEL = None
    log.warning(f"Embeddings model not available: {_e}")

_PROPOSAL_EMB_CACHE: Dict[str, List[float]] = {}
_PROPOSAL_EMB_CACHE_TS = 0.0

def _cos_sim(a: List[float], b: List[float]) -> float:
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return float(dot / (da * db))

@router.post("/estimate_days")
async def estimate_days(
    service_name: str = Form(...),
    requirements_text: Optional[str] = Form(None),
    method: str = Form("similarity"),
    top_k: int = Form(8),
):
    try:
        service_name_clean = (service_name or "").strip()
        if not service_name_clean:
            return JSONResponse({"error": "service_name required"}, status_code=400)

        req_text = (requirements_text or "").strip()
        combined_query = (service_name_clean + "\n\n" + req_text).strip()

        proposals_rows = _fetch_proposals_rows("proposals_updated")
        if not proposals_rows:
            return JSONResponse({"error": "no proposals found"}, status_code=500)

        # -------- heuristic from text --------
        text = req_text.lower()
        try:
            nodes = int(re.search(r"(\d{1,4})\s+nodes?", text).group(1))
        except Exception:
            nodes = 0
        m = re.search(r"(\d{1,6})\s*(?:vms?|workloads?)", text)
        workloads = int(m.group(1)) if m else 0
        cloud = any(k in text for k in ["aws", "azure", "gcp", "hybrid"])
        dr = any(k in text for k in ["replication", " dr ", "rpo", "rto", "disaster recovery"])
        compliance = sum(k in text for k in ["rbi", "sebi", "iso", "pci", "gdpr"])

        base = 1.0
        workload_days = workloads * 0.02
        node_days = nodes * 0.4
        overhead = (3 if dr else 0) + (4 if cloud else 0) + (compliance * 2)
        heuristic_days = max(1.0, base + workload_days + node_days + overhead)

        # -------- embeddings similarity (optional) --------
        sim_est = None
        sim_details: List[Dict[str, Any]] = []
        global _PROPOSAL_EMB_CACHE, _PROPOSAL_EMB_CACHE_TS
        now_ts = time.time()

        if _MODEL and (not _PROPOSAL_EMB_CACHE or (now_ts - _PROPOSAL_EMB_CACHE_TS) > 6 * 3600):
            _PROPOSAL_EMB_CACHE = {}
            _PROPOSAL_EMB_CACHE_TS = now_ts
            txts, ids = [], []
            for pr in proposals_rows:
                pid = str(pr.get("id") or pr.get("service_name") or "")[:120]
                txt = (pr.get("positioning") or pr.get("service_name") or "").strip()
                if not txt:
                    continue
                ids.append(pid); txts.append(txt)
            if txts:
                try:
                    embs = _MODEL.encode(txts, convert_to_numpy=True)
                    for pid, emb in zip(ids, embs):
                        _PROPOSAL_EMB_CACHE[pid] = emb.tolist()
                except Exception:
                    _PROPOSAL_EMB_CACHE = {}

        if _MODEL:
            try:
                q_emb = _MODEL.encode([combined_query], convert_to_numpy=True)[0].tolist()
                name_emb = _MODEL.encode([service_name_clean], convert_to_numpy=True)[0].tolist()
                scored = []
                for pr in proposals_rows:
                    pid = str(pr.get("id") or pr.get("service_name") or "")[:120]
                    txt = (pr.get("positioning") or pr.get("service_name") or "").strip()
                    if not txt:
                        continue
                    emb = _PROPOSAL_EMB_CACHE.get(pid)
                    if emb is None:
                        try:
                            emb = _MODEL.encode([txt], convert_to_numpy=True)[0].tolist()
                        except Exception:
                            continue
                    sim_q = _cos_sim(q_emb, emb)
                    sim_n = _cos_sim(name_emb, emb)
                    comb = 0.65 * sim_q + 0.35 * sim_n
                    scored.append({
                        "service_name": pr.get("service_name"),
                        "duration_days": int(pr.get("duration_days") or 1),
                        "sim_q": round(sim_q, 4),
                        "name_sim": round(sim_n, 4),
                        "combined_sim": comb,
                    })
                if scored:
                    scored.sort(key=lambda x: x["combined_sim"], reverse=True)
                    top = scored[:max(1, int(top_k))]
                    total_w = sum(max(0.0, t["combined_sim"]) for t in top)
                    if total_w > 0:
                        sim_est = sum(t["duration_days"] * max(0.0, t["combined_sim"]) for t in top) / total_w
                    else:
                        sim_est = statistics.median([t["duration_days"] for t in top])
                    sim_details = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()} for t in top]
            except Exception:
                sim_est = None

        # exact-name DB baseline
        matches = [
            int(pr.get("duration_days") or 1)
            for pr in proposals_rows
            if (pr.get("service_name") or "").strip().lower() == service_name_clean.lower()
        ]
        db_baseline = int(statistics.median(matches)) if matches else (
            int(round(sim_est)) if sim_est is not None else None
        )

        # -------- web+LLM estimator --------
        src = "vmware" if any(w in text for w in ["vmware", "vsphere", "vcenter", "esxi"]) else None
        tgt = "ahv" if "ahv" in text else None
        dep = "on prem" if ("on prem" in text or "on-prem" in text) else None

        ai_est = estimate_days_from_web(
            task_text=f"{service_name_clean} {req_text}".strip(),
            industry=None,
            deployment_type=dep,
            source_platform=src,
            target_platform=tgt,
            vm_count=workloads,
            node_count=nodes,
            search_hints=[service_name_clean],
        )

        final_ai_days = None
        if db_baseline is not None and ai_est is not None:
            final_ai_days = pick_days_with_rule(db_baseline, ai_est)
        elif ai_est is not None:
            final_ai_days = int(ai_est)

        # -------- blend for recommendation --------
        comps: List[float] = []
        wts: List[float] = []
        if final_ai_days is not None:
            comps.append(float(final_ai_days)); wts.append(0.45)
        if sim_est is not None:
            comps.append(float(sim_est)); wts.append(0.35 if final_ai_days is not None else 0.55)
        comps.append(float(heuristic_days)); wts.append(0.20 if (final_ai_days or sim_est) else 1.0)

        tw = sum(wts)
        final_avg = sum(v * (w / tw) for v, w in zip(comps, wts)) if tw else 1.0
        recommended = max(1, int(round(final_avg)))
        recommended_min = max(1, int(math.floor(final_avg * 0.85)))
        recommended_max = max(1, int(math.ceil(final_avg * 1.25)))

        vals_present = [v for v in [final_ai_days, sim_est, heuristic_days] if v is not None]
        spread = (max(vals_present) - min(vals_present)) if len(vals_present) > 1 else 0.0
        confidence = round(max(0.1, min(0.99, 1.0 - (spread / (final_avg + 1e-6)) * 0.35)), 2)

        return JSONResponse({
            "service_name": service_name_clean,
            "recommended_days": int(recommended),
            "recommended_min": int(recommended_min),
            "recommended_max": int(recommended_max),
            "confidence": confidence,
            "facts": {"nodes": nodes, "workloads": workloads, "cloud": cloud, "dr": dr, "compliance": compliance},
            "breakdown": {
                "db_baseline": db_baseline,
                "ai_web_est": ai_est,
                "final_ai_days": final_ai_days,
                "similarity_est": (None if sim_est is None else float(f"{sim_est:.3f}")),
                "heuristic_est": float(f"{heuristic_days:.3f}"),
                "similarity_details": sim_details,
            },
            "method": "ai_web_over_db_else_similarity_heuristic"
        })

    except Exception as e:
        log.exception("estimate_days failed")
        return JSONResponse({"error": "internal error", "detail": str(e)}, status_code=500)
