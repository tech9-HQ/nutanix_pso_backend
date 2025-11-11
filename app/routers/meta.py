# app/routers/meta.py
from fastapi import APIRouter, Query
from typing import List, Dict, Any
from app.services.proposals_repo import supabase_ro  # adjust import if your repo exposes this

router = APIRouter(prefix="", tags=["meta"])

@router.get("/categories")
async def categories() -> Dict[str, List[str]]:
    resp = supabase_ro.table("proposals").select("category_name").execute()
    rows = getattr(resp, "data", []) or []
    cats = sorted({ (r.get("category_name") or "").strip() for r in rows if r and r.get("category_name") })
    return {"categories": list(cats)}

@router.get("/services")
async def services_by_category(category: str = Query(...)) -> Dict[str, Any]:
    resp = (
        supabase_ro.table("proposals")
        .select("service_name,price_man_day,duration_days,positioning,category_name")
        .eq("category_name", category)
        .execute()
    )
    return {"services": getattr(resp, "data", []) or []}

@router.get("/service_details")
async def service_details(service_name: str = Query(...)) -> Dict[str, Any]:
    resp = supabase_ro.table("proposals").select("*").eq("service_name", service_name).limit(1).execute()
    row = (getattr(resp, "data", []) or [None])[0]
    return row or {}
