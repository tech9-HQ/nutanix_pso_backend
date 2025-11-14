# app/routers/meta.py
from fastapi import APIRouter, Query
from typing import List, Dict, Any
from app.services.proposals_repo import supabase_ro  # adjust import if needed

# This router exposes metadata endpoints (categories, services, service details).
# prefix="" means these routes sit at the API root.
# Suggestion: Add a prefix "/meta" or include under /api/v1/meta for better structure.
router = APIRouter(prefix="", tags=["meta"])


@router.get("/categories")
async def categories() -> Dict[str, List[str]]:
    """
    Returns all unique service categories.

    Notes:
    - Currently loads all rows into memory and extracts categories.
      If table grows large, consider selecting distinct values using:
         .select("category_name").neq("category_name", None)
         OR Supabase RPC for distinct.
    - Ensure category_name is indexed in DB for improved lookup performance.
    """
    resp = supabase_ro.table("proposals").select("category_name").execute()
    rows = getattr(resp, "data", []) or []
    # Deduplicate and normalize names.
    cats = sorted({
        (r.get("category_name") or "").strip()
        for r in rows
        if r and r.get("category_name")
    })
    return {"categories": list(cats)}


@router.get("/services")
async def services_by_category(category: str = Query(...)) -> Dict[str, Any]:
    """
    Returns all services for a given category.

    Notes:
    - Query parameter `category` is required.
    - Ensure category_name is indexed to avoid full-table scans.
    - Response returns raw DB rows; consider normalizing field names/schemas.
    - Potential improvement: case-insensitive filtering using `.ilike` if needed.
    """
    resp = (
        supabase_ro.table("proposals")
        .select("service_name,price_man_day,duration_days,positioning,category_name")
        .eq("category_name", category)
        .execute()
    )
    return {"services": getattr(resp, "data", []) or []}


@router.get("/service_details")
async def service_details(service_name: str = Query(...)) -> Dict[str, Any]:
    """
    Return full details for a single service.

    Notes:
    - This returns the first matching row. If service_name is not unique,
      you may want to enforce uniqueness or return multiple rows.
    - Using .limit(1) is good practice.
    - Consider using ilike for case-insensitive matching.
    - Consider returning a standardized schema rather than raw DB response.
    """
    resp = (
        supabase_ro.table("proposals")
        .select("*")
        .eq("service_name", service_name)
        .limit(1)
        .execute()
    )
    row = (getattr(resp, "data", []) or [None])[0]
    return row or {}
