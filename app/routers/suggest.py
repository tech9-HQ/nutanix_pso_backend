# app/routers/suggest.py
from typing import List, Optional
from fastapi import APIRouter, Query

from app.models.schemas import (
    SuggestResponse,
    ServiceItem,
    SuggestPlanRequest,
    SuggestPlanResponse,
)
from app.services.proposals_repo import suggest_services_repo
from app.services.ranker import plan_suggestions
from app.services.journey import make_journey  # <-- add

router = APIRouter(prefix="/suggest", tags=["suggest"])


@router.get("/services", response_model=SuggestResponse)
def suggest_services(
    product_family: Optional[str] = Query(None),
    target_platforms: Optional[str] = Query(None, description="comma separated"),
    limit: int = Query(6, ge=1, le=50),
    service_type: Optional[str] = Query(None, description="deployment|migration|assessment|design"),
    supports_db_migration: Optional[bool] = Query(None),
    max_duration: Optional[int] = Query(None, ge=1),
    price_cap: Optional[float] = Query(None, gt=0),
    q: Optional[str] = Query(None, description="keyword search"),
) -> SuggestResponse:
    platforms_list: Optional[List[str]] = None
    if target_platforms:
        platforms_list = [p.strip() for p in target_platforms.split(",") if p.strip()]

    rows = suggest_services_repo(
        product_family=product_family,
        platforms=platforms_list,
        limit=limit,
        service_type=service_type,
        supports_db_migration=supports_db_migration,
        max_duration=max_duration,
        price_cap=price_cap,
        q=q,
    )

    items: List[ServiceItem] = []
    for r in rows:
        # normalize types
        if isinstance(r.get("price_man_day"), str):
            try:
                r["price_man_day"] = float(r["price_man_day"])
            except Exception:
                pass
        r["canonical_names"] = r.get("canonical_names") or []
        r["target_platforms"] = r.get("target_platforms") or []
        items.append(ServiceItem(**r))

    return SuggestResponse(count=len(items), items=items)


@router.post("/plan", response_model=SuggestPlanResponse)
def suggest_plan(req: SuggestPlanRequest) -> SuggestPlanResponse:
    items, debug = plan_suggestions(req)

    # Build phase-wise journey based on selected services
    journey = make_journey([i.model_dump() for i in items])  # if Pydantic v2

    # Return items + journey + debug
    return SuggestPlanResponse(
        count=len(items),
        items=items,
        journey=journey,   # <-- new field
        debug=debug,
    )
