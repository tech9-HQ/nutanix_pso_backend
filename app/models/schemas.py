# app/models/schemas.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SuggestRequest(BaseModel):
    product_family: Optional[str] = None
    target_platforms: Optional[List[str]] = None
    limit: int = 6

class ServiceItem(BaseModel):
    id: int
    category_name: str
    service_name: str
    positioning: str
    duration_days: int
    price_man_day: float | str
    canonical_names: list[str] | None = None
    service_type: Optional[str] = None
    supports_db_migration: bool
    target_platforms: list[str] | None = None
    priority_score: float | None = 0
    popularity_score: float | None = 0
    product_family: str

class SuggestResponse(BaseModel):
    count: int
    items: list[ServiceItem]

class BOQItem(BaseModel):
    sku: Optional[str] = None
    description: str
    qty: float = 1
    vendor: Optional[str] = None
    notes: Optional[str] = None

class Constraints(BaseModel):
    target_platforms: List[str] = Field(default_factory=list)
    product_families: List[str] = Field(default_factory=list)
    prefer_migration_ready: bool = False
    must_include: List[str] = Field(default_factory=list)
    must_exclude: List[str] = Field(default_factory=list)

class SuggestPlanRequest(BaseModel):
    client_name: str
    requirements_text: str
    boq: List[BOQItem] = Field(default_factory=list)
    selected_vendor: Optional[str] = None
    # extra context
    industry: Optional[str] = None
    proposal_type: Optional[str] = None         # "short" | "detailed"
    deployment_type: Optional[str] = None       # "on prem" | "hybrid" | "cloud" | "dark site"
    providers: List[str] = Field(default_factory=list)

    constraints: Constraints = Field(default_factory=Constraints)
    limit: int = 8

class RankedService(BaseModel):
    id: int
    service_name: str
    category_name: str
    product_family: str
    score: float
    reason: str
    duration_days: int
    price_man_day: float
    service_type: Optional[str] = None
    supports_db_migration: bool
    target_platforms: List[str] = Field(default_factory=list)
    canonical_names: List[str] = Field(default_factory=list)
    popularity_score: float = 0.0
    priority_score: float = 0.0

# ---------- Journey models ----------

class JourneyPhase(BaseModel):
    phase: str
    services: List[Dict[str, Any]]
    phase_days: int
    phase_cost_usd: float

class JourneyTotals(BaseModel):
    days: int
    cost_usd: float

class JourneyModel(BaseModel):
    phases: List[JourneyPhase]
    totals: JourneyTotals

# ---------- Responses ----------

class SuggestPlanResponse(BaseModel):
    count: int
    items: List[RankedService]
    journey: Optional[JourneyModel] = None   # added
    debug: Optional[Dict[str, Any]] = None
