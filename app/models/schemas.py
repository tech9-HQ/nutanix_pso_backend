# app/models/schemas.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

# ---------- Suggest / Service models ----------
#
# Notes:
# - This file mixes domain models used for input (requests), internal ranking objects,
#   and output (responses). That's fine for a small codebase but consider splitting:
#   schemas.requests, schemas.responses, schemas.domain for clarity as the app grows.
# - Consider adding model_config (e.g. orm_mode, validate_assignment) centrally if
#   you need consistent pydantic behavior across models.
#

class SuggestRequest(BaseModel):
    # Request for a small suggest/autocomplete endpoint.
    # `limit` default is small for quick responses.
    product_family: Optional[str] = None
    target_platforms: Optional[List[str]] = None
    limit: int = 6


class ServiceItem(BaseModel):
    # Lightweight representation of a service record (typically from DB).
    # Keep this model small and JSON-serializable — avoid embedding DB objects.
    id: int
    category_name: str
    service_name: str
    positioning: str

    # duration_days is an integer number of days (no fractions).
    duration_days: int

    # price_man_day allows float | str — this is a code smell:
    # - Money should use Decimal with a currency code to avoid float precision bugs.
    # - If string is required because some rows have "TBD" or "N/A", consider a
    #   dedicated union type or an explicit Optional[Decimal] + price_note field.
    price_man_day: float | str

    # canonical_names may be available for matching / fuzzy search.
    canonical_names: list[str] | None = None

    service_type: Optional[str] = None

    # If this field is required for logic, avoid None and use empty list default.
    supports_db_migration: bool

    # target platforms sometimes missing => allow None. Consider default_factory=list
    # to make usage simpler (no need to `or []` everywhere).
    target_platforms: list[str] | None = None

    # Scores defaulting to numeric 0 — defined as float | None and default 0 might
    # be confusing. If presence of score matters, prefer Optional[float] without 0.
    priority_score: float | None = 0
    popularity_score: float | None = 0

    product_family: str


class SuggestResponse(BaseModel):
    # Response wrapper used by the suggest endpoint. Keeps count and items.
    count: int
    items: list[ServiceItem]


class BOQItem(BaseModel):
    # Bill Of Quantities line item — simple and serializable.
    sku: Optional[str] = None
    description: str
    qty: float = 1
    vendor: Optional[str] = None
    notes: Optional[str] = None

    # Suggestion: validate qty > 0 with a pydantic validator or use condecimal/confloat.


# ---------- Constraints & Plan ----------

class Constraints(BaseModel):
    # This definition uses default_factory=list to ensure a new list instance per model.
    # That's preferred to avoid sharing mutable defaults between instances.
    target_platforms: List[str] = Field(default_factory=list)
    product_families: List[str] = Field(default_factory=list)
    prefer_migration_ready: bool = False
    must_include: List[str] = Field(default_factory=list)
    must_exclude: List[str] = Field(default_factory=list)

    # Consider:
    # - Using typing.Literal or Enum for product_family or other constrained values.
    # - Adding validators to ensure must_include/must_exclude disjointness if needed.


class SuggestPlanRequest(BaseModel):
    # Request body for the plan suggestion endpoint.
    client_name: str
    requirements_text: str

    # Boq defaults to an empty list. Good.
    boq: List[BOQItem] = Field(default_factory=list)
    selected_vendor: Optional[str] = None

    # Extra context used to tailor suggestions. These are free-text for now.
    industry: Optional[str] = None

    # `proposal_type` and `deployment_type` look like enumerations; prefer Literal or Enum:
    #   proposal_type: Literal["short", "detailed"] | None
    #   deployment_type: Literal["on prem", "hybrid", "cloud", "dark site"] | None
    proposal_type: Optional[str] = None         # "short" | "detailed"
    deployment_type: Optional[str] = None       # "on prem" | "hybrid" | "cloud" | "dark site"

    providers: List[str] = Field(default_factory=list)

    # Constraints is a nested model with sane defaults.
    constraints: Constraints = Field(default_factory=Constraints)

    limit: int = 8


class RankedService(BaseModel):
    # RankedService is likely an internal model created by scoring/ranking logic.
    # Keep it small and well-typed because it's part of the business logic.
    id: int
    service_name: str
    category_name: str
    product_family: str
    score: float
    reason: str   # short explanation why it was ranked (useful for debugging / UX)
    duration_days: int

    # price_man_day here is a float — unlike ServiceItem where it could be string.
    # Inconsistency: unify money representation across models.
    price_man_day: float

    service_type: Optional[str] = None
    supports_db_migration: bool

    # Use default_factory=list for lists to simplify downstream code.
    target_platforms: List[str] = Field(default_factory=list)
    canonical_names: List[str] = Field(default_factory=list)

    # Default numeric scores are explicit floats.
    popularity_score: float = 0.0
    priority_score: float = 0.0

    # Consider adding:
    # - `source` field (db vs cache vs ai) for auditability
    # - `confidence` field (0..1) if you blend ML scores with heuristics


# ---------- Journey models ----------

class JourneyPhase(BaseModel):
    # A phase is a collection of services grouped by timeline milestone.
    # `services` is List[Dict[str, Any]] — that's permissive but loses structure.
    # Consider replacing with a typed object (e.g. ServiceSummary) so consumers get
    # predictable fields. Use `Any` sparingly.
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
    # Response for the "suggest plan" endpoint: ranked items + optional journey and debug.
    count: int
    items: List[RankedService]
    journey: Optional[JourneyModel] = None

    # `debug` may contain arbitrary internals — ensure you strip or gate it in prod.
    debug: Optional[Dict[str, Any]] = None


# ---------- Proposal generation (used by generate_proposal endpoint) ----------

class ProposalRequest(BaseModel):
    # This model accepts both `client_name` (alias -> company_name) and `requirements_text`
    # (alias -> client_requirements). Using aliases keeps the UI/old API compatible.
    #
    # Note: aliasing means external JSON field `client_name` maps to `company_name` here.
    # Pydantic will populate by field name or alias because of model_config below.
    company_name: str = Field(..., alias="client_name")
    client_requirements: str = Field(..., alias="requirements_text")

    industry: Optional[str] = None
    deployment_type: Optional[str] = None
    proposal_type: Optional[str] = None  # "short" | "detailed"

    # Raw list of services as provided by the frontend — ideally these are typed.
    # Consider using List[RankedService] or a dedicated ServiceSelection schema to
    # validate expected fields (id, qty, chosen_options, discount, etc.).
    services: List[Dict[str, Any]] = Field(default_factory=list)

    # If True, generate the PDF at runtime (blocking) — be careful with request timeouts.
    runtime_pdf: bool = False

    # allow population by field name or alias
    model_config = ConfigDict(populate_by_name=True)

    # Suggested model-level improvements:
    # - model_config = ConfigDict(populate_by_name=True, extra='forbid', orm_mode=True, validate_assignment=True)
    #   - extra='forbid' prevents silent acceptance of unexpected fields (helpful for catching typos).
    #   - validate_assignment=True allows runtime assignment to be validated.
    #   - orm_mode=True if you want to parse ORM objects directly.
    #
    # - Add field-level validators for:
    #   - company_name / client_requirements length checks
    #   - services structure (non-empty, item ids exist)
    #   - if runtime_pdf True, extra checks (e.g., ensure templates available)
    #
    # - Standardize money type: use Decimal for price fields, with a currency field
    #   or a Money typed object. This avoids float rounding issues when summing costs.
