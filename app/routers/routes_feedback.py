from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from app.utils import supabase_ro, supabase_rw
from app.routers.auth import get_current_user

# Feedback routes grouped under /feedback
router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackIn(BaseModel):
    """
    Incoming user feedback payload.

    Notes:
    - service_id must be validated against an existing service in the DB
      (currently not checked — consider adding validation).
    - `context` is loosely typed (dict). If this is stored as JSONB, this is fine,
      but consider using `dict[str, Any]` or a typed model for stronger guarantees.
    """
    service_id: int
    relevant: bool
    context: dict | None = None


@router.post("/relevance")
async def feedback_relevance(
    body: FeedbackIn,
    user_id: str = Depends(get_current_user),  # Auth token determines user identity
):
    """
    Record whether a suggested service was relevant to the user.

    Critical notes:
    - `supabase_ro` is a READ-ONLY client. You cannot rely on it for INSERT/UPSERT operations.
      If this works today, it's because your anon client has elevated privileges.
      FIX: use supabase_rw (service role) for inserts.
    - There is no error handling. Any database failure will result in a 200 response.
      Wrap calls in try/except and return HTTPException on failure.
    - Consider making this endpoint idempotent: what happens if user submits feedback twice?
    """

    # ---- insert raw feedback ----
    try:
        # MUST use supabase_rw (service-role) to bypass RLS safely.
        supabase_rw.table("user_feedback").insert({
            "user_id": user_id,
            "service_id": body.service_id,
            "relevant": body.relevant,
            "context": body.context
        }).execute()
    except Exception as e:
        # Log error but avoid leaking internals to the client
        # raise HTTPException(status_code=500, detail="Failed to record feedback")
        raise

    # ---- update aggregated stats via RPC ----
    # Important: RPC should be executed on RW client as well.
    try:
        supabase_rw.rpc("upsert_user_service_stats", {
            "p_user_id": user_id,
            "p_service_id": body.service_id,
            "p_pos": 1 if body.relevant else 0,
            "p_neg": 0 if body.relevant else 1
        }).execute()
    except Exception:
        # If aggregation fails, consider compensating logs or retry logic.
        # Do NOT silently swallow failures.
        # raise HTTPException(status_code=500, detail="Failed to update feedback stats")
        raise

    return {"ok": True, "message": "Feedback recorded"}
