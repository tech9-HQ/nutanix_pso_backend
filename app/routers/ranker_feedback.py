# app/routers/ranker_feedback.py
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services import ranker_feedback as svc

# Prefix "/ranker" groups feedback endpoints under ranking system.
router = APIRouter(prefix="/ranker", tags=["ranker-feedback"])


class FeedbackIn(BaseModel):
    """
    Incoming payload for ranking feedback.

    Notes:
    - user_id is provided directly by the client — this is insecure.
    - vote should ideally be validated to only {1, -1}.
    - service_id must match actual DB integer type, not arbitrary int.

    Consider replacing user_id with current authenticated user via dependency.
    """
    user_id: str = Field(...)
    service_id: int = Field(...)
    vote: int = Field(..., description="1 = like, -1 = dislike")


@router.post("/feedback")
async def submit_feedback(body: FeedbackIn):
    """
    Record like/dislike for a suggested service.

    SECURITY WARNING:
    - Endpoint currently trusts `user_id` provided by the client.
      A malicious caller can impersonate any user and corrupt rating data.
      FIX: replace with:
          user_id: str = Depends(get_current_user)
      so JWT determines user identity, not client input.

    Error Handling:
    - svc.record_feedback() should handle failures and raise HTTPException if needed.
      If it returns raw DB response, wrap it or define a response model.

    Idempotency:
    - If the same user votes twice for the same service, define behavior:
        * overwrite?
        * increment?
        * reject?
      svc.record_feedback should enforce consistent semantics.

    Return Format:
    - Currently returns whatever the service returns.
      Consider returning a structured success response:
          {"ok": True, "service_id": ..., "vote": ...}
    """
    return svc.record_feedback(body.user_id, body.service_id, body.vote)
