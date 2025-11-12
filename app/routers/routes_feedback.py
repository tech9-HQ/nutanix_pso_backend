from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.utils import supabase_ro, supabase_rw
from app.routers.auth import get_current_user

router = APIRouter(prefix="/feedback", tags=["feedback"])

class FeedbackIn(BaseModel):
    service_id: int
    relevant: bool
    context: dict | None = None

@router.post("/relevance")
async def feedback_relevance(
    body: FeedbackIn,
    user_id: str = Depends(get_current_user)  # Get from auth token
):
    # Insert feedback
    supabase_ro.table("user_feedback").insert({
        "user_id": user_id,
        "service_id": body.service_id,
        "relevant": body.relevant,
        "context": body.context
    }).execute()

    # Update aggregated stats
    supabase_ro.rpc("upsert_user_service_stats", {
        "p_user_id": user_id,
        "p_service_id": body.service_id,
        "p_pos": 1 if body.relevant else 0,
        "p_neg": 0 if body.relevant else 1
    }).execute()

    return {"ok": True, "message": "Feedback recorded"}