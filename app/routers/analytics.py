# app/routers/analytics.py
from fastapi import APIRouter, Depends
from app.routers.auth import get_current_user
from app.utils.deps import supabase_ro

# Router for analytics-related endpoints.
# prefix="/analytics" keeps analytics logically separated and allows clean versioning
# later (e.g., /analytics/v2/...).
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/user/feedback-summary")
async def get_user_feedback_summary(user_id: str = Depends(get_current_user)):
    """
    Returns an aggregate summary of the user's feedback history.

    Notes:
    - `get_current_user` must return a user identifier. Ensure it returns the *actual*
      user_id, not a user object (depends on your auth implementation).
    - Endpoint fetches from a read-only Supabase dependency (`supabase_ro`), 
      which is good practice for analytics.
    - Currently returns simple aggregates; consider caching if usage grows.
    """

    # Query the analytics table for all feedback records associated with the user.
    # Using .select("*") may fetch unnecessary fields; for large tables you may want:
    # .select("pos, neg") instead for better efficiency.
    stats = (
        supabase_ro
        .table("user_service_stats")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    # Aggregate counts. Ensure that pos/neg exist and default to 0.
    # If Supabase ever returns None values, this will throw — consider safe extraction:
    # s.get("pos") or 0
    total_pos = sum(s["pos"] for s in stats.data)
    total_neg = sum(s["neg"] for s in stats.data)

    # Provide a compact summary payload for the frontend.
    return {
        "total_feedback": total_pos + total_neg,
        "positive": total_pos,
        "negative": total_neg,
        "services_rated": len(stats.data),
    }

    # Possible enhancements:
    # - Add error handling for Supabase failures (network issues, invalid schema).
    # - Add security: ensure users cannot query other users' records (validate get_current_user).
    # - Add pagination or date filters if data volume grows.
    # - Add analytics dimensions (service breakdown, sentiment trends, etc.).
    # - Implement caching (Redis) if this endpoint is called frequently.
