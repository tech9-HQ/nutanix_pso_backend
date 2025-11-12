# app/routers/analytics.py
from fastapi import APIRouter, Depends
from app.routers.auth import get_current_user
from app.utils.deps import supabase_ro

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/user/feedback-summary")
async def get_user_feedback_summary(user_id: str = Depends(get_current_user)):
    """Get user's feedback history and impact"""
    
    # Get feedback count
    stats = supabase_ro.table("user_service_stats")\
        .select("*")\
        .eq("user_id", user_id)\
        .execute()
    
    total_pos = sum(s["pos"] for s in stats.data)
    total_neg = sum(s["neg"] for s in stats.data)
    
    return {
        "total_feedback": total_pos + total_neg,
        "positive": total_pos,
        "negative": total_neg,
        "services_rated": len(stats.data)
    }