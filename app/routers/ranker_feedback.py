# app/routers/ranker_feedback.py
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services import ranker_feedback as svc

router = APIRouter(prefix="/ranker", tags=["ranker-feedback"])


class FeedbackIn(BaseModel):
    user_id: str = Field(...)
    service_id: int = Field(...)
    vote: int = Field(..., description="1 = like, -1 = dislike")


@router.post("/feedback")
async def submit_feedback(body: FeedbackIn):
    """
    Record like/dislike for a suggested service.
    """
    return svc.record_feedback(body.user_id, body.service_id, body.vote)
