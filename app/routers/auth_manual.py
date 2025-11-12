# app/routers/auth_manual.py
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta, timezone
import jwt
from passlib.hash import bcrypt
from app.utils.deps import supabase_ro
from app.config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()

# ----- Schemas -----
class LoginIn(BaseModel):
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str

class MeOut(BaseModel):
    id: str
    email: str
    full_name: str | None = None
    company: str | None = None
    role: str | None = None
    created_at: datetime
    updated_at: datetime

# ----- Helpers -----
def _jwt(user_id: str, email: str) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "user_id": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(hours=settings.jwt_expiry_hours),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

async def get_current_user(
    cred: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    try:
        payload = jwt.decode(cred.credentials, settings.jwt_secret,
                             algorithms=[settings.jwt_algorithm])
        return payload["user_id"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ----- Routes -----
@router.post("/login", response_model=TokenOut)
async def login(body: LoginIn):
    # email is citext in DB, but keep client input normalized
    email = body.email.strip().lower()
    res = (
        supabase_ro.table("user_accounts")
        .select("id,email,password_hash")
        .eq("email", email)
        .single()
        .execute()
    )
    row = getattr(res, "data", None)
    if not row or not bcrypt.verify(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = _jwt(row["id"], row["email"])
    return TokenOut(access_token=token, user_id=row["id"], email=row["email"])

@router.get("/me", response_model=MeOut)
async def me(user_id: str = Depends(get_current_user)):
    res = (
        supabase_ro.table("user_accounts")
        .select("id,email,full_name,company,role,created_at,updated_at")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="User not found")
    return res.data
