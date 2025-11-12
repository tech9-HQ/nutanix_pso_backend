# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta, timezone
import jwt
import logging

from app.utils.deps import supabase_ro, supabase_rw  # anon client with RLS
from app.config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()
log = logging.getLogger("auth")

# =========================
# Schemas
# =========================

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserSignup(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None
    company: str | None = None
    role: str | None = None

class UserToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str

class UserProfile(BaseModel):
    id: str
    email: str
    full_name: str | None = None
    company: str | None = None
    role: str | None = None
    department: str | None = None
    created_at: datetime

# =========================
# Helpers
# =========================

def _create_jwt_token(user_id: str, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(tz=timezone.utc) + timedelta(hours=settings.jwt_expiry_hours),
        "iat": datetime.now(tz=timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
) -> str | None:
    if not credentials:
        return None
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload.get("user_id")
    except Exception:
        return None

# =========================
# Routes
# =========================

@router.post("/signup", response_model=UserToken)
async def signup(credentials: UserSignup):
    email = credentials.email.strip().lower()
    try:
        # 1) Create auth user (email provider must be enabled)
        auth_resp = supabase_ro.auth.sign_up({
            "email": email,
            "password": credentials.password,
            "options": {"data": {"full_name": credentials.full_name or ""}},
        })
        user = getattr(auth_resp, "user", None)
        if not user:
            raise HTTPException(status_code=400, detail="Signup failed")
        user_id = user.id

        # 2) Upsert profile with SERVICE ROLE (bypasses RLS)
        supabase_rw.table("user_profiles").upsert({
            "id": user_id,
            "email": email,
            "full_name": credentials.full_name or "",
            "company": credentials.company or None,
            "role": credentials.role or None,
        }, on_conflict="id").execute()

        # 3) Return first-party JWT
        token = _create_jwt_token(user_id, email)
        log.info(f"New user registered: {email} ({user_id})")
        return UserToken(access_token=token, user_id=user_id, email=email)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Signup failed: {e}")
        raise HTTPException(status_code=400, detail=f"Signup failed: {e}")

@router.post("/login", response_model=UserToken)
async def login(credentials: UserLogin):
    """
    Password login against Supabase Auth.
    Returns first-party JWT for API auth.
    """
    email = credentials.email.strip().lower()
    try:
        auth_resp = supabase_ro.auth.sign_in_with_password({
            "email": email,
            "password": credentials.password,
        })
        if not getattr(auth_resp, "user", None):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        user_id = auth_resp.user.id
        token = _create_jwt_token(user_id, email)
        log.info(f"User logged in: {email} ({user_id})")
        return UserToken(access_token=token, user_id=user_id, email=email)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid email or password")

@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(user_id: str = Depends(get_current_user)):
    """
    Fetch profile for current user.
    """
    try:
        res = (
            supabase_ro.table("user_profiles")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="User profile not found")
        return res.data
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/auth/me failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch user profile")

@router.put("/me")
async def update_profile(
    update_data: dict,
    user_id: str = Depends(get_current_user),
):
    """
    Update allowed fields on current user's profile.
    """
    allowed = {"full_name", "company", "role", "department"}
    payload = {k: v for k, v in (update_data or {}).items() if k in allowed and v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    try:
        res = (
            supabase_ro.table("user_profiles")
            .update(payload)
            .eq("id", user_id)
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="User profile not found")
        return {"ok": True, "updated": payload, "profile": res.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Update profile failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user profile")
