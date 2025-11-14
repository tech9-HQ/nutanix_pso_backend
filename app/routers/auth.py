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
    # Suggestion: Add password policy validation (min length, complexity) via a validator.


class UserToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    # Consider adding `expires_at: datetime` so client knows when token expires.


class UserProfile(BaseModel):
    id: str
    email: str
    full_name: str | None = None
    company: str | None = None
    role: str | None = None
    department: str | None = None
    created_at: datetime
    # Consider model_config = ConfigDict(orm_mode=True) if you'll parse ORM objects later.


# =========================
# Helpers
# =========================

def _create_jwt_token(user_id: str, email: str) -> str:
    # NOTE: calling datetime.now() twice (for exp and iat) can produce slightly different values.
    # Better to capture `now` once and reuse it so iat and exp are consistent.
    payload = {
        "user_id": user_id,
        "email": email,
        # Use a single "now" for consistent iat/exp values:
        # now = datetime.now(tz=timezone.utc)
        # "exp": now + timedelta(hours=settings.jwt_expiry_hours),
        # "iat": now,
        "exp": datetime.now(tz=timezone.utc) + timedelta(hours=settings.jwt_expiry_hours),
        "iat": datetime.now(tz=timezone.utc),
        # Consider adding claims: "iss", "aud", "jti" for revocation, and `scope` or `roles`.
    }
    # Security: ensure `settings.jwt_secret` is rotated securely (vault) and not logged.
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Validate bearer token and return user_id.

    Improvements:
    - Catch specific jwt exceptions to provide clearer error messages (ExpiredSignatureError, InvalidTokenError).
    - Validate `aud`/`iss` if you include them in tokens.
    - Consider returning a richer user object (id + roles/claims) instead of raw user_id.
    - Consider checking a token revocation store (redis/db) keyed by `jti` if you support logout/revocation.
    """
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        # Specific expired message helps clients know to use refresh tokens.
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
) -> str | None:
    """
    Optional token dependency that returns user_id or None.
    Useful for endpoints that support both anonymous and authenticated access.

    Notes:
    - This swallows any decode errors and returns None; that's convenient but can hide token issues.
    - If you want to surface token problems (for telemetry), log them at debug level.
    """
    if not credentials:
        return None
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload.get("user_id")
    except Exception:
        # Intentionally swallow exceptions (treat as anonymous)
        return None


# =========================
# Routes
# =========================

@router.post("/signup", response_model=UserToken)
async def signup(credentials: UserSignup):
    email = credentials.email.strip().lower()
    try:
        # 1) Create auth user (email provider must be enabled)
        # WARNING: Using `supabase_ro` for sign_up is surprising — variable name implies read-only.
        #   Ensure `supabase_ro.auth.sign_up` is actually allowed (auth API may be on a different client).
        #   If you have a service-role client, avoid using it in request handlers to prevent accidental misuse.
        auth_resp = supabase_ro.auth.sign_up({
            "email": email,
            "password": credentials.password,
            "options": {"data": {"full_name": credentials.full_name or ""}},
        })
        user = getattr(auth_resp, "user", None)
        if not user:
            # Keep response generic to avoid leaking provider internals.
            raise HTTPException(status_code=400, detail="Signup failed")
        user_id = user.id

        # 2) Upsert profile with SERVICE ROLE (bypasses RLS)
        # Make sure supabase_rw is the service-role client and is secured in server env.
        # Race consideration: if the upsert fails after sign_up succeeded, you have an orphan auth user.
        #   - Option: wrap in retry + compensating cleanup (delete auth user) or mark profile creation as async job.
        supabase_rw.table("user_profiles").upsert({
            "id": user_id,
            "email": email,
            "full_name": credentials.full_name or "",
            "company": credentials.company or None,
            "role": credentials.role or None,
        }, on_conflict="id").execute()

        # 3) Return first-party JWT
        # Consider adding `jti` to the token and storing it if you want revocation support.
        token = _create_jwt_token(user_id, email)
        log.info(f"New user registered: {email} ({user_id})")
        return UserToken(access_token=token, user_id=user_id, email=email)

    except HTTPException:
        # Re-raise already-constructed HTTPExceptions (keeps intended status code)
        raise
    except Exception as e:
        # Don't include raw exception messages in 400 responses in production (can leak info).
        # Log the real error (stack) at error level and return a sanitized message to the client.
        log.exception("Signup failed")
        raise HTTPException(status_code=400, detail="Signup failed")


@router.post("/login", response_model=UserToken)
async def login(credentials: UserLogin):
    """
    Password login against Supabase Auth.
    Returns first-party JWT for API auth.
    """
    email = credentials.email.strip().lower()
    try:
        # Using supabase_ro.auth.sign_in_with_password is fine; ensure this call behaves synchronously/async as expected.
        auth_resp = supabase_ro.auth.sign_in_with_password({
            "email": email,
            "password": credentials.password,
        })
        if not getattr(auth_resp, "user", None):
            # Generic message to avoid username enumeration
            raise HTTPException(status_code=401, detail="Invalid email or password")

        user_id = auth_resp.user.id
        token = _create_jwt_token(user_id, email)
        log.info(f"User logged in: {email} ({user_id})")
        # Consider returning token expiry metadata and optionally refresh token here.
        return UserToken(access_token=token, user_id=user_id, email=email)

    except HTTPException:
        raise
    except Exception:
        # Avoid leaking internal errors — return the same 401 for auth problems to prevent info leakage.
        log.exception("Login failed")
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
        # Consistency concern:
        # - Supabase client libraries may return `res.data` as an object or list depending on query type.
        # - Here you expect a single object; validate shape before returning.
        if not res.data:
            raise HTTPException(status_code=404, detail="User profile not found")
        # Consider mapping/resolving to UserProfile explicitly to avoid exposing sensitive fields.
        return res.data
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/auth/me failed: {e}")
        # Use 500 when external service fails; include instrumentation for debugging.
        raise HTTPException(status_code=500, detail="Failed to fetch user profile")


@router.put("/me")
async def update_profile(
    update_data: dict,
    user_id: str = Depends(get_current_user),
):
    """
    Update allowed fields on current user's profile.

    Notes:
    - Using `dict` for request body is permissive. Better to define a Pydantic schema
      (e.g., `UpdateProfile`) that whitelists fields and validates types.
    - Also use response_model to define what you return. Right now response shape is ad-hoc.
    """
    allowed = {"full_name", "company", "role", "department"}
    # Filter payload for allowed keys and non-None values (prevents null overwrites).
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
        # Supabase update often returns a list of updated rows; check shape.
        if not res.data:
            raise HTTPException(status_code=404, detail="User profile not found")
        # Return a consistent response: updated profile should be a single object.
        # If `res.data` is a list, pick first element.
        profile = res.data[0] if isinstance(res.data, list) else res.data
        return {"ok": True, "updated": payload, "profile": profile}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Update profile failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user profile")
