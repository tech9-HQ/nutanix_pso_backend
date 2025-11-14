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

# ---------------------------
# Schemas
# ---------------------------

class LoginIn(BaseModel):
    # Using EmailStr gives automatic validation of email format.
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    # Consider adding `expires_at: datetime` so clients know token lifetime.


class MeOut(BaseModel):
    id: str
    email: str
    full_name: str | None = None
    company: str | None = None
    role: str | None = None
    created_at: datetime
    updated_at: datetime
    # Consider `orm_mode = True` in model_config if parsing ORM objects later.


# ---------------------------
# Helpers
# ---------------------------

def _jwt(user_id: str, email: str) -> str:
    """
    Create a JWT access token.

    Security notes & improvements:
    - Use a dedicated signing key manager (vault/secret manager) for `settings.jwt_secret`.
    - Consider issuing `jti` (unique token id) to allow revocation/blacklisting.
    - Consider adding `iss` (issuer) and `aud` (audience) claims for extra validation.
    - PyJWT accepts datetimes for iat/exp, but converting to int timestamps is sometimes clearer:
        int(now.timestamp())
      Be consistent across encode/decode.
    - Consider using short lived access tokens + refresh tokens for better security.
    """
    now = datetime.now(tz=timezone.utc)
    payload = {
        "user_id": user_id,
        "email": email,
        # `iat` and `exp` are necessary for expiration handling.
        "iat": now,
        "exp": now + timedelta(hours=settings.jwt_expiry_hours),
        # "jti": str(uuid4()),  # consider adding for revocation tracking
        # "iss": "tech9",       # optional issuer
        # "aud": "api"          # optional audience
    }
    # jwt.encode returns a str with PyJWT. Be explicit about algorithm in settings.
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


async def get_current_user(
    cred: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Dependency to extract user_id from bearer token.

    Improvements / hardening:
    - Use specific PyJWT exceptions (ExpiredSignatureError, InvalidTokenError) to set
      meaningful messages / status codes and optionally add `WWW-Authenticate` header.
    - Consider returning a richer user object (id + scopes/role) instead of just str.
    - Validate `aud`/`iss` if you set them in the token.
    - Consider checking token revocation store (redis/db) if you implement logout.
    """
    try:
        payload = jwt.decode(
            cred.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            # options={"require": ["exp", "iat"]}  # enforce presence if desired
        )
        # Validate shape explicitly to avoid KeyError surprises later
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return user_id
    except jwt.ExpiredSignatureError:
        # Explicit expiry error -> re-authenticate or use refresh token flow
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------
# Routes
# ---------------------------

@router.post("/login", response_model=TokenOut)
async def login(body: LoginIn):
    """
    Authenticate user and return JWT.

    Observations & suggestions:
    - `supabase_ro` suggests a read-only client; that's fine for auth reads.
    - Normalize email (strip/lower) to match `citext` columns — good.
    - Use parameterized/select to limit fields returned (you already do `.select("id,email,password_hash")`).
    - Wrap the DB call in try/except to handle connectivity/timeouts and return 503 or 500.
    - Use a constant-time comparison for password checks (passlib.bcrypt.verify is okay).
    - Implement account lockout / rate-limiting to reduce brute force risk.
    - Consider logging failed attempts (with rate limits) to detect attacks.
    """
    email = body.email.strip().lower()
    try:
        res = (
            supabase_ro.table("user_accounts")
            .select("id,email,password_hash")
            .eq("email", email)
            .single()
            .execute()
        )
    except Exception as exc:
        # Database / network error — don't leak internals to clients.
        # Consider instrumentation (Sentry / logs) here.
        raise HTTPException(status_code=503, detail="Authentication service unavailable") from exc

    # `res.data` may be None if not found; some supabase libs set `.data` or `.json()`.
    row = getattr(res, "data", None)
    # If user not found or password mismatch, return generic auth error
    if not row or not bcrypt.verify(body.password, row["password_hash"]):
        # Optional: increment failed-login counter here for lockout logic.
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # If you want to support MFA, check for MFA flag here and return an intermediate response.

    token = _jwt(row["id"], row["email"])
    # You may want to include expires_at in response so clients can show token lifetime.
    return TokenOut(access_token=token, user_id=row["id"], email=row["email"])


@router.get("/me", response_model=MeOut)
async def me(user_id: str = Depends(get_current_user)):
    """
    Return current user profile.

    Suggestions:
    - Consider returning a typed domain object rather than raw DB response to avoid leaking fields.
    - Validate timestamps are proper datetimes (Supabase sometimes returns strings).
    - Add caching if this endpoint is frequently called.
    """
    try:
        res = (
            supabase_ro.table("user_accounts")
            .select("id,email,full_name,company,role,created_at,updated_at")
            .eq("id", user_id)
            .single()
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="User service unavailable") from exc

    if not res.data:
        # This should be rare if tokens are issued for existing users.
        raise HTTPException(status_code=404, detail="User not found")
    # Consider mapping/resolving fields explicitly rather than returning `res.data` directly
    # to avoid accidental exposure of sensitive columns.
    return res.data
