# app/routers/health.py
from fastapi import APIRouter

# Health router providing Kubernetes-style liveness & readiness endpoints.
# prefix="/health" keeps these endpoints standardized and discoverable by infra.
router = APIRouter(prefix="/health", tags=["health"])

@router.get("/live")
def live():
    """
    Liveness probe.
    Used by orchestrators (Kubernetes/ECS) to know if the process is alive.

    Current implementation always returns 200 — which is fine if:
      - the app crashes on unrecoverable failures,
      - you're not checking downstream dependencies here.

    DO NOT add external dependency checks here, otherwise Kubernetes
    may restart healthy pods during upstream outages.
    """
    return {"status": "ok"}


@router.get("/ready")
def ready():
    """
    Readiness probe.
    Indicates whether the application is ready to accept traffic.

    Recommended improvements for production:
    - Check critical dependencies such as:
        * database connectivity (Supabase or Postgres/PgBouncer ping)
        * cache availability (Redis)
        * model warmup completion (embeddings/LLM)
        * config validity (env var presence)
    - Cache dependency check results with a TTL to avoid heavy read checks.

    For now, it's fine returning 200, but this is where you implement
    real readiness logic.
    """
    return {"status": "ok"}
