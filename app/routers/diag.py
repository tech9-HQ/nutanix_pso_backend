# app/main.py

# Load .env very early so all imports downstream see environment variables.
# This is important because app.config, database clients, and logging config
# may read env vars at import time.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.middleware import install_middlewares
from app.logging_config import setup_logging
from app.routers import health, suggest

# Initialize logging before anything else that may log.
# Consider:
#   - Using structured logging (JSON) in production.
#   - Sending logs to APM (Datadog, ELK, CloudWatch).
setup_logging()

# FastAPI application instance creation.
# You may want to configure:
#   - docs_url=None, redoc_url=None in production to hide docs from external users.
#   - version="1.0.0" for API versioning.
app = FastAPI(title="nutanix_pso_generator")

# Install middlewares globally.
# This function likely sets:
#   - CORS
#   - Request/response logging
#   - GZip
#   - Security headers
# Validate that middlewares are added in the correct order.
install_middlewares(app)

# Register routers.
# Good: routers are modularized (health, suggest).
# Missing routers: auth, analytics, proposal, etc.
# Consider grouping routers under an APIRouter prefix like /api/v1 for long-term versioning.
app.include_router(health.router)
app.include_router(suggest.router)

# Health root.
# Minimal root endpoint for uptime check or banners.
# Consider adding:
#   - version
#   - environment
#   - build SHA / commit hash (useful for debugging deployments)
@app.get("/")
def root():
    return {"app": "nutanix_pso_generator", "status": "running"}
