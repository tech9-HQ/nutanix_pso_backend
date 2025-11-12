# app/main.py
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.middleware import install_middlewares
from app.logging_config import setup_logging
from app.routers import health, suggest, meta
from app.routers.generate_proposal import router as generate_proposal_router
from app.routers.generate_proposal_short import router as generate_proposal_short_router
from app.routers.routes_feedback import router as feedback_router

# REMOVE this line:
# from app.routers.auth import router as auth_router

# KEEP manual auth:
from app.routers import auth_manual

setup_logging()
app = FastAPI(
    title="Nutanix PSO Generator",
    description="AI-assisted proposal generation for Nutanix Professional Services",
    version="1.0.0",
)
install_middlewares(app)

# Routers
app.include_router(health.router)
# REMOVE this:
# app.include_router(auth_router)
app.include_router(auth_manual.router)          # only manual auth mounted
app.include_router(feedback_router)
app.include_router(meta.router)
app.include_router(suggest.router)
app.include_router(generate_proposal_router)
app.include_router(generate_proposal_short_router)

@app.get("/")
def root():
    return {"app":"nutanix_pso_generator","status":"running","version":"1.0.0","docs":"/docs"}
