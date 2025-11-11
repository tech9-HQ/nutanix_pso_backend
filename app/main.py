# app/main.py
from dotenv import load_dotenv
load_dotenv()  

from fastapi import FastAPI
from app.middleware import install_middlewares
from app.logging_config import setup_logging
from app.routers import health, suggest 
from app.routers.generate_proposal import router as generate_proposal_router
from app.routers.generate_proposal_short import router as generate_proposal_short_router
from app.routers import meta, suggest

setup_logging()

app = FastAPI(title="nutanix_pso_generator")
install_middlewares(app)
app.include_router(meta.router)
app.include_router(suggest.router)
app.include_router(health.router)
app.include_router(suggest.router)
app.include_router(generate_proposal_router)
app.include_router(generate_proposal_short_router)

@app.get("/")
def root():
    return {"app": "nutanix_pso_generator", "status": "running"}
