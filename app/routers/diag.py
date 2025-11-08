# app/main.py
from dotenv import load_dotenv
load_dotenv()  # must be first so all downstream imports see env

from fastapi import FastAPI
from app.middleware import install_middlewares
from app.logging_config import setup_logging
from app.routers import health, suggest

setup_logging()

app = FastAPI(title="nutanix_pso_generator")
install_middlewares(app)

# routers
app.include_router(health.router)
app.include_router(suggest.router)

@app.get("/")
def root():
    return {"app": "nutanix_pso_generator", "status": "running"}
