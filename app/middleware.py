# app/middleware.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

def install_middlewares(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400,
    )
