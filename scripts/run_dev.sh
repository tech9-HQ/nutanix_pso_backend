#!/usr/bin/env bash
export PYTHONPATH=.
uvicorn app.main:app --reload --port 8000
