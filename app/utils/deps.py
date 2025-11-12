# app/utils/deps.py
from __future__ import annotations
import os
from typing import Optional
from supabase import create_client, Client  # pip install supabase

_SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
_SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
_SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

if not _SUPABASE_URL or not (_SUPABASE_ANON_KEY or _SUPABASE_SERVICE_ROLE_KEY):
    raise RuntimeError("Set SUPABASE_URL and keys in environment or .env")

# Read-only: anon key
_supabase_ro: Client = create_client(_SUPABASE_URL, _SUPABASE_ANON_KEY or _SUPABASE_SERVICE_ROLE_KEY)
# Read-write: prefer service-role
_supabase_rw: Client = create_client(_SUPABASE_URL, _SUPABASE_SERVICE_ROLE_KEY or _SUPABASE_ANON_KEY)

# Re-export
supabase_ro: Client = _supabase_ro
supabase_rw: Client = _supabase_rw
