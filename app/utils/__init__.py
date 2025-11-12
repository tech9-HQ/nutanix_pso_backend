# app/utils/__init__.py
from __future__ import annotations
import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY are required")

# NOTE: Using basic constructor because your installed supabase-py
# does not accept http_client/options kwargs.
supabase_ro: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
supabase_rw: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY,
)
