from functools import lru_cache
from supabase import create_client, Client
from app.config import get_settings

@lru_cache()
def get_supabase() -> Client:
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_role)
