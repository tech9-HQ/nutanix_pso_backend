# app/utils/text.py
import re
from typing import List

_word = re.compile(r"[A-Za-z0-9\-\_]+")

def tokenize(s: str) -> List[str]:
    if not s:
        return []
    return [w.lower() for w in _word.findall(s)]

def any_token_in(text: str, tokens: List[str]) -> bool:
    tl = text.lower()
    return any(t in tl for t in tokens)

_invalid_filename_chars = re.compile(r'[^A-Za-z0-9\-\_\(\)\[\]\s]')

def sanitize_filename(name: str) -> str:
    """
    Remove characters unsafe for filenames and collapse spaces.
    Example: "ACME Proposal (v1)" -> "ACME_Proposal_v1"
    """
    if not name:
        return "document"
    cleaned = _invalid_filename_chars.sub("", name)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned or "document"

def sanitize_bm_name(name: str, prefix: str = "bm") -> str:
    """
    Sanitize a string for use as a Word bookmark name.
    Keeps alphanumerics and underscores only, prepends prefix if needed.
    """
    if not name:
        return f"{prefix}_auto"
    n = re.sub(r"[^A-Za-z0-9_]", "_", name)
    n = re.sub(r"_+", "_", n).strip("_")
    if not n:
        n = "auto"
    return f"{prefix}_{n}"