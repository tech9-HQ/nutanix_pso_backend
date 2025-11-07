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
