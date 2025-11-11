from __future__ import annotations
import os
from typing import List, Optional
from docx import Document
from fastapi import UploadFile

async def build_combined_requirements_text(text: Optional[str], files: List[UploadFile]) -> str:
    parts: List[str] = []
    if text and str(text).strip():
        parts.append(str(text).strip())
    for f in files or []:
        try:
            blob = await f.read()
            if not blob:
                continue
            name = (f.filename or "").lower()
            if name.endswith((".txt", ".md")):
                parts.append(blob.decode("utf-8", errors="ignore"))
        except Exception:
            # ignore file read errors in short mode
            pass
    return "\n\n".join(p for p in parts if p).strip()

def safe_save_doc(document: Document, filename: str) -> str:
    outdir = os.getenv("OUTPUT_DIR", "/tmp")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    document.save(path)
    return path
