# app/utils/timeit.py
from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Dict, Any

from docx.oxml import OxmlElement
from docx.oxml.ns import qn

@contextmanager
def timeit(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[timeit] {label}: {dt:.1f} ms")

def add_bookmark_to_paragraph(paragraph, bookmark_name: str, id_counter: Dict[str, Any]) -> None:
    """
    Insert a Word bookmark around the paragraph's first run.
    Expects id_counter like {'id': 1}. Increments it in-place.
    """
    if not bookmark_name:
        return

    # Allocate unique ID
    bid = int(id_counter.get("id", 1))
    id_counter["id"] = bid + 1

    # Ensure there is at least one run to anchor bookmark
    if not paragraph.runs:
        paragraph.add_run("")

    # Build bookmarkStart
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(bid))
    start.set(qn("w:name"), bookmark_name)

    # Build bookmarkEnd
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(bid))

    # Insert start before the first run, end after the last run
    first_run = paragraph.runs[0]._r
    last_run = paragraph.runs[-1]._r

    first_run.addprevious(start)
    last_run.addnext(end)

__all__ = ["timeit", "add_bookmark_to_paragraph"]
