# app/services/journey.py
from __future__ import annotations
from typing import List, Dict, Any
import re
from app.services.proposals_repo import PDF_CHUNKS

# -------------------------------------------------------------------
# Phase mapping logic (your existing code)
# -------------------------------------------------------------------
# Consider making this configurable (env / settings) if you expect different
# projects to use different phase names / orderings. Having it hard-coded is
# fine for a single product but reduces reusability.
_PHASE_ORDER = {
    "Kickoff / Assessment & Planning": 1,
    "Infrastructure Setup": 2,
    "Data Migration": 3,
    "Cutover & Go-Live": 4,
    "Post-Migration Optimization": 5,
}

def make_journey(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build phase view using the already-overwritten duration_days from ranker.

    Notes & expectations:
    - 'items' is expected to be a list of dict-like rows produced by the ranker,
      and to contain at least: id, service_name, product_family, duration_days,
      price_man_day, category_name, reason, service_type.
    - This function mutates no inputs and returns a serializable structure suitable
      for API responses or document generation.
    - Totals are computed as sums of per-service (rate * days) and days.
    - Be defensive about missing fields (we cast with int/float and default to 0).
    """

    def _phase_of(r: Dict[str, Any]) -> str:
        # Phase derivation is heuristic and uses multiple signals in priority:
        # 1) If 'reason' begins with a bracketed phase label (e.g., "[migration] ...")
        #    use that label (fast path).
        # 2) Fallback to category_name or service_name keywords.
        # 3) Default to 'Post-Migration Optimization' for everything else.
        #
        # Important: this logic is business-policy — changing keywords or
        # priority will change journey grouping. Consider centralizing mapping
        # or allowing ranking decisions to include an explicit 'phase' field.
        reason = r.get("reason", "")
        if reason.startswith("[") and "]" in reason:
            # extract between brackets: "[phase] reason..."
            return reason[1:reason.index("]")]
        cat = (r.get("category_name") or "").lower()
        # 'fitcheck' seems a domain-specific token — keep it if ranker uses it
        if "assessment" in cat or "fitcheck" in (r.get("service_name", "").lower()):
            return "Kickoff / Assessment & Planning"
        if "deployment" in cat:
            return "Infrastructure Setup"
        if "migration" in cat:
            return "Data Migration"
        # default catch-all
        return "Post-Migration Optimization"

    phases: Dict[str, Dict[str, Any]] = {}
    for it in items:
        # Use derived phase name and ensure bucket exists
        ph = _phase_of(it)
        bucket = phases.setdefault(
            ph, {"phase": ph, "services": [], "phase_days": 0, "phase_cost_usd": 0.0}
        )

        # Defensive coercion for numeric values — avoid exceptions if input shaped oddly.
        # Note: int(None) will raise; we therefore default to 0 for missing/invalid values.
        try:
            days = int(it.get("duration_days") or 0)
        except Exception:
            # Log at call-site if you want visibility; here we fallback silently.
            days = 0

        try:
            rate = float(it.get("price_man_day") or 0.0)
        except Exception:
            rate = 0.0

        # Append a compact service representation for downstream consumption.
        # Keep keys simple and stable: id, name, family, type, days, rate_usd_per_day, extended_usd, why
        bucket["services"].append(
            {
                "id": it["id"],
                "name": it["service_name"],
                "family": it["product_family"],
                "type": it.get("service_type"),
                "days": days,
                "rate_usd_per_day": rate,
                "extended_usd": rate * days,
                "why": it.get("reason", ""),
            }
        )

        # Aggregate totals per phase
        bucket["phase_days"] += days
        bucket["phase_cost_usd"] += rate * days

    # Order phases according to configured mapping; missing phases get high rank (99)
    ordered = sorted(phases.values(), key=lambda b: _PHASE_ORDER.get(b["phase"], 99))

    # Totals across phases — ensure consistent numeric types
    total_days = sum(p["phase_days"] for p in ordered)
    # rounding at presentation layer is better; here we round to 2 decimals for cost
    total_cost = round(sum(p["phase_cost_usd"] for p in ordered), 2)

    return {
        "phases": ordered,
        "totals": {"days": total_days, "cost_usd": total_cost},
    }

# -------------------------------------------------------------------
# Lightweight PDF chunk mapper used by generate_proposal
# -------------------------------------------------------------------
def map_kb_and_pdf_chunks_for_service(service_name: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Rank PDF_CHUNKS text snippets relevant to the given service_name.
    Returns a list of dicts: {'page': int, 'text': str, 'score': float}

    Implementation notes / caveats:
    - This is a very lightweight token-count scorer: counts occurrences of query tokens.
    - Pros: very fast, no external deps.
    - Cons: poor semantic understanding (misses synonyms, stemming, lemmatization),
      false positives if a token appears many times but not semantically relevant.
    - If the PDF_CHUNKS are large, this will iterate all chunks and could be slow;
      consider pre-indexing (inverted index or embeddings) for larger corpora.
    - Returned score is a simple integer count — consider normalizing by chunk length
      if you want to prefer dense matches over long documents with frequent tokens.
    """
    if not PDF_CHUNKS or not service_name:
        return []
    query = service_name.lower()
    # Tokenize on alphanumerics; this removes punctuation and splits camelCase poorly.
    # Consider using a more robust tokenizer (nlp library) if you need better matching.
    query_tokens = re.findall(r"[a-z0-9]+", query)
    ranked: List[Dict[str, Any]] = []

    # Iterate chunks and compute simple hit-count score
    for ch in PDF_CHUNKS:
        txt = (ch.get("text") or "").lower()
        if not txt:
            continue
        # Sum of counts for each token is simplest relevance heuristic
        score = sum(txt.count(tok) for tok in query_tokens)
        if score > 0:
            ranked.append(
                {"page": ch.get("page"), "text": ch.get("text"), "score": score}
            )

    # Sort descending by score. Ties preserve original order (stable sort).
    ranked.sort(key=lambda x: x["score"], reverse=True)

    # Ensure we always return at least 1 result when top_k >= 1 and matches exist.
    return ranked[: max(1, int(top_k or 1))]

# -------------------------------------------------------------------
# Explicit exports
# -------------------------------------------------------------------
__all__ = ["make_journey", "map_kb_and_pdf_chunks_for_service"]
