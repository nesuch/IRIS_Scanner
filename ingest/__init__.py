"""Deterministic PDF -> clauses segmentation engine (ported from IRIS Segmenter).

No LLM: segmentation is driven entirely by a saved JSON spec per document type.
extract (pdfplumber) -> preprocess -> segment -> validate. Output rows are
{id, clause, tag} which map directly onto IRIS's regulatory_clauses.
"""
import json
import os
import re

from .extract import extract
from .preprocess import explode_layout_tables, explode_stage_matrix
from .segment import segment
from .build import validate

SPECS_DIR = os.path.join(os.path.dirname(__file__), "specs")


def list_specs():
    """Available document specs (id + friendly label)."""
    out = []
    for fn in sorted(os.listdir(SPECS_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(SPECS_DIR, fn), encoding="utf-8") as fh:
                d = json.load(fh)
            out.append({"id": fn[:-5], "doc_id": d.get("doc_id", fn[:-5]),
                        "sheet_name": d.get("sheet_name", "")})
        except Exception:
            continue
    return out


def load_spec(spec_id):
    p = os.path.join(SPECS_DIR, os.path.basename(spec_id) + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _filter_ignore(blocks, dropped, spec):
    """Apply a spec's ignore_patterns to an already-extracted block list (so we
    don't have to re-run pdfplumber per spec). Lines that match move to dropped."""
    import re
    pats = [re.compile(p) for p in spec.get("ignore_patterns", []) if p]
    if not pats:
        return blocks, dropped
    kept, dl = [], list(dropped)
    for b in blocks:
        t = b.get("text", "") if b.get("kind") == "line" else ""
        if t and any(p.search(t) for p in pats):
            dl.append({"reason": "running_header"})
        else:
            kept.append(b)
    return kept, dl


# Words that are common in the BODY of nearly every IRDAI regulation (so they're
# useless for telling documents apart), even though they're rare across spec TITLES.
# The title-DF filter can't catch these; this list does.
_TITLE_COMMON = {
    "irdai", "irda", "insurance", "insurer", "insurers", "regulatory", "development",
    "authority", "india", "indian", "regulations", "regulation", "notification",
    "gazette", "extraordinary", "hyderabad", "company", "companies", "operations",
    "operation", "matters", "allied", "services", "service", "provider", "providers",
    "information", "sharing", "regulated", "entities", "entity", "including",
    "general", "board", "policy", "policies", "policyholder", "policyholders",
    "business", "requirements", "provisions", "conditions", "purpose", "framework",
    "chapter", "schedule", "section", "definitions", "applicability", "commencement",
    "persons", "person", "certificate", "application", "shall", "these", "under",
}


def _spec_title_tokens(spec):
    """>=4-char alpha words from a spec's human title (NOT the doc_id, whose code
    fragments like 'corp'/'info'/'regs' are noise)."""
    return set(re.findall(r"[a-z]{4,}", spec.get("sheet_name", "").lower()))


def _title_document_freq():
    """How many specs each title word appears in. A word in many titles (e.g.
    'insurance', 'regulations') is generic; one in a couple ('lloyd', 'sugam') is
    distinctive to its document — that's the signal we match against the PDF."""
    from collections import Counter
    cnt = Counter()
    for s in list_specs():
        sp = load_spec(s["id"])
        if sp:
            for w in _spec_title_tokens(sp):
                cnt[w] += 1
    return cnt


def detect_spec(pdf_path, limit=5):
    """Best-match spec for a PDF — deterministic and fast: extract ONCE, then for
    each spec apply its ignore_patterns as a post-filter and segment. The truly
    fitting spec leaves the fewest in-scope lines unassigned (orphans)."""
    base_blocks, base_dropped = extract(pdf_path)
    # A few gazettes set their whole English half below the default small-type
    # threshold, so the shared extraction throws the entire document away as
    # footnotes and EVERY spec scores zero clauses — nothing can be detected. Retry
    # once with the size filter off when that happens. Deliberately re-extracted for
    # everyone rather than per-spec: the threshold is a property of the DOCUMENT's
    # typesetting, and giving one spec a better view of the page than its rivals
    # would let it win documents that aren't its own.
    # Fires on a large FRACTION, not only on near-total loss. A re-OCR'd scan gets
    # its text-layer size from the glyph heights tesseract measured, which can land
    # just under the threshold for much of a document: the Corporate Agents scan lost
    # 177 of ~380 English lines that way — not everything, but enough that its own
    # spec did not place in the top three for its own document.
    _fn = sum(1 for d in base_dropped if d.get("reason") == "footnote")
    if _fn >= 10 and _fn >= 0.3 * (len(base_blocks) + _fn):
        base_blocks, base_dropped = extract(pdf_path, small=0)
    # Whole-document text (lower-cased) for a TITLE tie-breaker. Many IRDAI
    # regulations share the exact same structure (CHAPTER I..N + numbered clauses),
    # so structurally-identical specs segment them identically and tie on
    # orphans/clauses. The distinctive words in a spec's title (e.g. "lloyd",
    # "sugam", "aggregator") only appear in ITS document, so a small bonus for those
    # breaks the tie toward the right spec without overriding a genuine structural
    # fit (orphan/dupe penalties stay far larger).
    # Word FREQUENCIES in the document (exact whole words, so "corp" doesn't hit
    # "incorporated"). Frequency matters: a document's actual SUBJECT words (e.g.
    # "aggregator" in the web-aggregator regs) appear many times, while a coincidental
    # mention of another reg's topic appears once or twice — so we weight a spec's
    # distinctive title words by how often they occur here, not just whether they do.
    from collections import Counter as _Counter
    _doc_freq = _Counter(re.findall(r"[a-z]{4,}",
                                    " ".join((b.get("text") or "") for b in base_blocks).lower()))
    _title_df = _title_document_freq()
    # How many spec titles a word may appear in and still count as distinctive.
    # This was a hard "<= 2", tuned when the library held 31 specs, and it broke
    # silently as the library grew: adding the IAC (Meetings) 2000 spec put
    # "advisory" and "committee" into a THIRD title, which zeroed the entire title
    # bonus for the Re-insurance Advisory Committee regs on their own document and
    # handed detection to an unrelated spec. Scaling with the library keeps "in a
    # handful of titles" meaning the same thing as specs are added.
    _all_specs = list_specs()
    _df_max = max(2, round(len(_all_specs) * 0.08))
    ranked = []
    for s in _all_specs:
        spec = load_spec(s["id"])
        if not spec:
            continue
        try:
            blocks, dropped = _filter_ignore(base_blocks, base_dropped, spec)
            for step in spec.get("preprocess", []):
                if step == "explode_layout_tables":
                    blocks = explode_layout_tables(blocks)
                elif step == "explode_stage_matrix":
                    blocks = explode_stage_matrix(blocks)
            rows, inscope, excluded, assigned, spec_errors, warnings = segment(blocks, spec)
            report = validate(blocks, dropped, inscope, excluded, assigned, rows)
        except Exception:
            continue
        clauses = len(rows)
        if clauses == 0:
            continue
        orphans = report.get("orphan_lines", 0)
        dupes = len(warnings)
        # a good fit leaves few orphans AND few id collisions
        score = (0 if spec_errors else 1_000_000) - orphans * 100 - dupes * 60
        # TITLE match is the primary discriminator among structurally-clean specs.
        # Many IRDAI regulations share the identical structure (CHAPTER I..N +
        # numbered clauses), so a generic spec can segment the WRONG document cleanly
        # — even over-segmenting it into MORE clauses than the right spec. So the
        # clause bonus is capped LOW (over-segmentation can't win), and the DISTINCTIVE
        # title words (present in few spec titles, e.g. "lloyd"/"sugam"/"aggregator")
        # that also appear in the PDF drive the score. Weighted above the clause cap
        # but below the per-orphan penalty, so a bad structural fit still loses.
        distinct = [w for w in _spec_title_tokens(spec)
                    if _title_df.get(w, 0) <= _df_max and w not in _TITLE_COMMON]
        title_score = sum(min(_doc_freq.get(w, 0), 12) for w in distinct)
        score += title_score * 15 + min(clauses, 25)
        ranked.append({"spec_id": s["id"], "doc_id": s["doc_id"], "score": score,
                       "clauses": clauses, "orphans": orphans, "duplicates": dupes,
                       "accounted": bool(report.get("fully_accounted")), "errors": bool(spec_errors)})
    ranked.sort(key=lambda x: -x["score"])
    return ranked[:limit]


def segment_pdf(pdf_path, spec):
    """Run the deterministic pipeline. Returns (rows, report, spec_errors) where
    each row is {id, clause, tag}."""
    blocks, dropped = extract(pdf_path, ignore_patterns=spec.get("ignore_patterns"),
                              small=spec.get("small_type"))
    for step in spec.get("preprocess", []):
        if step == "explode_layout_tables":
            blocks = explode_layout_tables(blocks)
        elif step == "explode_stage_matrix":
            blocks = explode_stage_matrix(blocks)
    rows, inscope, excluded, assigned, spec_errors, warnings = segment(blocks, spec)
    report = validate(blocks, dropped, inscope, excluded, assigned, rows)
    report["duplicate_ids"] = warnings   # surfaced, not hidden — review signal
    return rows, report, spec_errors
