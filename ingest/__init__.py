"""Deterministic PDF -> clauses segmentation engine (ported from IRIS Segmenter).

No LLM: segmentation is driven entirely by a saved JSON spec per document type.
extract (pdfplumber) -> preprocess -> segment -> validate. Output rows are
{id, clause, tag} which map directly onto IRIS's regulatory_clauses.
"""
import json
import os

from .extract import extract
from .preprocess import explode_layout_tables
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


def detect_spec(pdf_path, limit=5):
    """Best-match spec for a PDF — deterministic and fast: extract ONCE, then for
    each spec apply its ignore_patterns as a post-filter and segment. The truly
    fitting spec leaves the fewest in-scope lines unassigned (orphans)."""
    base_blocks, base_dropped = extract(pdf_path)
    ranked = []
    for s in list_specs():
        spec = load_spec(s["id"])
        if not spec:
            continue
        try:
            blocks, dropped = _filter_ignore(base_blocks, base_dropped, spec)
            for step in spec.get("preprocess", []):
                if step == "explode_layout_tables":
                    blocks = explode_layout_tables(blocks)
            rows, inscope, excluded, assigned, spec_errors = segment(blocks, spec)
            report = validate(blocks, dropped, inscope, excluded, assigned, rows)
        except Exception:
            continue
        clauses = len(rows)
        if clauses == 0:
            continue
        orphans = report.get("orphan_lines", 0)
        score = (0 if spec_errors else 1_000_000) - orphans * 100 + min(clauses, 300)
        ranked.append({"spec_id": s["id"], "doc_id": s["doc_id"], "score": score,
                       "clauses": clauses, "orphans": orphans,
                       "accounted": bool(report.get("fully_accounted")), "errors": bool(spec_errors)})
    ranked.sort(key=lambda x: -x["score"])
    return ranked[:limit]


def segment_pdf(pdf_path, spec):
    """Run the deterministic pipeline. Returns (rows, report, spec_errors) where
    each row is {id, clause, tag}."""
    blocks, dropped = extract(pdf_path, ignore_patterns=spec.get("ignore_patterns"))
    for step in spec.get("preprocess", []):
        if step == "explode_layout_tables":
            blocks = explode_layout_tables(blocks)
    rows, inscope, excluded, assigned, spec_errors = segment(blocks, spec)
    report = validate(blocks, dropped, inscope, excluded, assigned, rows)
    return rows, report, spec_errors
