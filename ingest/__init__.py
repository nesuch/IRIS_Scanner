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
