#!/usr/bin/env python3
"""Ingest a Parliamentary Question reply (.docx) into IRIS.

Renders the document to HTML with mammoth (bold/italic/underline/tables/lists
preserved), extracts plain text + metadata, copies the original .docx for
download, and inserts a row into the pq_documents table.

Run the Flask app once first so the pq_documents table exists, then:
    python tools/pq_to_iris.py "/path/to/PQ.docx" [--tags "dental, claims"]
"""
import argparse
import io
import os
import re
import shutil
import sqlite3
from datetime import datetime

import mammoth
import docx as docxlib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(HERE, "iris.db")
STATIC_PQ_DIR = os.path.join(HERE, "static", "documents", "pqs")


def parse_docx(src, filename=""):
    """Render a PQ .docx (path or raw bytes) to HTML + text + metadata.
    Returns a dict the CLI and the upload endpoint both use."""
    fobj = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else open(src, "rb")
    try:
        html = mammoth.convert_to_html(fobj, style_map="u => u").value
    finally:
        if not isinstance(src, (bytes, bytearray)):
            fobj.close()
    d = docxlib.Document(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    pq_no, house, subject, doc_date, title = _meta(text, filename or "")
    return {"html": html, "text": text, "pq_no": pq_no, "house": house,
            "subject": subject, "doc_date": doc_date, "title": title}


def _meta(text, filename):
    pq_no = (re.search(r"(?:question\s*no\.?\s*|pq[\s_-]*)(\d{2,6})", filename + " " + text, re.I)
             or re.search(r"\b(\d{3,6})\b", filename))
    pq_no = pq_no.group(1) if pq_no else ""
    house = ("Lok Sabha" if re.search(r"\blok\s*sabha\b|\bls\b", filename + " " + text, re.I)
             else "Rajya Sabha" if re.search(r"\brajya\s*sabha\b|\brs\b", filename + " " + text, re.I)
             else "")
    starred = "Starred" if re.search(r"starred", text, re.I) else "Unstarred" if re.search(r"unstarred", text, re.I) else ""
    sm = re.search(r"Subject:\s*(.+)", text)
    subject = re.sub(r"\s+", " ", sm.group(1)).strip().strip('"“”').strip()[:400] if sm else ""
    dm = re.search(r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4})", text)
    doc_date = dm.group(1) if dm else ""
    house_label = f"{house} {starred} Q".strip()
    title = subject or f"{house_label} No. {pq_no}".strip()
    return pq_no, house, subject, doc_date, title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", help="Path to the PQ .docx file")
    ap.add_argument("--tags", default="", help="Comma-separated tags")
    args = ap.parse_args()

    parsed = parse_docx(args.docx, os.path.basename(args.docx))
    html, text = parsed["html"], parsed["text"]
    pq_no, house, subject, doc_date, title = (
        parsed["pq_no"], parsed["house"], parsed["subject"], parsed["doc_date"], parsed["title"])

    os.makedirs(STATIC_PQ_DIR, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(args.docx))
    shutil.copy(args.docx, os.path.join(STATIC_PQ_DIR, safe))

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS pq_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT, pq_no TEXT, house TEXT, title TEXT,
        subject TEXT, doc_date TEXT, tags TEXT, html TEXT, body_text TEXT,
        docx_filename TEXT, created_at DATETIME)""")
    conn.execute("""INSERT INTO pq_documents
        (pq_no, house, title, subject, doc_date, tags, html, body_text, docx_filename, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (pq_no, house, title, subject, doc_date, args.tags.strip(), html, text, safe,
         datetime.utcnow().isoformat(sep=" ", timespec="seconds")))
    conn.commit()
    conn.close()
    print(f"[+] Ingested PQ {pq_no or '?'} ({house}) — {title[:60]!r}")
    print(f"    html={len(html)} chars | tables={html.count('<table>')} | "
          f"strong={html.count('<strong>')} em={html.count('<em>')} u={html.count('<u>')} | docx={safe}")


if __name__ == "__main__":
    main()
