"""Out-of-process PDF ingestion worker.

`pdfplumber` extraction spikes memory ~+600 MB per PDF and does not fully return
it to the OS, so running it inside the long-lived gunicorn worker made memory
ratchet up across imports until the container OOM-killed. This script runs the
heavy extraction in a FRESH, short-lived interpreter: it imports only `ingest`
(never the 640 MB financial engine), does the work, writes JSON to an output
file, and exits — at which point the OS reclaims 100% of the extraction memory.

Launched by api.py via subprocess (a clean process, NOT a fork of the threaded
gunicorn worker, which would be deadlock-prone). It never touches the database;
the parent process owns all DB writes.

Usage:
  python ingest_worker.py detect  <pdf_path> <out_json>
  python ingest_worker.py segment <pdf_path> <spec_id> <out_json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _write(out_path, obj):
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, default=str)


def main():
    try:
        mode = sys.argv[1]
        pdf_path = sys.argv[2]
        import ingest  # standalone — does not load the financial engine

        if mode == "detect":
            out_path = sys.argv[3]
            _write(out_path, {"ok": True, "ranked": ingest.detect_spec(pdf_path)})

        elif mode == "segment":
            spec_id, out_path = sys.argv[3], sys.argv[4]
            spec = ingest.load_spec(spec_id)
            if not spec:
                _write(out_path, {"ok": False, "error": "unknown_spec"})
                return
            rows, report, spec_errors = ingest.segment_pdf(pdf_path, spec)
            _write(out_path, {"ok": True, "rows": rows, "report": report,
                              "spec_errors": spec_errors})
        else:
            _write(sys.argv[-1], {"ok": False, "error": f"unknown_mode:{mode}"})
    except Exception as e:  # any parse/segment failure -> structured error for parent
        try:
            _write(sys.argv[-1], {"ok": False, "error": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
