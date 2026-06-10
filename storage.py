"""Pluggable file storage for uploaded documents (e.g. PQ .docx).

Uses Google Cloud Storage when GCS_PQ_BUCKET is set (production on Cloud Run,
whose filesystem is ephemeral), and local disk otherwise (dev). Reads fall back
to the local static folder so files shipped inside the container still serve.
"""
import os

_BUCKET = os.environ.get("GCS_PQ_BUCKET", "").strip()
_HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_PQ_DIR = os.path.join(_HERE, "static", "documents", "pqs")
LOCAL_DOC_DIR = os.path.join(_HERE, "static", "documents", "uploads")
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_client = None


def using_gcs():
    return bool(_BUCKET)


def _bucket():
    global _client
    from google.cloud import storage  # imported lazily so dev needs no GCS SDK
    if _client is None:
        _client = storage.Client()
    return _client.bucket(_BUCKET)


def save_pq(filename, data):
    """Persist an uploaded PQ file; returns the stored filename."""
    if _BUCKET:
        _bucket().blob(f"pqs/{filename}").upload_from_string(data, content_type=_DOCX_MIME)
    else:
        os.makedirs(LOCAL_PQ_DIR, exist_ok=True)
        with open(os.path.join(LOCAL_PQ_DIR, filename), "wb") as fh:
            fh.write(data)
    return filename


def load_pq(filename):
    """Return the bytes of a stored PQ file, or None. GCS first, then local."""
    return _load("pqs", LOCAL_PQ_DIR, filename)


def save_doc_pdf(filename, data):
    """Persist an original source PDF attached to a document."""
    if _BUCKET:
        _bucket().blob(f"docpdf/{filename}").upload_from_string(data, content_type="application/pdf")
    else:
        os.makedirs(LOCAL_DOC_DIR, exist_ok=True)
        with open(os.path.join(LOCAL_DOC_DIR, filename), "wb") as fh:
            fh.write(data)
    return filename


def load_doc_pdf(filename):
    return _load("docpdf", LOCAL_DOC_DIR, filename)


def _load(prefix, local_dir, filename):
    if not filename:
        return None
    if _BUCKET:
        try:
            blob = _bucket().blob(f"{prefix}/{filename}")
            if blob.exists():
                return blob.download_as_bytes()
        except Exception as e:
            print(f"GCS load error: {e}")
    path = os.path.join(local_dir, os.path.basename(filename))
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read()
    return None
