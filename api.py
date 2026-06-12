"""IRIS_Scanner — additive JSON API layer for the React SPA.

This blueprint is a THIN wrapper. It does not contain business logic: every
handler calls the exact same `iris_brain.*` functions and the same auth/session
helpers already defined in `app.py`. The existing Jinja routes are left fully
intact; this only adds parallel `/api/*` endpoints that return JSON.

To avoid circular imports (app.py imports this module to register the blueprint)
and to avoid re-executing app.py as a second module, app.py injects its own
module object via `init_api(sys.modules[__name__])`. All shared objects (models,
db, login helpers, private helpers) are then reached through `_app`.
"""
import base64
import io
import os
import re
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
from flask import Blueprint, request, jsonify, session, send_file, Response

import iris_brain as brain
import storage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from pq_to_iris import parse_docx  # noqa: E402

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Injected reference to the running app.py module (set by init_api in app.py).
_app = None


def init_api(app_module):
    global _app
    _app = app_module


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _user_payload():
    """Shape of the current user for the SPA (mirrors inject_device_count)."""
    cu = _app.current_user
    if not cu.is_authenticated:
        return None
    return {
        "email": cu.email,
        "is_admin": bool(getattr(cu, "is_admin", False)),
        "role": _current_role(),
        "device_count": _app._get_active_device_count(cu.id),
        "display_name": getattr(cu, "display_name", None),
        "avatar": getattr(cu, "avatar", None),
    }


_ROLE_RANK = {"viewer": 0, "editor": 1, "admin": 2}


def _current_role():
    cu = _app.current_user
    if not cu.is_authenticated:
        return None
    if getattr(cu, "is_admin", False):
        return "admin"
    return getattr(cu, "role", "viewer") or "viewer"


def _require_role(minrole):
    r = _current_role()
    return r is not None and _ROLE_RANK.get(r, 0) >= _ROLE_RANK.get(minrole, 0)


def _require_admin():
    return _require_role("admin")


def _audit(action, status):
    """Record a content-change action in the admin audit log (who did what)."""
    try:
        email = getattr(_app.current_user, "email", "") or ""
        _app._record_admin_audit(email, str(action)[:255], str(status)[:64])
    except Exception as e:
        print(f"audit error: {e}")


# ----------------------------------------------------------------------------
# AUTH
# ----------------------------------------------------------------------------
@api_bp.post("/login")
def api_login():
    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    next_url = _app._safe_next_url(data.get("next") or "/")

    user = _app.User.query.filter(_app.db.func.lower(_app.User.email) == email).first()

    if not _app._is_allowed_email(email):
        _app._record_admin_audit(email, "login_attempt", "failure")
        return jsonify({"ok": False, "message": "Invalid credentials."}), 401
    if user and user.is_active and _app.check_password_hash(user.password_hash, password):
        _app.login_user(user)
        _app._start_user_session(user)
        _app._record_admin_audit(email, "login_attempt", "success")
        return jsonify({"ok": True, "user": _user_payload(), "next": next_url})
    _app._record_admin_audit(email, "login_attempt", "failure")
    return jsonify({"ok": False, "message": "Invalid credentials."}), 401


@api_bp.post("/logout")
def api_logout():
    if _app.current_user.is_authenticated:
        _app._deactivate_session_token(_app.current_user.id, session.get("auth_token"))
    _app.logout_user()
    session.clear()
    return jsonify({"ok": True})


@api_bp.post("/logout-all")
def api_logout_all():
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    user = _app.db.session.get(_app.User, _app.current_user.id)
    user.session_version = (user.session_version or 0) + 1
    _app.db.session.commit()
    _app._deactivate_all_user_sessions(user.id)
    _app._record_admin_audit(user.email, "logout_all_devices", "success")
    _app.logout_user()
    session.clear()
    return jsonify({"ok": True})


@api_bp.get("/me")
def api_me():
    payload = _user_payload()
    if not payload:
        return jsonify({"authenticated": False}), 401
    return jsonify({"authenticated": True, "user": payload})


@api_bp.post("/forgot-password")
def api_forgot_password():
    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip().lower()
    generic_msg = "If the account is eligible, a password reset link has been generated."
    if email and _app._is_allowed_email(email):
        user = _app.User.query.filter(_app.db.func.lower(_app.User.email) == email).first()
        if user and user.is_active:
            import secrets
            from datetime import timedelta
            raw_token = secrets.token_urlsafe(32)
            user.reset_token = _app._hash_reset_token(raw_token)
            user.reset_token_expiry = datetime.utcnow() + timedelta(minutes=15)
            _app.db.session.commit()
            reset_link = request.host_url.rstrip("/") + "/reset-password/" + raw_token
            try:
                _app.db.session.add(_app.PasswordResetAudit(
                    email=email, reset_link=reset_link,
                    requested_at=datetime.utcnow(), expires_at=user.reset_token_expiry))
                _app.db.session.commit()
            except Exception:
                pass
            _app._record_admin_audit(email, "password_reset_request", "success")
        else:
            _app._record_admin_audit(email, "password_reset_request", "failure")
    return jsonify({"ok": True, "message": generic_msg})


@api_bp.post("/reset-password/<token>")
def api_reset_password(token):
    token_hash = _app._hash_reset_token(token)
    user = _app.User.query.filter_by(reset_token=token_hash).first()
    valid = (user and user.is_active and user.reset_token_expiry
             and user.reset_token_expiry >= datetime.utcnow())
    if not valid:
        return jsonify({"ok": False, "message": "Invalid or expired reset link."}), 400
    data = request.get_json(silent=True) or request.form
    new_password = data.get("new_password") or ""
    confirm_password = data.get("confirm_password") or ""
    if len(new_password) < 8:
        return jsonify({"ok": False, "message": "Password must be at least 8 characters."}), 400
    if new_password != confirm_password:
        return jsonify({"ok": False, "message": "Passwords do not match."}), 400
    user.password_hash = _app.generate_password_hash(new_password)
    user.reset_token = None
    user.reset_token_expiry = None
    _app.db.session.commit()
    _app._record_admin_audit(user.email, "password_reset_complete", "success")
    return jsonify({"ok": True})


@api_bp.get("/reset-password/<token>/valid")
def api_reset_password_valid(token):
    token_hash = _app._hash_reset_token(token)
    user = _app.User.query.filter_by(reset_token=token_hash).first()
    valid = bool(user and user.is_active and user.reset_token_expiry
                 and user.reset_token_expiry >= datetime.utcnow())
    return jsonify({"valid": valid})


# ----------------------------------------------------------------------------
# SEARCH  (mirrors handle_search; returns structured data instead of HTML)
# ----------------------------------------------------------------------------
def _doc_pdf_url(source):
    """Prefer an admin-attached PDF for this document; else the bundled static PDF.
    Includes a version param so a replaced PDF busts the viewer/HTTP cache."""
    asset = _app.DocumentAsset.query.filter_by(source_doc=str(source)).first()
    if asset and asset.pdf_filename:
        v = int(asset.uploaded_at.timestamp()) if asset.uploaded_at else 0
        return f"/api/doc-pdf/{asset.id}/download?v={v}"
    p = _app.resolve_pdf_path(str(source).strip().upper())
    return ("/static/" + p) if p else None


def _match_payload(m):
    return {
        "source": m.get("source", "UNKNOWN"),
        "type": m.get("type", "UNKNOWN"),
        "id": str(m.get("id", "")).strip(),
        "header": m.get("header", ""),
        "raw_text": str(m.get("raw_text", "")),
        "pdf_url": _doc_pdf_url(m.get("source", "")),
        "tags": brain.clause_tags(m.get("id", ""), m.get("source", "")),
        "html": brain.clause_html(m.get("id", ""), m.get("source", "")),
        **_doc_status(m.get("source", "")),
    }


def _build_chips(kw_tuples, original_query):
    chips = []
    shown = set()
    for raw, clean in kw_tuples:
        label = f'Search "{raw}"'
        if label in shown:
            continue
        chips.append({"label": label, "payload": f"{raw}|{clean}", "kind": "keyword"})
        shown.add(label)
    clean_original = " ".join(original_query.split()).strip()
    phrase_label = f'Search Phrase "{clean_original}"'
    has_special_compound = bool(re.search(r"[-/]", clean_original))
    if (len(clean_original.split()) > 1 or has_special_compound) and phrase_label not in shown:
        chips.append({"label": phrase_label,
                      "payload": f"{clean_original}|{clean_original}", "kind": "phrase"})
    if len(kw_tuples) > 1:
        all_payload = "||".join([f"{t[0]}|{t[1]}" for t in kw_tuples])
        chips.append({"label": "Search All", "payload": all_payload, "kind": "all"})
    return chips


_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "document_registry.json")
_REGISTRY_CACHE = {}


def _load_registry():
    """Document registry (hierarchy + active/repealed status). Cached by mtime."""
    try:
        mtime = os.path.getmtime(_REGISTRY_PATH)
    except OSError:
        return []
    if _REGISTRY_CACHE.get("mtime") != mtime:
        import json
        with open(_REGISTRY_PATH, encoding="utf-8") as fh:
            docs = (json.load(fh) or {}).get("documents", [])
        _REGISTRY_CACHE.update(mtime=mtime, docs=docs,
                               by_id={d["id"]: d for d in docs})
    return _REGISTRY_CACHE.get("docs", [])


def _doc_status(source):
    d = _load_registry() and _REGISTRY_CACHE["by_id"].get(source)
    if not d:
        return {}
    return {"doc_status": d.get("status", "Active"),
            "effective_date": d.get("effective_date"),
            "repealed_on": d.get("repealed_on")}


@api_bp.get("/documents")
def api_documents():
    """Document tree (Act → Regulation → Circular) + download links + status, for
    the Downloads page."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    docs = _load_registry()
    # clause counts per document
    counts = {}
    try:
        kb = brain.load_knowledge_base()
        if kb is not None and not kb.empty:
            counts = kb["Source_Doc"].value_counts().to_dict()
    except Exception:
        pass
    nodes = {}
    for d in docs:
        pdf = _app.resolve_pdf_path(str(d["id"]).upper())
        nodes[d["id"]] = {
            "id": d["id"], "title": d.get("title", d["id"]), "type": d.get("type", "Document"),
            "category": d.get("category", ""), "parent": d.get("parent"),
            "status": d.get("status", "Active"),
            "effective_date": d.get("effective_date"), "repealed_on": d.get("repealed_on"),
            "repealed_by": d.get("repealed_by"),
            "clauses": int(counts.get(d["id"], 0)),
            "download_url": ("/static/" + pdf) if pdf else None,
            "children": [],
        }
    roots, repealed = [], []
    for n in nodes.values():
        if n["status"].lower() == "repealed":
            repealed.append(n)
        elif n["parent"] and n["parent"] in nodes:
            nodes[n["parent"]]["children"].append(n)
        else:
            roots.append(n)
    return jsonify({"tree": roots, "repealed": repealed})


def _pq_snippet(body, limit=220):
    """Skip the letterhead (Ref/date/address/Subject) and snippet the actual content."""
    if not body:
        return ""
    m = re.search(r"Subject\s*:.*?(?:\n|$)", body, re.I) or re.search(r"Dear\s+Sir.*?(?:\n|$)", body, re.I)
    rest = body[m.end():] if m else body
    rest = re.sub(r"\s+", " ", rest).strip()
    # Drop the standard covering-letter intro ("This has reference … seriatim for the same.")
    rest = re.sub(r"^This has reference.*?seriatim for the same\.\s*", "", rest, flags=re.I).strip()
    return rest[:limit] + ("…" if len(rest) > limit else "")


# Controlled department vocabulary for PQs — same lines as the KB modules.
# A PQ may belong to several (e.g. HEALTH + NONLIFE). Stored comma-separated.
PQ_DEPARTMENTS = [
    {"code": "HEALTH", "label": "Health"},
    {"code": "LIFE", "label": "Life"},
    {"code": "NONLIFE", "label": "Non-Life"},
]
_PQ_DEPT_CODES = {d["code"] for d in PQ_DEPARTMENTS}
# Accept loose inputs ("non-life", "Health", "nonlife") -> canonical code.
_PQ_DEPT_ALIASES = {"HEALTH": "HEALTH", "LIFE": "LIFE", "NONLIFE": "NONLIFE",
                    "NON-LIFE": "NONLIFE", "NON_LIFE": "NONLIFE", "GENERAL": "NONLIFE"}


def _norm_departments(value):
    """Normalise a list or comma-string of department inputs to canonical codes,
    de-duplicated and in the canonical Health/Life/Non-Life order."""
    if value is None:
        parts = []
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = str(value).split(",")
    seen = set()
    for p in parts:
        key = re.sub(r"\s+", "", str(p).strip().upper())
        code = _PQ_DEPT_ALIASES.get(key)
        if code:
            seen.add(code)
    return [d["code"] for d in PQ_DEPARTMENTS if d["code"] in seen]


def _dept_list(r):
    return _norm_departments(getattr(r, "departments", None))


def _pq_card(r):
    return {
        "id": r.id, "pq_no": r.pq_no, "house": r.house, "title": r.title,
        "subject": r.subject, "date": r.doc_date,
        "tags": [t.strip() for t in (r.tags or "").split(",") if t.strip()],
        "departments": _dept_list(r),
        "snippet": _pq_snippet(r.body_text or ""),
    }


@api_bp.get("/pq/tags")
def api_pq_tags():
    """Distinct tags across all PQs — the typeahead vocabulary. Deduplicated
    case- and space-insensitively (so 'Dental' / 'dental insurance' collapse)."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    canon = {}  # normalised key -> display label (prefer the most Title-cased variant)
    for r in _app.PqDocument.query.all():
        for t in (r.tags or "").split(","):
            t = t.strip()
            if not t:
                continue
            key = re.sub(r"\s+", "", t.lower())
            prev = canon.get(key)
            # Keep the variant with more capitalised words (nicer display).
            if prev is None or sum(c.isupper() for c in t) > sum(c.isupper() for c in prev):
                canon[key] = t
    return jsonify({"tags": sorted(canon.values(), key=str.lower)})


@api_bp.get("/pq")
def api_pq_list():
    """List/search Parliamentary Question replies for the dedicated PQ view."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    num = re.sub(r"\D", "", request.args.get("num") or "")
    tag = (request.args.get("tag") or "").strip()
    q = (request.args.get("q") or "").strip()
    deep = (request.args.get("deep") or "").strip()
    chips = []
    deep_flag = False
    if deep:
        # Deep Scan — read every reply body and return all that contain the query.
        mode = (request.args.get("mode") or "all").strip().lower()
        if mode not in {"word", "all", "phrase"}:
            mode = "all"
        items = _deep_scan_pqs(deep, mode=mode)
        deep_flag = True
        try:
            email = _app.current_user.email if _app.current_user.is_authenticated else None
            _app._record_search(email, "pq", "[Deep Scan] " + deep, len(items))
        except Exception:
            pass
    elif num:
        rows = _app.PqDocument.query.order_by(_app.PqDocument.id.desc()).all()
        items = [_pq_card(r) for r in rows if num in re.sub(r"\D", "", r.pq_no or "")]
    elif tag:
        # Exact-tag filter (case/space-insensitive) — not a body word search.
        key = re.sub(r"\s+", "", tag.lower())
        rows = _app.PqDocument.query.order_by(_app.PqDocument.id.desc()).all()
        items = [_pq_card(r) for r in rows
                 if key in {re.sub(r"\s+", "", t.strip().lower()) for t in (r.tags or "").split(",") if t.strip()}]
    elif q:
        items = _search_pqs(q, limit=100)
        chips = _pq_chips(q)   # offer Deep-Scan suggestions alongside headline hits
        try:
            email = _app.current_user.email if _app.current_user.is_authenticated else None
            _app._record_search(email, "pq", q, len(items))
        except Exception:
            pass
    else:
        rows = _app.PqDocument.query.order_by(_app.PqDocument.id.desc()).all()
        items = [_pq_card(r) for r in rows]
    # Department facet — narrows whatever the base set is (a PQ matches if it
    # carries the requested department; multi-dept PQs match any of theirs).
    depts = _norm_departments(request.args.get("dept"))
    if depts:
        want = set(depts)
        items = [it for it in items if want & set(it.get("departments") or [])]
    return jsonify({"items": items, "chips": chips, "deep": deep_flag})


@api_bp.get("/pq/<int:pid>")
def api_pq_get(pid):
    """Full rendered HTML + metadata + original-file link for one PQ."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    r = _app.PqDocument.query.get_or_404(pid)
    return jsonify({
        "id": r.id, "pq_no": r.pq_no, "house": r.house, "title": r.title,
        "subject": r.subject, "date": r.doc_date,
        "tags": [t.strip() for t in (r.tags or "").split(",") if t.strip()],
        "departments": _dept_list(r),
        "html": r.html,
        "download_url": (f"/api/pq/{r.id}/download" if r.docx_filename else None),
    })


@api_bp.get("/pq/<int:pid>/download")
def api_pq_download(pid):
    """Stream the original .docx from storage (GCS in prod, local in dev)."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    r = _app.PqDocument.query.get_or_404(pid)
    data = storage.load_pq(r.docx_filename)
    if data is None:
        return jsonify({"ok": False, "message": "Original file not found."}), 404
    return send_file(io.BytesIO(data),
                     mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                     as_attachment=True, download_name=r.docx_filename)


def _pq_duplicate(pq_no, house):
    """An existing PQ with the same number + house = a likely duplicate."""
    pq_no = (pq_no or "").strip()
    if not pq_no:
        return None
    return _app.PqDocument.query.filter_by(pq_no=pq_no, house=house or "").first()


@api_bp.post("/pq/upload")
def api_pq_upload():
    """Admin-only: upload a PQ .docx → render + store + make it searchable."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".docx"):
        return jsonify({"ok": False, "message": "Please upload a .docx file."}), 400
    tags = (request.form.get("tags") or "").strip()
    departments = ",".join(_norm_departments(request.form.get("departments")))
    force = str(request.form.get("force") or "").lower() in {"1", "true", "yes"}
    raw = f.read()
    try:
        parsed = parse_docx(raw, filename=f.filename)
    except Exception as e:
        print(f"PQ parse error: {e}")
        return jsonify({"ok": False, "message": "Could not read that document."}), 400
    dup = _pq_duplicate(parsed["pq_no"], parsed["house"])
    if dup and not force:
        return jsonify({"ok": False, "duplicate": True,
                        "existing": {"id": dup.id, "title": dup.title, "pq_no": dup.pq_no, "house": dup.house},
                        "message": f"A PQ {dup.house} No. {dup.pq_no} already exists: \"{dup.title[:80]}\"."}), 409
    m = _app
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(f.filename))
    storage.save_pq(safe, raw)   # GCS in prod, local disk in dev
    row = m.PqDocument(
        pq_no=parsed["pq_no"], house=parsed["house"], title=parsed["title"],
        subject=parsed["subject"], doc_date=parsed["doc_date"], tags=tags,
        departments=departments,
        html=parsed["html"], body_text=parsed["text"], docx_filename=safe,
        created_at=datetime.utcnow())
    m.db.session.add(row)
    m.db.session.commit()
    _audit(f"Uploaded PQ: {row.title}", "PQ upload")
    return jsonify({"ok": True, "id": row.id, "title": row.title}), 201


@api_bp.post("/pq/bulk-upload")
def api_pq_bulk_upload():
    """Admin-only: upload several PQ .docx at once (no tags yet). Returns the
    created PQs so the UI can walk through titles/tags."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "message": "No files provided."}), 400
    m = _app
    created, skipped, duplicates = [], [], []
    for f in files:
        if not f.filename.lower().endswith(".docx"):
            skipped.append(f.filename); continue
        raw = f.read()
        try:
            parsed = parse_docx(raw, filename=f.filename)
        except Exception as e:
            print(f"bulk PQ parse error ({f.filename}): {e}")
            skipped.append(f.filename); continue
        dup = _pq_duplicate(parsed["pq_no"], parsed["house"])
        if dup:
            duplicates.append({"filename": f.filename, "pq_no": dup.pq_no,
                               "house": dup.house, "existing_title": dup.title})
            continue
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(f.filename))
        storage.save_pq(safe, raw)
        row = m.PqDocument(
            pq_no=parsed["pq_no"], house=parsed["house"], title=parsed["title"],
            subject=parsed["subject"], doc_date=parsed["doc_date"], tags="",
            html=parsed["html"], body_text=parsed["text"], docx_filename=safe,
            created_at=datetime.utcnow())
        m.db.session.add(row)
        m.db.session.commit()
        created.append({"id": row.id, "title": row.title, "pq_no": row.pq_no,
                        "house": row.house, "filename": f.filename})
    _audit(f"Bulk uploaded {len(created)} PQs", "PQ upload")
    return jsonify({"ok": True, "created": created, "skipped": skipped})


@api_bp.post("/pq/<int:pid>/delete")
def api_pq_delete(pid):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    r = m.PqDocument.query.get_or_404(pid)
    pqno = r.pq_no or pid
    m.db.session.delete(r)
    m.db.session.commit()
    _audit(f"Deleted PQ {pqno}", "PQ delete")
    return jsonify({"ok": True})


# Short words that shouldn't drive PQ matching on their own.
_PQ_STOP = {"the", "and", "for", "with", "that", "this", "from", "are", "was",
            "were", "has", "have", "had", "not", "you", "your", "our", "its",
            "into", "per", "all", "any", "can", "under", "over", "about", "shall",
            "such", "been", "than", "then", "they", "them", "their", "would",
            "regarding", "respect", "whether", "question", "answer", "reply"}


def _pq_terms(text):
    """Significant lowercase words in a query (drops stopwords + tiny tokens)."""
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower())
            if len(t) > 2 and t not in _PQ_STOP]


def _search_pqs(query, limit=100):
    """Headline (precise) tier — match a PQ by its title, subject or tags only.
    Full reply bodies are reserved for Deep Scan, so this stays high-precision."""
    m = _app
    terms = _pq_terms(query)
    if not terms:
        return []
    out = []
    for r in m.PqDocument.query.order_by(m.PqDocument.id.desc()).all():
        hay = " ".join([r.title or "", r.subject or "", r.tags or ""]).lower()
        score = sum(hay.count(t) for t in terms)
        if not score:
            continue
        out.append((score, _pq_card(r)))
    out.sort(key=lambda x: -x[0])
    return [p for _, p in out[:limit]]


def _deep_scan_pqs(text, mode="all", limit=300):
    """Deep Scan (recall) tier — scan the FULL rendered reply text of every PQ
    and return each one that contains the query. mode:
      word   -> a single term anywhere in the reply
      all    -> every significant word present (AND)
      phrase -> the exact phrase appears verbatim"""
    m = _app
    phrase = " ".join((text or "").lower().split())
    terms = _pq_terms(text)
    out = []
    for r in m.PqDocument.query.order_by(m.PqDocument.id.desc()).all():
        hay = " ".join([r.title or "", r.subject or "", r.tags or "", r.body_text or ""]).lower()
        if mode == "phrase":
            if not phrase or phrase not in hay:
                continue
            score = hay.count(phrase)
        else:  # word / all  (a single-word chip is just AND over one term)
            if not terms or not all(t in hay for t in terms):
                continue
            score = sum(hay.count(t) for t in terms)
        out.append((score, _pq_card(r)))
    out.sort(key=lambda x: -x[0])
    return [p for _, p in out[:limit]]


def _pq_chips(query):
    """Smart Deep-Scan suggestion chips for a query, mirroring the KB modules:
    one per significant word, an exact-phrase chip, and a 'Search All' chip."""
    seen, words = set(), []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/]*", query or ""):
        wl = w.lower()
        if len(wl) > 2 and wl not in _PQ_STOP and wl not in seen:
            seen.add(wl)
            words.append(w)
    chips = [{"label": f'Search "{w}"', "mode": "word", "text": w, "kind": "keyword"}
             for w in words]
    clean = " ".join((query or "").split())
    if (len(clean.split()) > 1 or re.search(r"[-/]", clean)):
        chips.append({"label": f'Search Phrase "{clean}"', "mode": "phrase",
                      "text": clean, "kind": "phrase"})
    if len(words) > 1:
        chips.append({"label": "Search All", "mode": "all", "text": clean, "kind": "all"})
    return chips


@api_bp.post("/pq/<int:pid>/retag")
def api_pq_retag(pid):
    """Admin-only: update a PQ's tags (drives search)."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    r = _app.PqDocument.query.get_or_404(pid)
    data = request.get_json(silent=True) or {}
    r.tags = (data.get("tags") or "").strip()
    _app.db.session.commit()
    _audit(f"Re-tagged PQ {r.pq_no or r.id}", "PQ tags")
    return jsonify({"ok": True, "tags": [t.strip() for t in r.tags.split(",") if t.strip()]})


@api_bp.get("/clause/docs")
def api_clause_docs():
    """Admin: documents in the knowledge base, with clause + edited counts."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    df = brain.load_knowledge_base()
    if df is None or df.empty:
        return jsonify({"docs": []})
    has_html = "clause_html" in df.columns
    docs = []
    for src, g in df.groupby("Source_Doc"):
        edited = int(g["clause_html"].apply(lambda v: bool(pd.notna(v) and str(v).strip())).sum()) if has_html else 0
        asset = _app.DocumentAsset.query.filter_by(source_doc=str(src)).first()
        docs.append({
            "source": str(src),
            "type": str(g["Doc_Type"].iloc[0]) if "Doc_Type" in g.columns else "",
            "clauses": int(len(g)),
            "edited": edited,
            "pdf_url": _doc_pdf_url(str(src)),
            "has_uploaded_pdf": bool(asset and asset.pdf_filename),
        })
    docs.sort(key=lambda d: d["source"].lower())
    return jsonify({"docs": docs})


@api_bp.get("/clause/specs")
def api_clause_specs():
    """Admin: available document-type specs for PDF import."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    import ingest
    return jsonify({"specs": ingest.list_specs()})


@api_bp.post("/clause/detect-spec")
def api_clause_detect_spec():
    """Admin: rank existing specs by how well they segment the uploaded PDF."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    import ingest, tempfile
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "message": "Upload a PDF"}), 400
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(f.read()); tmp.close()
        ranked = ingest.detect_spec(tmp.name)
    except Exception as e:
        print(f"detect-spec error: {e}")
        return jsonify({"ok": False, "message": "Could not read that PDF."}), 400
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass
    return jsonify({"ok": True, "ranked": ranked})


@api_bp.post("/clause/import-pdf")
def api_clause_import_pdf():
    """Admin: PDF -> deterministic segmentation -> new document of editable clauses."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    import ingest, tempfile, sqlite3 as _sql
    f = request.files.get("file")
    spec_id = (request.form.get("spec_id") or "").strip()
    source = (request.form.get("source") or "").strip()
    doc_type = (request.form.get("doc_type") or "REGULATION").strip().upper()
    category = (request.form.get("category") or "GENERAL").strip().upper()
    if not f or not f.filename.lower().endswith(".pdf") or not spec_id or not source:
        return jsonify({"ok": False, "message": "Provide a PDF, a spec, and a document name."}), 400

    df = brain.load_knowledge_base()
    if df is not None and not df.empty and (df["Source_Doc"].astype(str) == source).any():
        return jsonify({"ok": False, "message": f"A document named '{source}' already exists. Use a new name."}), 409

    spec = ingest.load_spec(spec_id)
    if not spec:
        return jsonify({"ok": False, "message": "Unknown spec."}), 400

    raw = f.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(raw); tmp.close()
        rows, report, spec_errors = ingest.segment_pdf(tmp.name, spec)
    except Exception as e:
        print(f"PDF import error: {e}")
        return jsonify({"ok": False, "message": "Could not segment that PDF."}), 400
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass

    if not rows:
        return jsonify({"ok": False, "message": "No clauses were found — wrong spec for this document?"}), 400

    # Insert the produced clauses as a new document.
    conn = _sql.connect(brain.DB_NAME)
    for i, r in enumerate(rows, 1):
        clause = str(r.get("clause", ""))
        header = clause.split("\n", 1)[0].rstrip(":")[:120]
        is_header = 1 if clause.strip().endswith(":") and "\n" not in clause.strip() else 0
        conn.execute(
            "INSERT INTO regulatory_clauses (source_doc, doc_category, doc_type, clause_id, "
            "clause_text, context_header, regulatory_tags, priority, is_header, sort_order) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (source, category, doc_type, str(r.get("id", "")), clause, header,
             str(r.get("tag", "")).replace("_", " "), 99, is_header, i * 10))
    conn.commit(); conn.close()

    # Attach the source PDF to the new document, and refresh search.
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", source) + ".pdf"
    storage.save_doc_pdf(safe, raw)
    m = _app
    asset = m.DocumentAsset(source_doc=source, pdf_filename=safe,
                            uploaded_by=getattr(m.current_user, "email", "") or "",
                            uploaded_at=datetime.utcnow())
    m.db.session.add(asset)
    m.db.session.commit()
    brain.refresh_kb()
    _audit(f"Imported document '{source}' ({len(rows)} clauses)", "Document import")
    return jsonify({"ok": True, "source": source, "clauses": len(rows),
                    "orphans": report.get("orphan_lines", 0), "spec_errors": spec_errors})


@api_bp.get("/clause/history")
def api_clause_history():
    """Admin: prior versions of a clause (newest first) for review/restore."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    import sqlite3 as _sql
    source = (request.args.get("source") or "").strip()
    cid = (request.args.get("id") or "").strip()
    conn = _sql.connect(brain.DB_NAME)
    rows = conn.execute(
        "SELECT id, html, body_text, tags, edited_by, edited_at FROM clause_versions "
        "WHERE source_doc=? AND clause_id=? ORDER BY id DESC LIMIT 50", (source, cid)).fetchall()
    conn.close()
    out = []
    for vid, html, body, tags, by, at in rows:
        h = html or _clause_text_to_html(body or "")
        snip = re.sub(r"\s+", " ", body or re.sub(r"<[^>]+>", " ", h))[:160]
        out.append({"version_id": vid, "edited_by": by or "", "edited_at": at or "",
                    "snippet": snip, "html": h,
                    "tags": [t.strip() for t in (tags or "").split(",") if t.strip()]})
    return jsonify({"source": source, "id": cid, "versions": out})


@api_bp.post("/clause/editing")
def api_clause_editing():
    """Advisory presence heartbeat: mark me as editing `source`, return who else is."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    data = request.get_json(silent=True) or {}
    source = (data.get("source") or "").strip()
    email = getattr(_app.current_user, "email", "") or ""
    now = datetime.utcnow()
    m = _app
    if data.get("leave"):
        m.EditingSession.query.filter_by(source_doc=source, email=email).delete()
        m.db.session.commit()
        return jsonify({"others": []})
    sess = m.EditingSession.query.filter_by(source_doc=source, email=email).first()
    if sess:
        sess.last_seen = now
    else:
        m.db.session.add(m.EditingSession(source_doc=source, email=email, last_seen=now))
    m.db.session.commit()
    cutoff = now - timedelta(seconds=60)
    others = [s.email for s in m.EditingSession.query.filter(
        m.EditingSession.source_doc == source, m.EditingSession.email != email,
        m.EditingSession.last_seen >= cutoff).all()]
    return jsonify({"others": sorted(set(others))})


@api_bp.get("/clause/doc-full")
def api_clause_doc_full():
    """Admin: every clause of a document with full HTML + tags, in order — for the
    Studio client-side editor."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    source = (request.args.get("source") or "").strip()
    df = brain.load_knowledge_base()
    if df is None or df.empty:
        return jsonify({"source": source, "clauses": []})
    sub = df[df["Source_Doc"].astype(str) == source]
    if "sort_order" in sub.columns:
        sub = sub.sort_values("sort_order", kind="stable", na_position="last")
    out = []
    has_upd = "updated_by" in sub.columns
    for _, r in sub.iterrows():
        cid = str(r.get("Clause_ID", "")).strip()
        html = brain.clause_html(cid, source) or _clause_text_to_html(str(r.get("Clause_Text", "")))
        edited = bool(has_upd and pd.notna(r.get("updated_by")) and str(r.get("updated_by")).strip())
        out.append({"id": cid, "html": html, "edited": edited,
                    "tags": [t.strip() for t in str(r.get("Regulatory_Tags", "")).split(",") if t.strip()]})
    return jsonify({"source": source, "clauses": out, "rev": brain.doc_revision(source)})


@api_bp.post("/clause/doc-save")
def api_clause_doc_save():
    """Admin: bulk-save the edited document (full ordered clause list)."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    data = request.get_json(silent=True) or {}
    source = (data.get("source") or "").strip()
    clauses = data.get("clauses") or []
    if not source or not isinstance(clauses, list) or not clauses:
        return jsonify({"ok": False, "message": "Nothing to save."}), 400
    base_rev = (data.get("base_rev") or "").strip()
    if base_rev and brain.doc_revision(source) != base_rev:
        return jsonify({"ok": False, "conflict": True,
                        "message": "This document was changed by someone else since you opened it. "
                                   "Reload to get the latest version before saving."}), 409
    n = brain.replace_document_clauses(source, clauses, getattr(_app.current_user, "email", "") or "")
    changed = sum(1 for c in clauses if c.get("changed"))
    _audit(f"Saved '{source}' — {changed} of {n} clause(s) changed", "Document edit")
    return jsonify({"ok": True, "clauses": n, "rev": brain.doc_revision(source)})


@api_bp.post("/clause/doc-delete")
def api_doc_delete():
    """Admin: delete an entire document (all clauses + attached PDF)."""
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    data = request.get_json(silent=True) or {}
    source = (data.get("source") or "").strip()
    if not source:
        return jsonify({"ok": False, "message": "Missing document"}), 400
    n = brain.delete_document(source, getattr(_app.current_user, "email", "") or "")
    asset = _app.DocumentAsset.query.filter_by(source_doc=source).first()
    if asset:
        try:
            storage.delete_doc_pdf(asset.pdf_filename)
        except Exception as e:
            print(f"doc pdf delete: {e}")
        _app.db.session.delete(asset)
        _app.db.session.commit()
    _audit(f"Deleted document '{source}' ({n} clauses)", "Document delete")
    return jsonify({"ok": True, "deleted": n})


@api_bp.post("/clause/doc-pdf")
def api_doc_pdf_upload():
    """Admin: attach/replace the original source PDF for a document."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    source = (request.form.get("source") or "").strip()
    f = request.files.get("file")
    if not source or not f or not f.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "message": "Provide a document and a .pdf file."}), 400
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", source) + ".pdf"
    storage.save_doc_pdf(safe, f.read())
    m = _app
    asset = m.DocumentAsset.query.filter_by(source_doc=source).first()
    if asset is None:
        asset = m.DocumentAsset(source_doc=source)
        m.db.session.add(asset)
    asset.pdf_filename = safe
    asset.uploaded_by = getattr(m.current_user, "email", "") or ""
    asset.uploaded_at = datetime.utcnow()
    m.db.session.commit()
    _audit(f"Attached source PDF to '{source}'", "Document PDF")
    return jsonify({"ok": True, "pdf_url": f"/api/doc-pdf/{asset.id}/download?v={int(asset.uploaded_at.timestamp())}"})


@api_bp.get("/doc-pdf/<int:aid>/download")
def api_doc_pdf_serve(aid):
    """Stream an attached document PDF (inline) for viewing/download."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    a = _app.DocumentAsset.query.get_or_404(aid)
    data = storage.load_doc_pdf(a.pdf_filename)
    if data is None:
        return jsonify({"ok": False, "message": "PDF not found"}), 404
    return send_file(io.BytesIO(data), mimetype="application/pdf",
                     as_attachment=False, download_name=f"{a.source_doc}.pdf")


@api_bp.get("/clause/list")
def api_clause_list_for_doc():
    """Admin: ordered clauses of one document for the editor workspace."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    source = (request.args.get("source") or "").strip()
    df = brain.load_knowledge_base()
    if df is None or df.empty:
        return jsonify({"clauses": []})
    sub = df[df["Source_Doc"].astype(str) == source]
    if "sort_order" in sub.columns:
        sub = sub.sort_values("sort_order", kind="stable", na_position="last")
    has_html = "clause_html" in sub.columns
    out = []
    for _, r in sub.iterrows():
        txt = str(r.get("Clause_Text", ""))
        edited = bool(has_html and pd.notna(r.get("clause_html")) and str(r.get("clause_html")).strip())
        out.append({
            "id": str(r.get("Clause_ID", "")).strip(),
            "header": str(r.get("Context_Header", "")),
            "is_header": bool(r.get("Is_Header")),
            "edited": edited,
            "preview": re.sub(r"\s+", " ", txt)[:90],
        })
    return jsonify({"source": source, "clauses": out})


def _clause_text_to_html(text):
    """Seed the editor from a legacy plain-text/markdown clause: lines -> <p>,
    a line ending in ':' -> bold heading, GFM table blocks -> <table>."""
    from html import escape
    lines = str(text or "").split("\n")
    out, tbl = [], []

    def flush_tbl():
        if not tbl:
            return
        rows = [r for r in tbl if not re.match(r"^\|[\s\-:|]+\|$", r.strip())]
        cells_html = []
        for i, r in enumerate(rows):
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            tag = "th" if i == 0 else "td"
            cells_html.append("<tr>" + "".join(f"<{tag}>{escape(c)}</{tag}>" for c in cells) + "</tr>")
        out.append("<table>" + "".join(cells_html) + "</table>")
        tbl.clear()

    for ln in lines:
        s = ln.strip()
        if s.startswith("|") and s.endswith("|"):
            tbl.append(s)
            continue
        flush_tbl()
        if not s:
            continue
        if s.endswith(":"):
            out.append(f"<p><strong>{escape(s)}</strong></p>")
        else:
            out.append(f"<p>{escape(s)}</p>")
    flush_tbl()
    return "".join(out) or "<p></p>"


@api_bp.get("/clause/edit")
def api_clause_edit_get():
    """Admin-only: initial rich-text content for editing a clause."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    cid = (request.args.get("id") or "").strip()
    source = (request.args.get("source") or "").strip()
    existing = brain.clause_html(cid, source)
    if not existing:
        # seed from the current plain-text/markdown clause
        df = brain.KB_CACHE_DF
        txt = ""
        if df is not None and not df.empty:
            mask = (df["Clause_ID"].astype(str) == cid) & (df["Source_Doc"].astype(str) == source)
            sub = df.loc[mask, "Clause_Text"]
            if len(sub):
                txt = str(sub.iloc[0])
        existing = _clause_text_to_html(txt)
    return jsonify({"ok": True, "id": cid, "source": source, "html": existing})


@api_bp.post("/clause/edit")
def api_clause_edit_save():
    """Admin-only: save edited clause HTML (+ derived plain text)."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    data = request.get_json(silent=True) or {}
    cid = (data.get("id") or "").strip()
    source = (data.get("source") or "").strip()
    if not cid or not source:
        return jsonify({"ok": False, "message": "Missing clause id/source"}), 400
    editor = getattr(_app.current_user, "email", "") or ""
    ok = brain.update_clause_content(cid, source, data.get("html") or "", data.get("text") or "", editor)
    if not ok:
        return jsonify({"ok": False, "message": "Clause not found"}), 404
    _audit(f"Edited clause {cid} in '{source}'", "Clause edit")
    return jsonify({"ok": True, "html": data.get("html") or ""})


@api_bp.post("/clause/add")
def api_clause_add():
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    d = request.get_json(silent=True) or {}
    src, after = (d.get("source") or "").strip(), (d.get("after_id") or "").strip()
    if not src or not after:
        return jsonify({"ok": False, "message": "Missing source/after_id"}), 400
    new_id = brain.add_clause(src, after, getattr(_app.current_user, "email", "") or "")
    if not new_id:
        return jsonify({"ok": False, "message": "Could not add clause"}), 400
    return jsonify({"ok": True, "id": new_id})


@api_bp.post("/clause/remove")
def api_clause_remove():
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    d = request.get_json(silent=True) or {}
    src, cid = (d.get("source") or "").strip(), (d.get("id") or "").strip()
    if not brain.delete_clause(src, cid, getattr(_app.current_user, "email", "") or ""):
        return jsonify({"ok": False, "message": "Clause not found"}), 404
    return jsonify({"ok": True})


@api_bp.post("/clause/merge")
def api_clause_merge():
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    d = request.get_json(silent=True) or {}
    src, cid = (d.get("source") or "").strip(), (d.get("id") or "").strip()
    if not brain.merge_clause(src, cid, getattr(_app.current_user, "email", "") or ""):
        return jsonify({"ok": False, "message": "Nothing to merge into (it may be the first clause)."}), 400
    return jsonify({"ok": True})


@api_bp.post("/clause/move")
def api_clause_move():
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    d = request.get_json(silent=True) or {}
    src, cid = (d.get("source") or "").strip(), (d.get("id") or "").strip()
    direction = "up" if (d.get("direction") or "up") == "up" else "down"
    if not brain.move_clause(src, cid, direction):
        return jsonify({"ok": False, "message": "Can't move further."}), 400
    return jsonify({"ok": True})


@api_bp.post("/clause/retag")
def api_clause_retag():
    """Admin-only: edit a clause's tags in-app (persists + refreshes search)."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    data = request.get_json(silent=True) or {}
    cid = (data.get("id") or "").strip()
    source = (data.get("source") or "").strip()
    if not cid or not source:
        return jsonify({"ok": False, "message": "Missing clause id/source"}), 400
    new_tags = brain.update_clause_tags(cid, source, data.get("tags") or "")
    if new_tags is None:
        return jsonify({"ok": False, "message": "Clause not found"}), 404
    _audit(f"Re-tagged clause {cid} in '{source}'", "Clause tags")
    return jsonify({"ok": True, "tags": new_tags})


@api_bp.post("/pq/<int:pid>/update")
def api_pq_update(pid):
    """Admin: edit a PQ's title, tags and/or departments."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    r = _app.PqDocument.query.get_or_404(pid)
    data = request.get_json(silent=True) or {}
    if "title" in data and (data.get("title") or "").strip():
        r.title = data["title"].strip()[:400]
    if "tags" in data:
        r.tags = (data.get("tags") or "").strip()
    if "departments" in data:
        r.departments = ",".join(_norm_departments(data.get("departments")))
    _app.db.session.commit()
    _audit(f"Edited PQ {r.pq_no or pid}", "PQ edit")
    return jsonify({"ok": True, "title": r.title,
                    "tags": [t.strip() for t in (r.tags or "").split(",") if t.strip()],
                    "departments": _dept_list(r)})


@api_bp.get("/clause-suggest")
def api_clause_suggest():
    """Typeahead for the '/'-prefixed clause-number search."""
    q = request.args.get("q", "")
    sources = request.args.getlist("source") or None
    KB_DF = brain.load_knowledge_base()
    rows = brain.search_by_clause_number(q, KB_DF, sources=sources, limit=30)
    out = [{"id": r["id"], "source": r["source"], "type": r["type"],
            "snippet": (str(r["raw_text"])[:90] + ("…" if len(str(r["raw_text"])) > 90 else ""))}
           for r in rows]
    return jsonify({"suggestions": out})


@api_bp.post("/search")
def api_search():
    data = request.get_json(silent=True) or request.form
    module = (data.get("module") or "universal").strip()
    query = (data.get("query") or "").strip()
    # Optional source filter: None => all docs (default); a list => only those docs.
    sources = data.get("sources")

    def _scope(matches):
        if sources is None:
            return matches
        allow = set(sources)
        return [m for m in matches if m.get("source") in allow]

    KB_DF = brain.load_knowledge_base()

    if not query:
        return jsonify({"ok": False, "message": "Empty query."}), 400

    # --- Deep scan (triggered by chips) ---
    if query.startswith("__DEEP_SCAN__:"):
        raw_payload = query.replace("__DEEP_SCAN__:", "")
        pairs = raw_payload.split("||")
        keyword_tuples = [(p.split("|")[0], p.split("|")[1]) for p in pairs if len(p.split("|")) == 2]
        display_kws = [t[0] for t in keyword_tuples]
        tag_matches = brain.search_tags_only(keyword_tuples, KB_DF, module=module)
        exclude_ids = [m["id"] for m in tag_matches]
        matches = _scope(brain.deep_scan_brain(keyword_tuples, KB_DF, exclude_ids=exclude_ids, module=module))
        # Record deep scans too (with a readable label + result count).
        try:
            email = _app.current_user.email if _app.current_user.is_authenticated else None
            _app._record_search(email, module, "[Deep Scan] " + ", ".join(display_kws), len(matches))
        except Exception:
            pass
        return jsonify({
            "ok": True, "module": module, "kind": "deep_scan",
            "query_label": "Deep Scan",
            "keywords": display_kws, "highlight": display_kws,
            "matches": [_match_payload(m) for m in matches],
            "chips": [],
            "note": None if matches else f"No additional matches found in {module.capitalize()} module.",
        })

    # --- Clause-number lookup (query starts with "/") ---
    if query.lstrip().startswith("/"):
        matches = _scope(brain.search_by_clause_number(query, KB_DF, sources=sources))
        try:
            email = _app.current_user.email if _app.current_user.is_authenticated else None
            _app._record_search(email, module, query, len(matches))
        except Exception:
            pass
        return jsonify({
            "ok": True, "module": module, "kind": "clause_number",
            "query_label": query, "keywords": [], "highlight": [],
            "matches": [_match_payload(m) for m in matches], "chips": [],
            "note": None if matches else "No clause found with that number in the selected documents.",
        })

    # --- Greeting ---
    if brain.check_greeting(query):
        return jsonify({"ok": True, "module": module, "kind": "greeting",
                        "query_label": query, "matches": [], "chips": [], "keywords": []})

    # --- Standard search ---
    kw_tuples = brain.get_clean_keywords(query)
    display_kws = [t[0] for t in kw_tuples]
    if not kw_tuples:
        return jsonify({"ok": True, "module": module, "kind": "rejected",
                        "query_label": query, "matches": [], "chips": [], "keywords": [],
                        "note": "Query rejected. Please use regulatory terms."})

    tag_matches = _scope(brain.search_tags_only(kw_tuples, KB_DF, module=module))
    highlight_kws = [raw for (raw, clean) in kw_tuples if clean in brain.ALL_UNIQUE_TAGS]

    # Record the query for admin usage visibility (best-effort, non-blocking).
    try:
        email = _app.current_user.email if _app.current_user.is_authenticated else None
        _app._record_search(email, module, query, len(tag_matches))
    except Exception:
        pass

    note = None
    if not tag_matches:
        if module == "life":
            note = "Life Department: No documents currently loaded."
        elif module == "nonlife":
            note = "Non-Life Department: No documents currently loaded."
        elif module == "data":
            note = "Data Module: Text search is disabled here."
        else:
            note = f"No matches found in {module.capitalize()} for: {', '.join(display_kws)}"

    return jsonify({
        "ok": True, "module": module, "kind": "tags",
        "query_label": query,
        "keywords": display_kws,
        "highlight": highlight_kws if tag_matches else display_kws,
        "matches": [_match_payload(m) for m in tag_matches],
        "chips": _build_chips(kw_tuples, query),
        "note": note,
    })


@api_bp.get("/vocab")
def api_vocab():
    brain.load_knowledge_base()
    return jsonify(brain.get_autocomplete_data())


# Documents available to a module, grouped by doc type (for the search source filter).
_DOC_TYPE_ORDER = ["ACT", "REGULATION", "MASTER", "CIRCULAR", "GUIDELINE", "UNKNOWN"]
_DOC_TYPE_LABELS = {
    "ACT": "Acts", "REGULATION": "Regulations", "MASTER": "Master Circulars",
    "CIRCULAR": "Circulars", "GUIDELINE": "Guidelines", "UNKNOWN": "Other",
}


@api_bp.get("/docs")
def api_docs():
    module = (request.args.get("module") or "universal").strip()
    df = brain.load_knowledge_base()
    scoped = brain.filter_df_by_module(df, module)

    src_type = {}
    src_tags = {}
    if scoped is not None and not scoped.empty:
        for _, row in scoped.iterrows():
            src = str(row.get("Source_Doc") or "").strip()
            if not src:
                continue
            src_type.setdefault(src, str(row.get("Doc_Type") or "UNKNOWN").strip().upper() or "UNKNOWN")
            raw_tags = str(row.get("Regulatory_Tags") or "")
            if raw_tags:
                tset = src_tags.setdefault(src, set())
                for t in raw_tags.split(","):
                    clean = t.strip().replace("_", " ").lower()
                    if len(clean) >= 2:
                        tset.add(clean)

    by_type = {}
    for src, typ in src_type.items():
        by_type.setdefault(typ, []).append(src)
    for docs in by_type.values():
        docs.sort()

    ordered_types = [t for t in _DOC_TYPE_ORDER if t in by_type]
    ordered_types += [t for t in by_type if t not in _DOC_TYPE_ORDER]
    groups = [{"type": t, "label": _DOC_TYPE_LABELS.get(t, t.title()), "docs": by_type[t]} for t in ordered_types]
    doc_tags = {src: sorted(tags) for src, tags in src_tags.items()}
    return jsonify({"module": module, "groups": groups, "doc_tags": doc_tags})


# ----------------------------------------------------------------------------
# DATA EXPLORER
# ----------------------------------------------------------------------------
def _filters_from_request():
    data = request.get_json(silent=True)
    if data is not None:
        return {
            "dimension": data.get("dimension", "Insurer"),
            "entities": data.get("entities", []) or [],
            "metrics": data.get("metrics", []) or [],
            "years": data.get("years", []) or [],
            "quarters": data.get("quarters", []) or [],
            "lobs": data.get("lobs", []) or [],
            "classes": data.get("classes", []) or [],
        }
    return {
        "dimension": request.form.get("dimension", "Insurer"),
        "entities": request.form.getlist("entities"),
        "metrics": request.form.getlist("metrics"),
        "years": request.form.getlist("years"),
        "quarters": request.form.getlist("quarters"),
        "lobs": request.form.getlist("lobs"),
        "classes": request.form.getlist("classes"),
    }


@api_bp.get("/data/options")
def api_data_options():
    return jsonify(brain.get_filter_options())


@api_bp.post("/data/filter")
def api_data_filter():
    return jsonify(brain.filter_data(_filters_from_request()))


@api_bp.post("/data/statement")
def api_data_statement():
    data = request.get_json(silent=True) or {}
    return jsonify(brain.build_financial_statement(
        data.get("entities", []) or [],
        data.get("statement", ""),
        data.get("years", []) or [],
    ))


@api_bp.post("/data/statement/download")
def api_data_statement_download():
    data = request.get_json(silent=True) or {}
    statement = data.get("statement", "")
    excel_file = brain.generate_statement_excel(
        data.get("entities", []) or [], statement, data.get("years", []) or [])
    if excel_file:
        safe = "".join(c for c in statement if c.isalnum() or c in " _-").strip() or "Statement"
        return send_file(
            excel_file,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=f"IRIS_{safe}.xlsx",
        )
    return jsonify({"ok": False, "message": "No statement data."}), 400


_GUIDE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base",
                           "IRIS_Handbook_Access_Guide.xlsx")
_GUIDE_CACHE = {}


def _load_access_guide():
    if "rows" not in _GUIDE_CACHE:
        try:
            df = pd.read_excel(_GUIDE_PATH).fillna("")
            _GUIDE_CACHE["rows"] = df.to_dict("records")
        except Exception as e:
            print(f"access guide load error: {e}")
            _GUIDE_CACHE["rows"] = []
    return _GUIDE_CACHE["rows"]


@api_bp.get("/guide")
def api_guide():
    """Handbook→IRIS access guide as JSON, for the in-app searchable viewer."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    return jsonify({"rows": _load_access_guide()})


@api_bp.get("/guide/download")
def api_guide_download():
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    if not os.path.exists(_GUIDE_PATH):
        return jsonify({"ok": False, "message": "Guide not available."}), 404
    return send_file(_GUIDE_PATH,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True, download_name="IRIS_Handbook_Access_Guide.xlsx")


@api_bp.post("/data/download")
def api_data_download():
    excel_file = brain.generate_excel(_filters_from_request())
    if excel_file:
        return send_file(
            excel_file,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="IRIS_Financial_Report.xlsx",
        )
    return jsonify({"ok": False, "message": "No data found for these filters."}), 400


# ----------------------------------------------------------------------------
# COMPLIANCE
# ----------------------------------------------------------------------------
@api_bp.get("/compliance")
def api_compliance():
    years = brain.get_compliance_years()
    selected_year = request.args.get("year")
    companies = brain.get_compliance_dashboard(target_year=selected_year)
    return jsonify({
        "companies": companies,
        "years": years,
        "active_year": selected_year if selected_year else "Latest",
    })


# ----------------------------------------------------------------------------
# ANALYTICS  (ports the computation block from analytics_dashboard, unchanged)
# ----------------------------------------------------------------------------
@api_bp.get("/analytics")
def api_analytics():
    m = _app
    raw_logs = []
    try:
        raw_logs = m.SystemLog.query.order_by(m.SystemLog.id.desc()).all()
    except Exception as e:
        print(f"DB Error in Analytics: {e}")

    module_logs = [r for r in raw_logs if r.endpoint in m.TRACKED_MODULE_ENDPOINTS]
    valid_logs = [r for r in raw_logs if r.endpoint != "/favicon.ico"]

    total_requests = len(module_logs)
    unique_users = m.db.session.query(
        m.db.func.count(m.db.func.distinct(m.db.func.lower(m.User.email)))).scalar() or 0
    errors = [r for r in valid_logs if (r.status or 0) >= 500]
    can_view_crash_details = bool(getattr(m.current_user, "is_admin", False))
    error_count = len(errors) if can_view_crash_details else 0
    visible_errors = [
        {"timestamp": m._format_dt_local(r.timestamp), "endpoint": r.endpoint, "error": r.error_msg}
        for r in errors
    ] if can_view_crash_details else []

    endpoints = {}
    for r in module_logs:
        label = m.TRACKED_MODULE_ENDPOINTS.get(r.endpoint, r.endpoint.lstrip("/").replace("_", " ").title())
        endpoints[label] = endpoints.get(label, 0) + 1
    chart_labels = list(endpoints.keys())
    chart_data = list(endpoints.values())

    current_year = datetime.now().year
    monthly_labels = [datetime(current_year, mo, 1).strftime("%b %Y") for mo in range(1, 13)]
    monthly_data = [0] * 12
    yearly_stats = []
    try:
        if module_logs:
            df = pd.DataFrame([{"dt": r.timestamp, "endpoint": r.endpoint}
                               for r in module_logs if r.timestamp is not None])
            df["dt"] = pd.to_datetime(df["dt"])
            df_this_year = df[df["dt"].dt.year == current_year]
            counts = df_this_year["dt"].dt.month.value_counts()
            for month_num, count in counts.items():
                if 1 <= month_num <= 12:
                    monthly_data[month_num - 1] = int(count)
            yearly_counts = df["dt"].dt.year.value_counts().sort_index()
            yearly_stats = [{"year": int(y), "count": int(c)} for y, c in yearly_counts.items()]
    except Exception as e:
        print(f"Analytics Data Processing Error: {e}")

    return jsonify({
        "stats": {"total": total_requests, "users": unique_users, "errors": error_count},
        "logs": visible_errors,
        "can_view_crash_details": can_view_crash_details,
        "chart": {"labels": chart_labels, "data": chart_data},
        "monthly": {"labels": monthly_labels, "data": monthly_data},
        "yearly": yearly_stats,
    })


@api_bp.post("/clear-logs")
def api_clear_logs():
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    try:
        m.SystemLog.query.filter(
            (m.SystemLog.status >= 500) | (m.SystemLog.endpoint == "/favicon.ico")
        ).delete(synchronize_session=False)
        m.db.session.commit()
    except Exception as e:
        print(f"Error clearing logs: {e}")
        return jsonify({"ok": False}), 500
    return jsonify({"ok": True})


# ----------------------------------------------------------------------------
# ADMIN
# ----------------------------------------------------------------------------
@api_bp.get("/admin/overview")
def api_admin_overview():
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    usage_insights = m._collect_admin_usage_insights()

    reset_audit = []
    try:
        rows = (m.PasswordResetAudit.query.with_entities(
            m.PasswordResetAudit.email, m.PasswordResetAudit.reset_link,
            m.PasswordResetAudit.requested_at, m.PasswordResetAudit.expires_at)
            .order_by(m.PasswordResetAudit.id.desc()).limit(20).all())
        reset_audit = [{"email": r.email, "reset_link": r.reset_link,
                        "requested_at": m._format_dt_local(r.requested_at),
                        "expires_at": m._format_dt_local(r.expires_at)} for r in rows]
    except Exception as e:
        print(f"reset audit load error: {e}")

    users = m.User.query.order_by(m.User.created_at.desc()).all()
    user_rows = [{"id": u.id, "email": u.email, "is_active": u.is_active,
                  "is_admin": u.is_admin,
                  "role": ("admin" if u.is_admin else (getattr(u, "role", "viewer") or "viewer")),
                  "created_at": m._format_dt_local(u.created_at),
                  "display_name": getattr(u, "display_name", None),
                  "device_count": m._get_active_device_count(u.id)} for u in users]

    audit_logs = []
    try:
        rows = (m.AdminAuditLog.query.with_entities(
            m.AdminAuditLog.email, m.AdminAuditLog.action_type,
            m.AdminAuditLog.status, m.AdminAuditLog.timestamp)
            .order_by(m.AdminAuditLog.id.desc()).limit(100).all())
        audit_logs = [{"email": r.email, "action_type": r.action_type, "status": r.status,
                       "timestamp": m._format_dt_local(r.timestamp)} for r in rows]
    except Exception as e:
        print(f"audit log load error: {e}")

    feedback_filter = (request.args.get("feedback_type") or "ALL").strip()
    feedback_rows = []
    try:
        feedback_rows = _feedback_with_comments(category=feedback_filter)
    except Exception as e:
        print(f"feedback load error: {e}")

    flags = []
    try:
        flags = _flags_with_comments()
    except Exception as e:
        print(f"flags load error: {e}")

    search_logs = []
    try:
        rows = (m.SearchLog.query
                .order_by(m.SearchLog.id.desc()).limit(100).all())
        search_logs = [{"user_email": r.user_email, "module": r.module, "query": r.query_text,
                        "result_count": r.result_count, "timestamp": m._format_dt_local(r.timestamp)}
                       for r in rows]
    except Exception as e:
        print(f"search log load error: {e}")

    announcements = []
    try:
        rows = m.Announcement.query.order_by(m.Announcement.id.desc()).limit(50).all()
        announcements = [{"id": a.id, "title": a.title, "body": a.body, "level": a.level,
                          "active": a.active, "created_by": a.created_by,
                          "created_at": m._format_dt_local(a.created_at)} for a in rows]
    except Exception as e:
        print(f"announcement load error: {e}")

    return jsonify({
        "sync_state": m.SYNC_STATE,
        "usage_insights": usage_insights,
        "reset_audit": reset_audit,
        "users": user_rows,
        "audit_logs": audit_logs,
        "feedback_rows": feedback_rows,
        "feedback_filter": feedback_filter,
        "search_logs": search_logs,
        "announcements": announcements,
        "flags": flags,
    })


@api_bp.post("/admin/create-user")
def api_admin_create_user():
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    role = (data.get("role") or "").strip().lower()
    if role not in _ROLE_RANK:
        # back-compat: is_admin flag -> admin, else viewer
        role = "admin" if str(data.get("is_admin") or "").lower() in {"1", "true", "yes", "on"} else "viewer"
    if not m._is_allowed_email(email):
        return jsonify({"message": "Email must use @irdai.gov.in domain."}), 400
    if len(password) < 8:
        return jsonify({"message": "Password must be at least 8 characters."}), 400
    if m.User.query.filter(m.db.func.lower(m.User.email) == email).first():
        return jsonify({"message": "User already exists."}), 409
    m.db.session.add(m.User(email=email, password_hash=m.generate_password_hash(password),
                            is_active=True, is_admin=(role == "admin"), role=role))
    m.db.session.commit()
    m._record_admin_audit(email, f"user_create ({role})", "success")
    return jsonify({"ok": True, "message": "User created successfully."}), 201


@api_bp.post("/admin/user/<int:user_id>/role")
def api_admin_set_role(user_id):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    role = ((request.get_json(silent=True) or {}).get("role") or "").strip().lower()
    if role not in _ROLE_RANK:
        return jsonify({"message": "Invalid role"}), 400
    user = m.User.query.get_or_404(user_id)
    user.role = role
    user.is_admin = (role == "admin")
    m.db.session.commit()
    m._record_admin_audit(user.email, f"role_set:{role}", "success")
    return jsonify({"ok": True, "role": role})


@api_bp.post("/admin/user/<int:user_id>/toggle-active")
def api_admin_toggle_active(user_id):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    user = m.User.query.get_or_404(user_id)
    user.is_active = not user.is_active
    m.db.session.commit()
    m._record_admin_audit(user.email, "user_activate" if user.is_active else "user_deactivate", "success")
    return jsonify({"ok": True, "is_active": user.is_active})


@api_bp.post("/admin/user/<int:user_id>/delete")
def api_admin_delete_user(user_id):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    user = m.User.query.get_or_404(user_id)
    if user.id == m.current_user.id:
        return jsonify({"ok": False, "message": "Cannot delete yourself."}), 400
    try:
        # Clear FK-dependent rows first so the delete succeeds on Postgres too.
        m._purge_user_dependents(user.id)
        email = user.email
        m.db.session.delete(user)
        m.db.session.commit()
        m._record_admin_audit(email, "user_delete", "success")
    except Exception as e:
        m.db.session.rollback()
        print(f"user delete error: {e}")
        return jsonify({"ok": False, "message": "Could not delete user."}), 500
    return jsonify({"ok": True})


@api_bp.post("/admin/user/<int:user_id>/logout-all")
def api_admin_logout_user(user_id):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    user = m.User.query.get_or_404(user_id)
    user.session_version = (user.session_version or 0) + 1
    m.db.session.commit()
    m._deactivate_all_user_sessions(user.id)
    m._record_admin_audit(user.email, "logout_all_devices", "success")
    return jsonify({"ok": True})


@api_bp.post("/admin/user/<int:user_id>/trigger-reset")
def api_admin_trigger_reset(user_id):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    import secrets
    from datetime import timedelta
    user = m.User.query.get_or_404(user_id)
    raw_token = secrets.token_urlsafe(32)
    user.reset_token = m._hash_reset_token(raw_token)
    user.reset_token_expiry = datetime.utcnow() + timedelta(minutes=15)
    m.db.session.commit()
    reset_link = request.host_url.rstrip("/") + "/reset-password/" + raw_token
    try:
        m.db.session.add(m.PasswordResetAudit(
            email=user.email.lower(), reset_link=reset_link,
            requested_at=datetime.utcnow(), expires_at=user.reset_token_expiry))
        m.db.session.commit()
    except Exception:
        pass
    m._record_admin_audit(user.email, "password_reset_request", "success")
    return jsonify({"ok": True, "reset_link": reset_link})


@api_bp.post("/admin/clear-reset-audit")
def api_admin_clear_reset_audit():
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    m.PasswordResetAudit.query.delete()
    m.db.session.commit()
    m._record_admin_audit(m.current_user.email, "clear_reset_audit", "success")
    return jsonify({"ok": True})


@api_bp.post("/admin/clear-search-logs")
def api_admin_clear_search_logs():
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    try:
        m.SearchLog.query.delete()
        m.db.session.commit()
        m._record_admin_audit(m.current_user.email, "clear_search_logs", "success")
    except Exception as e:
        print(f"clear search logs error: {e}")
        return jsonify({"ok": False}), 500
    return jsonify({"ok": True})


@api_bp.post("/admin/clear-audit-logs")
def api_admin_clear_audit_logs():
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    m.AdminAuditLog.query.delete()
    m.db.session.commit()
    m._record_admin_audit(m.current_user.email, "clear_admin_audit_logs", "success")
    return jsonify({"ok": True})


@api_bp.post("/admin/sync_start")
def api_admin_sync_start():
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    import threading
    if m.SYNC_STATE["status"] == "running":
        return jsonify({"status": "error", "message": "Sync already in progress."})
    m.SYNC_STATE["status"] = "starting"
    m.SYNC_STATE["message"] = "Initializing background process..."
    t = threading.Thread(target=m.run_background_sync)
    t.daemon = True
    t.start()
    return jsonify({"status": "started"})


@api_bp.get("/admin/sync_status")
def api_admin_sync_status():
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    return jsonify(_app.SYNC_STATE)


# ----------------------------------------------------------------------------
# PROFILE
# ----------------------------------------------------------------------------
def _feedback_with_comments(user_id=None, category=None):
    """Serialize feedback rows (optionally one user / category) with comment threads."""
    m = _app
    q = (m.db.session.query(m.FeedbackEntry, m.User.email)
         .join(m.User, m.User.id == m.FeedbackEntry.user_id)
         .order_by(m.FeedbackEntry.id.desc()))
    if user_id is not None:
        q = q.filter(m.FeedbackEntry.user_id == user_id)
    if category in {"Bug", "Suggestion", "UI Issue", "Other (please specify)"}:
        q = q.filter(m.FeedbackEntry.category == category)
    pairs = q.limit(200).all()
    rows = [f for f, _ in pairs]
    ids = [f.id for f in rows]
    comments_by_fid = {}
    if ids:
        for c in (m.FeedbackComment.query
                  .filter(m.FeedbackComment.feedback_id.in_(ids))
                  .order_by(m.FeedbackComment.id.asc()).all()):
            comments_by_fid.setdefault(c.feedback_id, []).append({
                "author_email": c.author_email, "is_admin": bool(c.is_admin),
                "body": c.body, "created_at": m._format_dt_local(c.created_at),
            })
    return [{
        "id": f.id, "category": f.category, "message": f.message,
        "user_email": ue,
        "status": _norm_status(getattr(f, "status", "Open")),
        "created_at": m._format_dt_local(f.created_at),
        "comments": comments_by_fid.get(f.id, []),
    } for f, ue in pairs]


# Canonical statuses are Open / Resolved / Ignored (shared by feedback + flags).
# Map legacy values so old rows display consistently.
def _norm_status(s):
    s = (s or "Open").strip()
    return {"Done": "Resolved", "Dismissed": "Ignored"}.get(s, s) or "Open"


def _flags_with_comments(user_email=None):
    """Serialize flags (optionally one user's) with comment threads + normalized status."""
    m = _app
    q = m.Flag.query.order_by(m.Flag.id.desc())
    if user_email is not None:
        q = q.filter(m.Flag.user_email == user_email)
    rows = q.limit(200).all()
    ids = [f.id for f in rows]
    comments_by_id = {}
    if ids:
        try:
            for c in (m.FlagComment.query.filter(m.FlagComment.flag_id.in_(ids))
                      .order_by(m.FlagComment.id.asc()).all()):
                comments_by_id.setdefault(c.flag_id, []).append({
                    "author_email": c.author_email, "is_admin": bool(c.is_admin),
                    "body": c.body, "created_at": m._format_dt_local(c.created_at)})
        except Exception as e:
            print(f"flag comments load error: {e}")
    return [{
        "id": f.id, "user_email": f.user_email, "kind": f.kind, "reason": f.reason,
        "description": f.description, "target": f.target,
        "status": _norm_status(f.status),
        "created_at": m._format_dt_local(f.created_at),
        "has_screenshot": bool(getattr(f, "screenshot", None)),
        "comments": comments_by_id.get(f.id, []),
    } for f in rows]


@api_bp.get("/profile")
def api_profile():
    if not _app.current_user.is_authenticated:
        return jsonify({"authenticated": False}), 401
    cu = _app.current_user
    flags = []
    try:
        flags = _flags_with_comments(user_email=cu.email)
    except Exception as e:
        print(f"profile flags load error: {e}")
    return jsonify({
        "email": cu.email,
        "display_name": getattr(cu, "display_name", None),
        "avatar": getattr(cu, "avatar", None),
        "is_admin": bool(getattr(cu, "is_admin", False)),
        "sessions": _app._list_user_sessions(cu.id),
        "feedback": _feedback_with_comments(cu.id),
        "flags": flags,
    })


@api_bp.post("/profile")
def api_profile_update():
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    m = _app
    data = request.get_json(silent=True) or request.form
    user = m.db.session.get(m.User, m.current_user.id)
    if "display_name" in data:
        name = (data.get("display_name") or "").strip()
        user.display_name = name[:120] or None
    if "avatar" in data:
        avatar = data.get("avatar") or ""
        # Small avatars only (client resizes/compresses to a data URL). Cap size.
        if avatar and not (avatar.startswith("data:image/") and len(avatar) <= 200000):
            return jsonify({"ok": False, "message": "Image too large. Please use a smaller photo."}), 400
        user.avatar = avatar or None
    m.db.session.commit()
    return jsonify({"ok": True, "user": _user_payload()})


@api_bp.post("/profile/password")
def api_profile_password():
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    m = _app
    data = request.get_json(silent=True) or request.form
    current_password = data.get("current_password") or ""
    new_password = data.get("new_password") or ""
    confirm_password = data.get("confirm_password") or ""
    if not m.check_password_hash(m.current_user.password_hash, current_password):
        return jsonify({"ok": False, "message": "Current password is incorrect."}), 400
    if len(new_password) < 8:
        return jsonify({"ok": False, "message": "New password must be at least 8 characters."}), 400
    if new_password != confirm_password:
        return jsonify({"ok": False, "message": "New password and confirm password do not match."}), 400
    user = m.db.session.get(m.User, m.current_user.id)
    user.password_hash = m.generate_password_hash(new_password)
    user.session_version = (user.session_version or 0) + 1
    m.db.session.commit()
    m._deactivate_all_user_sessions(user.id)
    m._record_admin_audit(user.email, "password_change", "success")
    m.logout_user()
    session.clear()
    return jsonify({"ok": True, "message": "Password changed. Please login again."})


@api_bp.post("/profile/session/<int:session_id>/logout")
def api_profile_session_logout(session_id):
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    m = _app
    m._deactivate_session_by_id(m.current_user.id, session_id)
    forced = False
    if session.get("auth_token") and not m._is_session_active(m.current_user.id, session.get("auth_token")):
        m.logout_user()
        session.clear()
        forced = True
    return jsonify({"ok": True, "logged_out": forced})


# ----------------------------------------------------------------------------
# FEEDBACK
# ----------------------------------------------------------------------------
@api_bp.post("/feedback")
def api_feedback():
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    m = _app
    data = request.get_json(silent=True) or request.form
    category = (data.get("category") or "Suggestion").strip()
    message = (data.get("message") or "").strip()
    if category not in {"Bug", "Suggestion", "UI Issue", "Other (please specify)"}:
        category = "Suggestion"
    if len(message) < 5:
        return jsonify({"ok": False, "message": "Please enter at least 5 characters."}), 400
    m.db.session.add(m.FeedbackEntry(user_id=m.current_user.id, category=category,
                                     message=message, status="Open", created_at=datetime.utcnow()))
    m.db.session.commit()
    return jsonify({"ok": True, "message": "Thanks! Your feedback has been submitted."})


@api_bp.post("/admin/feedback/<int:fid>/status")
def api_admin_feedback_status(fid):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    data = request.get_json(silent=True) or request.form
    status = (data.get("status") or "").strip().capitalize()
    if status not in {"Open", "Resolved", "Ignored"}:
        return jsonify({"ok": False, "message": "Invalid status."}), 400
    entry = m.FeedbackEntry.query.get_or_404(fid)
    entry.status = status
    m.db.session.commit()
    return jsonify({"ok": True, "status": status})


@api_bp.post("/admin/feedback/<int:fid>/delete")
def api_admin_feedback_delete(fid):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    entry = m.FeedbackEntry.query.get_or_404(fid)
    m.FeedbackComment.query.filter_by(feedback_id=fid).delete(synchronize_session=False)
    m.db.session.delete(entry)
    m.db.session.commit()
    return jsonify({"ok": True})


@api_bp.post("/feedback/<int:fid>/delete")
def api_feedback_self_delete(fid):
    """A user may retract/delete their own feedback (admins may delete any)."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    m = _app
    entry = m.FeedbackEntry.query.get_or_404(fid)
    is_admin = bool(getattr(m.current_user, "is_admin", False))
    if entry.user_id != m.current_user.id and not is_admin:
        return jsonify({"ok": False, "message": "Not allowed."}), 403
    m.FeedbackComment.query.filter_by(feedback_id=fid).delete(synchronize_session=False)
    m.db.session.delete(entry)
    m.db.session.commit()
    return jsonify({"ok": True})


@api_bp.post("/flag/<int:flag_id>/delete")
def api_flag_self_delete(flag_id):
    """A user may retract/delete their own flag (admins may delete any)."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    m = _app
    flag = m.Flag.query.get_or_404(flag_id)
    is_admin = bool(getattr(m.current_user, "is_admin", False))
    if (flag.user_email or "") != m.current_user.email and not is_admin:
        return jsonify({"ok": False, "message": "Not allowed."}), 403
    m.FlagComment.query.filter_by(flag_id=flag_id).delete(synchronize_session=False)
    m.db.session.delete(flag)
    m.db.session.commit()
    return jsonify({"ok": True})


@api_bp.post("/feedback/<int:fid>/comment")
def api_feedback_comment(fid):
    """Follow-up comment on a feedback entry. Owners and admins may post."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    m = _app
    entry = m.FeedbackEntry.query.get_or_404(fid)
    is_admin = bool(getattr(m.current_user, "is_admin", False))
    if entry.user_id != m.current_user.id and not is_admin:
        return jsonify({"ok": False, "message": "Not allowed."}), 403
    data = request.get_json(silent=True) or request.form
    body = (data.get("body") or "").strip()
    if len(body) < 2:
        return jsonify({"ok": False, "message": "Please enter a comment."}), 400
    m.db.session.add(m.FeedbackComment(feedback_id=fid, author_email=m.current_user.email,
                                       is_admin=is_admin, body=body[:2000],
                                       created_at=datetime.utcnow()))
    m.db.session.commit()
    return jsonify({"ok": True})


# ----------------------------------------------------------------------------
# FLAGS  (user-reported issues on a clause or a financial data row)
# ----------------------------------------------------------------------------
_FLAG_REASONS = {"Wrong information", "Outdated / superseded", "Wrong document/source",
                 "Formatting issue", "Other",
                 # No-result (no clauses found) reasons:
                 "Relevant clause exists but was not found",
                 "Content missing from the knowledge base",
                 "Document not loaded for this module",
                 "Searched the wrong department/module"}


@api_bp.post("/flag")
def api_flag_create():
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    m = _app
    data = request.get_json(silent=True) or request.form
    kind = (data.get("kind") or "clause").strip()
    if kind not in {"clause", "financial"}:
        kind = "clause"
    reason = (data.get("reason") or "Other").strip()
    if reason not in _FLAG_REASONS:
        reason = "Other"
    description = (data.get("description") or "").strip()[:2000] or None
    target = (data.get("target") or "").strip()[:300] or None
    detail = (data.get("detail") or "").strip()[:4000] or None
    # Optional screenshot of the user's screen (base64 PNG data URL). Capped so a
    # single flag can't bloat the DB / Litestream replica.
    screenshot = data.get("screenshot") or ""
    if not (isinstance(screenshot, str) and screenshot.startswith("data:image/")
            and len(screenshot) <= 6_000_000):
        screenshot = None
    m.db.session.add(m.Flag(user_email=m.current_user.email, kind=kind, reason=reason,
                            description=description, target=target, detail=detail,
                            screenshot=screenshot,
                            status="Open", created_at=datetime.utcnow()))
    m.db.session.commit()
    return jsonify({"ok": True, "message": "Thanks — flag submitted for review."}), 201


@api_bp.get("/flag/<int:flag_id>/screenshot")
def api_flag_screenshot(flag_id):
    """Serve a flag's captured screenshot (admin, or the flag's own author)."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    m = _app
    f = m.Flag.query.get(flag_id)
    if not f or not f.screenshot:
        return jsonify({"ok": False, "message": "No screenshot"}), 404
    is_admin = getattr(m.current_user, "is_admin", False)
    if not is_admin and f.user_email != m.current_user.email:
        return jsonify({"ok": False}), 403
    try:
        header, b64 = f.screenshot.split(",", 1)
        mime = header.split(";")[0].replace("data:", "") or "image/png"
        return Response(base64.b64decode(b64), mimetype=mime)
    except Exception:
        return jsonify({"ok": False, "message": "Corrupt screenshot"}), 500


@api_bp.post("/admin/flag/<int:flag_id>/status")
def api_admin_flag_status(flag_id):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    data = request.get_json(silent=True) or request.form
    status = (data.get("status") or "").strip().capitalize()
    if status not in {"Open", "Resolved", "Ignored"}:
        return jsonify({"ok": False, "message": "Invalid status."}), 400
    flag = m.Flag.query.get_or_404(flag_id)
    flag.status = status
    m.db.session.commit()
    return jsonify({"ok": True, "status": status})


@api_bp.post("/admin/flag/<int:flag_id>/delete")
def api_admin_flag_delete(flag_id):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    flag = m.Flag.query.get_or_404(flag_id)
    m.FlagComment.query.filter_by(flag_id=flag_id).delete(synchronize_session=False)
    m.db.session.delete(flag)
    m.db.session.commit()
    return jsonify({"ok": True})


@api_bp.post("/flag/<int:flag_id>/comment")
def api_flag_comment(flag_id):
    """Follow-up comment on a flag. The flag's owner and admins may post."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    m = _app
    flag = m.Flag.query.get_or_404(flag_id)
    is_admin = bool(getattr(m.current_user, "is_admin", False))
    if (flag.user_email or "") != m.current_user.email and not is_admin:
        return jsonify({"ok": False, "message": "Not allowed."}), 403
    data = request.get_json(silent=True) or request.form
    body = (data.get("body") or "").strip()
    if len(body) < 2:
        return jsonify({"ok": False, "message": "Please enter a comment."}), 400
    m.db.session.add(m.FlagComment(flag_id=flag_id, author_email=m.current_user.email,
                                   is_admin=is_admin, body=body[:2000], created_at=datetime.utcnow()))
    m.db.session.commit()
    return jsonify({"ok": True})


# ----------------------------------------------------------------------------
# ANNOUNCEMENTS / COMMUNICATION
# ----------------------------------------------------------------------------
@api_bp.get("/announcements")
def api_announcements():
    """Active announcements for the logged-in user's notification bell."""
    if not _app.current_user.is_authenticated:
        return jsonify({"announcements": []}), 401
    m = _app
    items = []
    try:
        rows = (m.Announcement.query.filter_by(active=True)
                .order_by(m.Announcement.id.desc()).limit(30).all())
        items = [{"id": a.id, "title": a.title, "body": a.body, "level": a.level,
                  "created_at": m._format_dt_local(a.created_at)} for a in rows]
    except Exception as e:
        print(f"announcement feed error: {e}")
    return jsonify({"announcements": items})


@api_bp.post("/admin/announcement")
def api_admin_announcement_create():
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    data = request.get_json(silent=True) or request.form
    title = (data.get("title") or "").strip()
    body = (data.get("body") or "").strip()
    level = (data.get("level") or "info").strip().lower()
    if level not in {"info", "success", "warning"}:
        level = "info"
    if len(title) < 3:
        return jsonify({"ok": False, "message": "Title must be at least 3 characters."}), 400
    if len(body) < 3:
        return jsonify({"ok": False, "message": "Message must be at least 3 characters."}), 400
    m.db.session.add(m.Announcement(title=title[:200], body=body, level=level,
                                    created_by=m.current_user.email, active=True,
                                    created_at=datetime.utcnow()))
    m.db.session.commit()
    m._record_admin_audit(m.current_user.email, "announcement_create", "success")
    return jsonify({"ok": True}), 201


@api_bp.post("/admin/announcement/<int:aid>/toggle")
def api_admin_announcement_toggle(aid):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    a = m.Announcement.query.get_or_404(aid)
    a.active = not a.active
    m.db.session.commit()
    return jsonify({"ok": True, "active": a.active})


@api_bp.post("/admin/announcement/<int:aid>/delete")
def api_admin_announcement_delete(aid):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    a = m.Announcement.query.get_or_404(aid)
    m.db.session.delete(a)
    m.db.session.commit()
    return jsonify({"ok": True})
