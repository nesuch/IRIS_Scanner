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
import bisect
import difflib
import io
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
from flask import Blueprint, request, jsonify, session, send_file, Response, g

import iris_brain as brain
import storage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from pq_to_iris import parse_docx, parse_pq, _iso_date  # noqa: E402

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
# Brute-force throttle for auth endpoints. In-process + thread-safe, which is
# sufficient for the single-worker gunicorn deployment (one shared process). We
# track failures by both client IP and target email, and block if either trips
# the threshold, so neither IP-rotation nor email-rotation alone defeats it.
# ----------------------------------------------------------------------------
_AUTH_LOCK = threading.Lock()
_auth_fails = {}                 # key -> [unix timestamps of recent failures]
_AUTH_WINDOW = 900               # 15 minutes
_AUTH_MAX = 8                    # failures per window before lockout

def _client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else "") or request.remote_addr or "?"

def _auth_keys(email):
    keys = ["ip:" + _client_ip()]
    if email:
        keys.append("em:" + email)
    return keys

def _auth_blocked(email):
    now = time.time()
    with _AUTH_LOCK:
        for k in _auth_keys(email):
            recent = [t for t in _auth_fails.get(k, []) if now - t < _AUTH_WINDOW]
            _auth_fails[k] = recent
            if len(recent) >= _AUTH_MAX:
                return True
    return False

def _auth_record_fail(email):
    now = time.time()
    with _AUTH_LOCK:
        for k in _auth_keys(email):
            _auth_fails.setdefault(k, []).append(now)
        if len(_auth_fails) > 5000:          # bound memory: drop empty/expired keys
            for k in list(_auth_fails):
                _auth_fails[k] = [t for t in _auth_fails[k] if now - t < _AUTH_WINDOW]
                if not _auth_fails[k]:
                    del _auth_fails[k]

def _auth_clear(email):
    with _AUTH_LOCK:
        for k in _auth_keys(email):
            _auth_fails.pop(k, None)


# Password policy: min 8 (ASVS L1), and a hard max so an attacker can't feed a
# megabyte-long password into the (deliberately slow) hash to burn CPU.
_PW_MIN = 8
_PW_MAX = 128

# A throwaway hash used to spend the same CPU on a non-existent/ineligible account
# as on a real one, so login response time can't reveal whether an email exists.
from werkzeug.security import generate_password_hash as _wz_gen, check_password_hash as _wz_check
_DUMMY_HASH = _wz_gen("timing-equalization-dummy-not-a-real-password")

def _password_error(pw, confirm):
    if len(pw) < _PW_MIN:
        return f"Password must be at least {_PW_MIN} characters."
    if len(pw) > _PW_MAX:
        return f"Password must be at most {_PW_MAX} characters."
    if pw != confirm:
        return "Passwords do not match."
    return None


# ----------------------------------------------------------------------------
# AUTH
# ----------------------------------------------------------------------------
@api_bp.post("/login")
def api_login():
    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    next_url = _app._safe_next_url(data.get("next") or "/")

    if _auth_blocked(email):
        _app._record_admin_audit(email, "login_attempt", "rate_limited")
        return jsonify({"ok": False, "message": "Too many attempts. Please wait a few minutes and try again."}), 429

    # Over-long passwords can't be valid (policy max) but must not skip the hash
    # path, or that itself becomes a timing signal — cap to bound hash CPU.
    password = password[:_PW_MAX + 1]

    user = _app.User.query.filter(_app.db.func.lower(_app.User.email) == email).first()
    eligible = bool(user and user.is_active and _app._is_allowed_email(email))
    # Always perform one hash comparison so the response time is the same whether
    # or not the account exists/is eligible (mitigates user-enumeration via timing).
    if eligible:
        ok = _app.check_password_hash(user.password_hash, password)
    else:
        _wz_check(_DUMMY_HASH, password)
        ok = False

    if ok:
        _auth_clear(email)
        # Transparently upgrade a legacy/weaker hash to the current policy now that
        # we hold the plaintext (e.g. after a work-factor or algorithm change).
        if _app.password_needs_rehash(user.password_hash):
            user.password_hash = _app.hash_password(password)
            _app.db.session.commit()
        _app.login_user(user)
        _app._start_user_session(user)
        _app._record_admin_audit(email, "login_attempt", "success")
        return jsonify({"ok": True, "user": _user_payload(), "next": next_url})
    _auth_record_fail(email)
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
    # Throttle reset-link generation per IP to prevent spam/abuse (generic
    # response regardless, so this leaks nothing about account existence).
    if _auth_blocked(""):
        return jsonify({"ok": True, "message": generic_msg})
    if email and _app._is_allowed_email(email):
        _auth_record_fail("")
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
    err = _password_error(new_password, confirm_password)
    if err:
        return jsonify({"ok": False, "message": err}), 400
    user.password_hash = _app.hash_password(new_password)
    user.reset_token = None
    user.reset_token_expiry = None
    # Invalidate every existing session for this account — a reset is often a
    # response to compromise, so any logged-in attacker must be kicked out too.
    user.session_version = (user.session_version or 0) + 1
    _app.db.session.commit()
    _app._deactivate_all_user_sessions(user.id)
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
def _asset_map():
    """All DocumentAssets keyed by source_doc, loaded ONCE per request.

    _doc_pdf_url and _doc_bundle are called per clause, and a search renders up to
    70 of them — that was 70 (now 140) single-row queries to answer a question
    about at most a handful of distinct documents. flask.g scopes the cache to the
    request, so an upload in the same process is still seen by the next one.
    """
    m = getattr(g, "_iris_assets", None)
    if m is None:
        try:
            m = {str(a.source_doc): a for a in _app.DocumentAsset.query.all()}
        except Exception:
            m = {}
        g._iris_assets = m
    return m


def _doc_pdf_url(source):
    """Prefer an admin-attached PDF for this document; else the bundled static PDF.
    Includes a version param so a replaced PDF busts the viewer/HTTP cache."""
    asset = _asset_map().get(str(source))
    if asset and asset.pdf_filename:
        v = int(asset.uploaded_at.timestamp()) if asset.uploaded_at else 0
        return f"/api/doc-pdf/{asset.id}/download?v={v}"
    p = _app.resolve_pdf_path(str(source).strip().upper())
    return ("/static/" + p) if p else None


# Extension -> content type for supplementary bundles. Anything unlisted streams
# as octet-stream, which browsers download rather than try to render.
_BUNDLE_TYPES = {
    ".zip": "application/zip",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".csv": "text/csv",
}


def _doc_bundle(source):
    """{url, name} for a document's supplementary bundle, else None.

    Separate from _doc_pdf_url because the two are consumed differently: the PDF
    feeds an inline viewer, the bundle is download-only. Every clause of a
    document with a bundle offers it, so an officer reading one clause can pull
    the annexure forms without navigating anywhere.
    """
    a = _asset_map().get(str(source))
    if not a or not a.bundle_filename:
        return None
    ts = a.bundle_uploaded_at or a.uploaded_at
    v = int(ts.timestamp()) if ts else 0
    return {"url": f"/api/doc-bundle/{a.id}/download?v={v}", "name": a.bundle_filename}


def _make_scorer(phrase_l, core_roots, multi, run_len_cap=None,
                 root_keys=None, known_keys=None, root_masks=None):
    """Query-derived relevance scoring: returns (score, promote).

    Lifted out of api_search so a second corpus (PQ replies) can rank by the same
    rules. Deliberately shape-agnostic — it reads only m["raw_text"], never a
    DataFrame — which is what makes it shareable at all.

    run_len_cap suspends the run scan for long queries. _run_len is O(n^2) in the
    query and scans the whole text: ~0.1ms on a 25k body for 3 words but ~27ms for
    13, so a back-catalogue of PQ bodies would spend seconds in it. Clause callers
    pass None (clauses average 1,894 chars and are already capped at ~80 candidates).

    root_keys is the prefix index pre-resolved by the caller into
    {root: set of (source, clause_id)} — the same identity the score memo keys on —
    and known_keys is the full set of keys the index covers. Together they replace
    the per-(root, clause) regex in `cover`, which the pre-ranking pass otherwise
    runs over EVERY candidate.

    Both are optional and per-call on purpose. PQ builds its own scorer without
    them, because PQ rows are not clauses and would miss every set — silently
    scoring zero coverage. known_keys makes that failure mode impossible in general:
    any match not in the indexed universe falls back to the regex instead of being
    treated as "absent everywhere".
    """
    phrase_join = phrase_l.replace("-", "")
    # The query's content words IN ORDER (stopwords dropped) — used to score the
    # longest CONSECUTIVE partial-phrase run a clause contains. For a long natural-
    # language query no clause holds every word, so a clause with a real 3-word run
    # ("acceptance of a proposal") must outrank one matching a single common word
    # ("before"). Adjacent query words may be separated by up to 2 filler words in
    # the clause (articles/prepositions), so "acceptance of the proposal" matches
    # "acceptance of a proposal".
    # Content words only. FUNCTION_WORDS, not just STOPWORDS_STRONG (9 words): a
    # preposition like "from" is not a term the user is searching for, and counting
    # it here does double damage — it becomes a link the run has to match, while the
    # `sep` below is already designed to STEP OVER exactly such filler. It also
    # inflates len(_qseq), which raises the promotion bar for the whole query.
    _qseq = [brain.search_root(w) for w in re.findall(r"\w+", phrase_l)
             if len(w) > 2 and w not in brain.STOPWORDS_STRONG
             and w not in brain.FUNCTION_WORDS]
    _scan_runs = multi and not (run_len_cap is not None and len(_qseq) > run_len_cap)

    def _run_len(t):
        n = len(_qseq)
        if n < 2:
            return 0
        best = 0
        sep = r"\W+(?:\w+\W+){0,2}"
        for i in range(n):
            for j in range(n, i + 1, -1):
                if j - i <= best:
                    break
                pat = sep.join(rf"\b{re.escape(r)}\w*" for r in _qseq[i:j])
                if re.search(pat, t):
                    best = j - i
                    break
        return best

    # Proximity scan, precompiled. The obvious implementation — for each sentence,
    # test each root — is O(roots x sentences) regex calls per clause and profiled at
    # 1.05M regex searches (2.7s) for a 13-word query, because re.escape and the
    # pattern cache are re-entered on every one. This walks each root's matches over
    # the whole clause ONCE and buckets them into sentences by offset, so the cost is
    # O(roots) passes regardless of how long the clause is.
    _root_res = [re.compile(rf"\b{re.escape(r)}\w*") for r in core_roots]
    _SENT_SPLIT = brain.SENT_SPLIT   # shared with the index, which bakes it in at load

    def _line_cover(t):
        """The most distinct query roots any ONE sentence of the clause holds."""
        if len(core_roots) < 2:
            return 0
        starts, pos = [0], 0
        for mm in _SENT_SPLIT.finditer(t):
            starts.append(mm.end())
        seen = [0] * len(starts)          # bitmask of roots seen per sentence
        best = 0
        for ri, rx in enumerate(_root_res):
            bit = 1 << ri
            for mm in rx.finditer(t):
                i = bisect.bisect_right(starts, mm.start()) - 1
                if i >= 0 and not (seen[i] & bit):
                    seen[i] |= bit
                    n = bin(seen[i]).count("1")
                    if n > best:
                        best = n
                        if best == len(core_roots):
                            return best
        return best

    def _line_cover_idx(k):
        """Proximity from the prefix index: each root carries a bitmask of the
        sentences it occurs in, so the answer is the highest number of roots sharing
        any single bit. Returns None when the index cannot answer for this row (or
        for any one root), and the caller falls back to the scan above."""
        if root_masks is None or known_keys is None or k not in known_keys:
            return None
        if len(core_roots) < 2:
            return 0
        ms = []
        for r in core_roots:
            d = root_masks.get(r)
            if d is None:
                return None
            ms.append(d.get(k, 0))
        bits = 0
        for m in ms:
            bits |= m
        best = 0
        while bits:
            b = bits & -bits          # lowest set bit == one sentence
            n = 0
            for m in ms:
                if m & b:
                    n += 1
            if n > best:
                best = n
                if best == len(core_roots):
                    return best
            bits ^= b
        return best

    _scores = {}

    def _cover(k, t, tj):
        """How many distinct query roots the clause holds. Served from the prefix
        index when the caller supplied one AND this row is part of the indexed
        corpus; otherwise the original regex, per root, so a root the index cannot
        represent (a non-\\w compound) degrades on its own rather than poisoning the
        whole count."""
        if root_keys is None or known_keys is None or k not in known_keys:
            return sum(1 for r in core_roots if brain._root_present(r, t, tj))
        n = 0
        for r in core_roots:
            s = root_keys.get(r)
            if s is None:
                if brain._root_present(r, t, tj):
                    n += 1
            elif k in s:
                n += 1
        return n

    def _score(m):
        """Graded relevance tuple (higher = better): verbatim phrase, most typed
        words sharing ONE sentence, all typed words present, longest consecutive
        word-run, distinct-word coverage. Hyphen-insensitive."""
        # Key on the clause's own identity, NOT id(m). The pre-ranking pass scores a
        # whole candidate pool and then keeps only its head; CPython recycles the
        # addresses of the discarded dicts for the NEXT pool, so an id()-keyed memo
        # silently returns another clause's score. That made results depend on what
        # had been searched before — the same query returned a different #1 run alone
        # versus run after other queries. Match dicts without an id (PQ synthesises
        # bare {raw_text} ones) fall back to id(m), and those callers materialise
        # their list first so the addresses stay live.
        _cid = m.get("id")
        k = (m.get("source"), _cid) if _cid else id(m)
        if k in _scores:
            return _scores[k]
        t = str(m.get("raw_text", "")).lower()
        tj = t.replace("-", "")
        # The exact query text appears in the clause. For a single word this is the
        # EXACT word (e.g. "promotion" matches "promotion"/"promotional" but not the
        # stem cousin "promote"), so exact-word clauses can be promoted above
        # stem-family / loosely-tagged ones.
        verbatim = 1 if (phrase_l in t or phrase_join in tj) else 0
        cover = _cover(k, t, tj)
        allwords = 1 if (len(core_roots) >= 2 and cover == len(core_roots)) else 0
        # Proximity: the most typed words any SINGLE sentence holds. Whole-clause
        # coverage alone rewards length — a 2,600-char "general guidelines" clause
        # can contain every word the user typed, scattered across two pages, and
        # outrank the one sentence that actually answers them. This was previously a
        # binary same_line flag that required ALL words in one line, so "3 of your 4
        # words in one sentence" scored the same as none of them; it was also never
        # read by any caller.
        _lc_idx = _line_cover_idx(k)
        line_cover = _line_cover(t) if _lc_idx is None else _lc_idx
        run = _run_len(t) if _scan_runs else 0
        s = (verbatim, line_cover, allwords, run, cover)
        _scores[k] = s
        return s

    # Run threshold for promotion. A 2-word run ("the authority") is fine evidence
    # for a SHORT query but meaningless for a long one — it would promote hundreds of
    # clauses (any common consecutive pair), exploding the rendered result set. So
    # require the run to cover a real fraction of the query for longer queries; short
    # queries keep the sensitive >=2 threshold (typo robustness).
    _promote_run = 2 if len(_qseq) <= 4 else max(3, (len(_qseq) + 1) // 2)

    # Promotion on sentence coverage: MOST of the query answered in one sentence is
    # strong evidence even when a word is missing entirely. Searching "charged to
    # shareholders account", the clause saying "charged ... from shareholder's fund"
    # has two of the three words in one sentence but says "fund" where the user
    # guessed "account" — so it never qualified on verbatim/allwords/run, stayed in
    # the subordinate content tier, and was cut by its cap before anyone saw it.
    # People rarely know a regulation's exact wording, so a near miss on one word
    # should not be the difference between first page and invisible.
    # Floored at 2 (one word in a sentence proves nothing) and scaled by a majority
    # of the query, so a long natural-language query cannot promote hundreds.
    _promote_line = max(2, (len(core_roots) + 1) // 2)

    def _promote(m):
        # A verbatim phrase, ALL typed words, a substantial consecutive run, or most
        # of the query inside ONE sentence earns the top "best match" tier; a lone
        # common-word (or common-pair) hit does not.
        v, line_cover, aw, run, _c = _score(m)
        return bool(v or aw or run >= _promote_run or line_cover >= _promote_line)

    return _score, _promote


def _match_payload(m):
    return {
        "source": m.get("source", "UNKNOWN"),
        "type": m.get("type", "UNKNOWN"),
        "id": str(m.get("id", "")).strip(),
        "header": m.get("header", ""),
        "raw_text": str(m.get("raw_text", "")),
        "pdf_url": _doc_pdf_url(m.get("source", "")),
        # Present on EVERY clause of a document that has one, so the annexures are
        # reachable from whichever clause the search happened to land on.
        "bundle": _doc_bundle(m.get("source", "")),
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
        # Imported (non-registry) docs: use the status/effective date stored on
        # their DocumentAsset (set via Studio settings); default to Active.
        a = _app.DocumentAsset.query.filter_by(source_doc=str(source)).first()
        st = (a.status if a and a.status else "Active")
        ed = (a.effective_date if a else None)
        return {"doc_status": st, "effective_date": ed,
                "repealed_on": (ed if str(st).lower() == "repealed" else None)}
    return {"doc_status": d.get("status", "Active"),
            "effective_date": d.get("effective_date"),
            "repealed_on": d.get("repealed_on")}


_DOC_TYPE_LABELS = {
    "ACT": "Act", "REGULATION": "Regulation", "MASTER": "Master Circular",
    "MASTER CIRCULAR": "Master Circular", "CIRCULAR": "Circular",
    "GUIDELINE": "Guideline", "GUIDELINES": "Guideline",
}

def _norm_doc_type(t):
    """Consistent display label for a document type, so imported docs (whose KB
    Doc_Type is e.g. 'MASTER') read the same as registry docs ('Master Circular')."""
    key = str(t or "").strip().upper()
    return _DOC_TYPE_LABELS.get(key, str(t or "Document").strip() or "Document")



# ---------------------------------------------------------------------------
# INSURER 360 — every metric IRIS holds on one insurer, on one screen.
#
# Built on the entity/class layer (entity_type, insurer_class, metric_scope).
# Two rules are enforced here rather than left to the caller, because getting
# either wrong produces a confident wrong answer rather than an error:
#
#   1. entity_type='insurer' ALWAYS. The `insurer` column is polymorphic — it
#      also holds states, channels, countries and sector subtotals. Omitting this
#      is how a naive market share ranked "Maharashtra" and "Brokers" as insurers.
#   2. Peer comparison is scoped to the insurer's OWN class. Only 23 of 957
#      metrics are reported by all classes; persistency is a life concept and
#      combined ratio a general one, so cross-class ranking is meaningless.
#
# The KPI registry is deliberately hand-picked and small. Coverage does not imply
# usefulness — several metrics that every class reports are unlabelled balance
# sheet subtotals ("total_a__inr"), useless on a dashboard.
_KPIS = [
    # metric_id, label, unit, higher_is_better, agg
    #
    # `agg` matters more than it looks. Summing is only valid for ADDITIVE units.
    # Ratios are stored per-quarter (solvency: Q1..Q4 as 1.83/1.81/1.90/1.91) or
    # per-line-of-business, so summing them yields nonsense — the first cut of this
    # endpoint reported New India's solvency as 7.45 and its claims ratio as 475.
    #   sum       add rows (money, counts)
    #   latest_q  point-in-time ratio measured quarterly -> take the last quarter
    ("gross_direct_premium_within_india__inr", "Gross Direct Premium", "inr", True, "sum"),
    ("profit_after_tax__inr", "Profit After Tax", "inr", True, "sum"),
    ("solvency_ratio_of_general_health_and_reinsurance_companies", "Solvency Ratio", "ratio", True, "latest_q"),
    ("solvency_ratio", "Solvency Ratio", "ratio", True, "latest_q"),
    ("reported_during_the_year__count", "Grievances Reported", "count", False, "sum"),
    ("resolved_during_the_year__count", "Grievances Resolved", "count", True, "sum"),
    ("no_of_policies__count", "Policies", "count", True, "sum"),
    ("no_of_persons_covered__count", "Persons Covered", "count", True, "sum"),
    ("commission__inr", "Commission", "inr", None, "sum"),
    ("operating_expenses_related_to_insurance_business__inr", "Operating Expenses", "inr", False, "sum"),
    ("equity_share_capital__inr", "Equity Share Capital", "inr", None, "sum"),
]
# DELIBERATELY EXCLUDED — incurred_claims_ratio__percent. It is stored per line of
# business with no all-segments row, AND on two different scales in the same field
# (Health carries both 100.98 and 0.96). A single headline figure needs a
# premium-weighted roll-up plus a unit fix upstream; showing an unweighted mean
# would be a confident wrong number on a supervisory screen.
_KPI_BY_ID = {k[0]: k for k in _KPIS}


def _ins_frame():
    """Insurer-only slice of the financial engine, cached per data load."""
    df = brain.UNIFIED_DF
    if df is None or df.empty or "entity_type" not in df.columns:
        return None
    key = (id(df), len(df))
    cached = getattr(_ins_frame, "_c", None)
    if cached and cached[0] == key:
        return cached[1]
    sub = df[(df["entity_type"] == "insurer") & df["value_base"].notna()]
    _ins_frame._c = (key, sub)
    return sub


_Q_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Annual": 5}

# fy_canonical carries TWO formats: the dominant "2024-25" (19,129 insurer rows)
# and a bare "2025" (354 rows) from a handful of tables. Lexical sort puts "2025"
# after "2024-25", so a naive max() picked the 354-row fragment as the latest year.
# Analytics restrict to the financial-year form; the bare-year rows remain in the
# data but never define a period boundary.
_FY_RE = re.compile(r"^\d{4}-\d{2}$")


def _fy_series(values):
    """Sorted financial years (YYYY-YY only)."""
    return sorted({str(v) for v in values if v and _FY_RE.match(str(v))})


def _agg_by_year(frame, mode):
    """Series fy -> value, aggregated per `mode`. Summing a ratio is the single
    easiest way to put a wrong number on this screen, so the mode is explicit."""
    if frame.empty:
        return None
    if mode == "latest_q":
        f = frame.copy()
        f["_q"] = f["Quarter"].map(lambda q: _Q_ORDER.get(str(q), 0)) if "Quarter" in f.columns else 0
        f = f.sort_values("_q")
        return f.groupby("fy_canonical")["value_base"].last()
    return frame.groupby("fy_canonical")["value_base"].sum()


def _fy_sort(fys):
    """Financial years sort lexically ('2014-15' < '2015-16'), but guard anyway."""
    return sorted({str(f) for f in fys if f and str(f) != "nan"})


@api_bp.get("/insurer/list")
def api_insurer_list():
    """Insurers grouped by class, for the picker."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    df = _ins_frame()
    if df is None:
        return jsonify({"classes": {}})
    out = {}
    # NB: load_master_data_engine renames `insurer` -> `Entity`; the canonical
    # id columns keep their snake_case names.
    seen = df[["insurer_id", "Entity", "insurer_class"]].drop_duplicates("insurer_id")
    for _, r in seen.iterrows():
        cls = r["insurer_class"] or "Other"
        out.setdefault(cls, []).append({"id": r["insurer_id"], "name": r["Entity"]})
    for c in out:
        out[c].sort(key=lambda x: x["name"])
    return jsonify({"classes": out})


@api_bp.get("/insurer/<insurer_id>")
def api_insurer_360(insurer_id):
    """Everything IRIS holds on one insurer: KPIs with peer context, trends,
    derived metrics, and the regulatory documents that govern it."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    df = _ins_frame()
    if df is None:
        return jsonify({"ok": False, "message": "Financial engine not loaded."}), 503
    mine = df[df["insurer_id"] == insurer_id]
    if mine.empty:
        return jsonify({"ok": False, "message": "Unknown insurer."}), 404

    name = str(mine["Entity"].iloc[0])
    cls = mine["insurer_class"].iloc[0]
    peers = df[(df["insurer_class"] == cls) & (df["insurer_id"] != insurer_id)]
    cohort = df[df["insurer_class"] == cls]
    years = _fy_sort(mine["fy_canonical"])

    # ---- KPI cards: latest value, YoY, and where it sits in its own class ----
    kpis = []
    for mid, label, unit, higher_better, mode in _KPIS:
        m = mine[mine["metric_id"] == mid]
        if m.empty:
            continue
        g = _agg_by_year(m, mode)
        if g is None or g.empty:
            continue
        g = g[[i for i in _fy_sort(g.index)]]
        if g.empty:
            continue
        fy = g.index[-1]
        val = float(g.iloc[-1])
        prev = float(g.iloc[-2]) if len(g) > 1 else None
        yoy = ((val - prev) / abs(prev) * 100.0) if prev not in (None, 0) else None

        # peer distribution for the SAME year and SAME class
        pf = peers[(peers["metric_id"] == mid) & (peers["fy_canonical"] == fy)]
        pv = (pf.groupby("insurer_id")["value_base"].last() if mode == "latest_q"
              else pf.groupby("insurer_id")["value_base"].sum())
        pct = median = None
        if len(pv) >= 3:
            median = float(pv.median())
            pct = round(float((pv < val).sum()) / len(pv) * 100.0)
        kpis.append({
            "id": mid, "label": label, "unit": unit, "fy": fy, "value": val,
            "yoy_pct": round(yoy, 1) if yoy is not None else None,
            "peer_median": median, "percentile": pct, "peer_n": int(len(pv)),
            "higher_is_better": higher_better,
            # The handbook stopped publishing some series (grievances RESOLVED ends
            # at 2017-18 while REPORTED runs to 2024-25). Surfacing that beats
            # showing a stale figure that looks current.
            "stale": bool(years and fy != years[-1]),
            "latest_year": years[-1] if years else None,
            "trend": [{"fy": f, "v": float(v)} for f, v in g.items()],
        })

    # ---- derived: market share within class (not in the handbook) ----
    derived = []
    gdp = "gross_direct_premium_within_india__inr"
    mg = mine[mine["metric_id"] == gdp]
    if not mg.empty:
        fy = _fy_sort(mg["fy_canonical"])[-1]
        mine_v = float(mg[mg["fy_canonical"] == fy]["value_base"].sum())
        tot = cohort[(cohort["metric_id"] == gdp) & (cohort["fy_canonical"] == fy)]
        tot_by = tot.groupby("insurer_id")["value_base"].sum().sort_values(ascending=False)
        if tot_by.sum() > 0:
            derived.append({
                "label": f"Market Share ({cls})", "unit": "percent",
                "value": round(mine_v / float(tot_by.sum()) * 100.0, 2), "fy": fy,
                "note": f"rank {list(tot_by.index).index(insurer_id) + 1} of {len(tot_by)} in {cls}"
                        if insurer_id in tot_by.index else None,
            })

    # Grievance quality: a resolution RATE and a size-normalised rate say far more
    # than a raw count, which just ranks big insurers first.
    rep = mine[mine["metric_id"] == "reported_during_the_year__count"]
    res = mine[mine["metric_id"] == "resolved_during_the_year__count"]
    if not rep.empty and not res.empty:
        fys = sorted(set(_fy_sort(rep["fy_canonical"])) & set(_fy_sort(res["fy_canonical"])))
        if fys:
            fy = fys[-1]
            r = float(rep[rep["fy_canonical"] == fy]["value_base"].sum())
            s = float(res[res["fy_canonical"] == fy]["value_base"].sum())
            if r > 0:
                derived.append({"label": "Grievance Resolution Rate", "unit": "percent",
                                "value": round(s / r * 100.0, 1), "fy": fy,
                                "stale": bool(years and fy != years[-1]),
                                "note": ("resolved series ends " + fy +
                                         " — reported runs to " + years[-1])
                                        if years and fy != years[-1] else
                                        "may exceed 100%: prior-year grievances resolved this year"})
    pol = mine[mine["metric_id"] == "no_of_policies__count"]
    if not rep.empty and not pol.empty:
        fys = sorted(set(_fy_sort(rep["fy_canonical"])) & set(_fy_sort(pol["fy_canonical"])))
        if fys:
            fy = fys[-1]
            r = float(rep[rep["fy_canonical"] == fy]["value_base"].sum())
            p = float(pol[pol["fy_canonical"] == fy]["value_base"].sum())
            if p > 0:
                derived.append({"label": "Grievances per lakh policies", "unit": "rate",
                                "value": round(r / p * 100000.0, 1), "fy": fy,
                                "note": "size-normalised — comparable across insurers"})

    return jsonify({
        "ok": True,
        "insurer": {"id": insurer_id, "name": name, "class": cls},
        "years": years,
        "kpis": kpis,
        "derived": derived,
        "cohort_size": int(cohort["insurer_id"].nunique()),
    })


@api_bp.get("/insurer/compare")
def api_insurer_compare():
    """Side-by-side comparison of 2-4 insurers.

    Cross-class comparison is the trap this endpoint exists to manage. Only 23 of
    957 metrics are reported by every class, so comparing a life insurer's book to
    a general insurer's is usually meaningless. Rather than refuse it (supervisors
    sometimes DO want a cross-class view of universal balance-sheet items), the
    response reports `mixed_class` and restricts the metric list to lines every
    selected insurer actually reports — so a row is never half-empty by surprise.
    """
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    ids = [i for i in request.args.getlist("id") if i][:4]
    if len(ids) < 2:
        return jsonify({"ok": False, "message": "Select at least two insurers."}), 400
    df = _ins_frame()
    if df is None:
        return jsonify({"ok": False, "message": "Financial engine not loaded."}), 503

    subs, meta = {}, []
    for iid in ids:
        m = df[df["insurer_id"] == iid]
        if m.empty:
            continue
        subs[iid] = m
        meta.append({"id": iid, "name": str(m["Entity"].iloc[0]),
                     "class": m["insurer_class"].iloc[0]})
    if len(subs) < 2:
        return jsonify({"ok": False, "message": "Not enough valid insurers."}), 404
    classes = {x["class"] for x in meta}

    rows = []
    for mid, label, unit, higher_better, mode in _KPIS:
        series = {}
        for iid, m in subs.items():
            g = _agg_by_year(m[m["metric_id"] == mid], mode)
            if g is None or g.empty:
                continue
            g = g[[i for i in _fy_sort(g.index)]]
            if not g.empty:
                series[iid] = g
        # Every selected insurer must report it, else the row misleads by omission.
        if len(series) != len(subs):
            continue
        # Compare on the newest year they SHARE — comparing 2024-25 against
        # 2017-18 would silently reward whoever has fresher filings.
        common = set.intersection(*[set(s.index) for s in series.values()])
        if not common:
            continue
        fy = sorted(common)[-1]
        vals = {iid: float(s[fy]) for iid, s in series.items()}
        if higher_better is None:
            best = worst = None
        else:
            best = max(vals, key=vals.get) if higher_better else min(vals, key=vals.get)
            worst = min(vals, key=vals.get) if higher_better else max(vals, key=vals.get)
        rows.append({
            "id": mid, "label": label, "unit": unit, "fy": fy,
            "higher_is_better": higher_better,
            "best": best, "worst": worst,
            "values": {iid: vals[iid] for iid in vals},
            "trends": {iid: [{"fy": f, "v": float(v)} for f, v in s.items()]
                       for iid, s in series.items()},
        })

    # Size-normalised conduct metric. Raw grievance counts just rank by size — New
    # India files 7,768 against ICICI Lombard's 1,211 largely because it is bigger.
    # Per lakh policies is what makes a small insurer's conduct problem visible
    # next to a large one's, so it is computed for the comparison too.
    norm = {}
    for iid, m in subs.items():
        rp = _agg_by_year(m[m["metric_id"] == "reported_during_the_year__count"], "sum")
        pl = _agg_by_year(m[m["metric_id"] == "no_of_policies__count"], "sum")
        if rp is None or pl is None or rp.empty or pl.empty:
            continue
        common = sorted(set(rp.index) & set(pl.index))
        if not common:
            continue
        fy = common[-1]
        if float(pl[fy]) > 0:
            norm[iid] = {"fy": fy, "v": round(float(rp[fy]) / float(pl[fy]) * 100000.0, 1)}
    if len(norm) == len(subs) and norm:
        fys = {v["fy"] for v in norm.values()}
        vals = {i: v["v"] for i, v in norm.items()}
        rows.append({
            "id": "_grievances_per_lakh", "label": "Grievances per lakh policies",
            "unit": "rate", "fy": sorted(fys)[-1], "higher_is_better": False,
            "best": min(vals, key=vals.get), "worst": max(vals, key=vals.get),
            "values": vals, "trends": {}, "derived": True,
            "note": "size-normalised — the comparable conduct measure",
        })

    # Market share is only meaningful inside one class — a share of "all insurance"
    # would mix incompatible premium definitions.
    share = {}
    if len(classes) == 1:
        cls = next(iter(classes))
        gdp = "gross_direct_premium_within_india__inr"
        cohort = df[(df["insurer_class"] == cls) & (df["metric_id"] == gdp)]
        if not cohort.empty:
            fy = _fy_sort(cohort["fy_canonical"])[-1]
            tot = cohort[cohort["fy_canonical"] == fy].groupby("insurer_id")["value_base"].sum()
            if tot.sum() > 0:
                for iid in subs:
                    if iid in tot.index:
                        share[iid] = round(float(tot[iid]) / float(tot.sum()) * 100.0, 2)
                share["_fy"] = fy
                share["_class"] = cls

    return jsonify({
        "ok": True,
        "insurers": meta,
        "mixed_class": len(classes) > 1,
        "classes": sorted(classes),
        "metrics": rows,
        "market_share": share,
    })


@api_bp.get("/insurer/industry")
def api_industry_trends():
    """Market-level trends: size and growth by class, concentration, channel mix,
    and industry-wide conduct.

    Everything here is built by aggregating INSURER rows rather than reading the
    handbook's own industry totals. That is deliberate: the totals rows are
    inconsistent across years and dimensions (some exist only at sector level, some
    double as entities), whereas summing entity_type='insurer' is reproducible and
    matches what the per-insurer screens show. entity_type is what makes it safe —
    without it the sum would swallow states, channels and sector subtotals.
    """
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    df = _ins_frame()
    if df is None:
        return jsonify({"ok": False, "message": "Financial engine not loaded."}), 503

    # There is NO valid cross-class premium line, and reaching for one is the exact
    # trap this module keeps warning about: `gross_premium__inr` looks universal but
    # for life insurers it is a minor sub-line — LIC reports ₹334 Cr on it against a
    # real book of lakhs of crores, which would have put the entire life industry at
    # ₹2,386 Cr on this screen. Each class therefore uses ITS OWN premium basis, and
    # the classes are never summed into one "industry total".
    PREM_BY_CLASS = {
        "General": ("gross_direct_premium_within_india__inr", "Gross Direct Premium"),
        "SAHI": ("gross_direct_premium_within_india__inr", "Gross Direct Premium"),
        "Life": ("total_premium__inr", "Total Premium"),
    }

    size, concentration, basis = {}, {}, {}
    for cls, (mid, plabel) in PREM_BY_CLASS.items():
        basis[cls] = plabel
        c = df[(df["metric_id"] == mid) & (df["insurer_class"] == cls)]
        if c.empty:
            continue
        by_year = c.groupby("fy_canonical")["value_base"].sum()
        size[cls] = [{"fy": f, "v": float(v)} for f, v in
                     sorted(by_year.items(), key=lambda kv: kv[0])]
        # Concentration is a supervisory measure the handbook does NOT publish.
        # HHI = sum of squared percentage shares (0-10,000). Above ~2,500 is
        # conventionally "highly concentrated".
        hh = []
        for fy, grp in c.groupby("fy_canonical"):
            shares = grp.groupby("insurer_id")["value_base"].sum()
            tot = float(shares.sum())
            if tot <= 0 or len(shares) < 2:
                continue
            pct = (shares / tot * 100.0)
            top5 = float(pct.sort_values(ascending=False).head(5).sum())
            hh.append({"fy": fy, "hhi": round(float((pct ** 2).sum())),
                       "top5": round(top5, 1), "n": int(len(shares))})
        concentration[cls] = sorted(hh, key=lambda x: x["fy"])

    # CAGR over the longest common window per class — a single number that says
    # more than a slope, and immune to a one-off spike at either end.
    growth = {}
    for cls, series in size.items():
        if len(series) >= 2 and series[0]["v"] > 0:
            n = len(series) - 1
            growth[cls] = round(((series[-1]["v"] / series[0]["v"]) ** (1 / n) - 1) * 100, 1)

    # Channel mix — how the market actually reaches customers, and how that shifts.
    full = brain.UNIFIED_DF
    chan = []
    if full is not None and "entity_type" in full.columns:
        cf = full[(full["entity_type"] == "channel")
                  & (full["metric_id"] == "gross_premium__inr")
                  & full["value_base"].notna()]
        if not cf.empty:
            latest = sorted(cf["fy_canonical"].dropna().unique())[-1]
            cur = cf[cf["fy_canonical"] == latest].groupby("Entity")["value_base"].sum()
            tot = float(cur.sum())
            if tot > 0:
                chan = [{"name": k, "v": float(v), "pct": round(float(v) / tot * 100, 1)}
                        for k, v in cur.sort_values(ascending=False).items() if v > 0][:8]
            chan_fy = latest
        else:
            chan_fy = None
    else:
        chan_fy = None

    # Industry conduct: raw counts grow with the market, so the normalised rate is
    # the one that tells a supervisor whether things are actually getting worse.
    conduct = []
    rep = df[df["metric_id"] == "reported_during_the_year__count"]
    pol = df[df["metric_id"] == "no_of_policies__count"]
    if not rep.empty and not pol.empty:
        r_by = rep.groupby("fy_canonical")["value_base"].sum()
        p_by = pol.groupby("fy_canonical")["value_base"].sum()
        for fy in sorted(set(r_by.index) & set(p_by.index)):
            if float(p_by[fy]) > 0:
                conduct.append({"fy": fy, "reported": float(r_by[fy]),
                                "per_lakh": round(float(r_by[fy]) / float(p_by[fy]) * 1e5, 1)})

    return jsonify({
        "ok": True,
        "premium_basis": basis,
        "size": size,
        "growth_cagr": growth,
        "concentration": concentration,
        "channel_mix": chan,
        "channel_fy": chan_fy,
        "conduct": conduct,
    })


# Regulatory floor for the solvency ratio (IRDAI control level). This is a real
# statutory threshold, not a tuned parameter — which is why breaching it outranks
# every statistical signal in the worklist.
SOLVENCY_FLOOR = 1.50


@api_bp.get("/insurer/exceptions")
def api_insurer_exceptions():
    """Ranked supervisory worklist: which insurers need attention, and why.

    This is the inversion of a dashboard. A dashboard answers "what do we know
    about X"; this answers "who should I look at today" — which is the question a
    supervisor actually arrives with, and the reason the existing explorer screens
    go unused.

    Every exception is EXPLAINABLE by construction: each carries the value, the
    comparison it failed, and a sentence a human can act on. Comparisons use the
    median of the insurer's own class (never the mean, which one outlier drags,
    and never across classes, where metrics mean different things).
    """
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    df = _ins_frame()
    if df is None:
        return jsonify({"ok": False, "message": "Financial engine not loaded."}), 503

    all_fy = _fy_series(df["fy_canonical"].dropna().unique())
    latest_fy = all_fy[-1] if all_fy else None
    # A figure two years stale is a different kind of signal from a current breach,
    # so staleness is recorded rather than silently treated as live.
    recent = all_fy[-2:]
    out = []

    def add(row, kind, severity, metric, label, value, fy, why, extra=None):
        e = {"insurer_id": row["insurer_id"], "insurer": row["insurer"],
             "class": row["class"], "kind": kind, "severity": severity,
             "metric_id": metric, "metric": label, "value": value, "fy": fy,
             "why": why, "stale": fy not in recent}
        if extra:
            e.update(extra)
        out.append(e)

    ids = df[["insurer_id", "Entity", "insurer_class"]].drop_duplicates("insurer_id")
    meta = {r["insurer_id"]: {"insurer_id": r["insurer_id"], "insurer": r["Entity"],
                              "class": r["insurer_class"]} for _, r in ids.iterrows()}

    # ---- 1. Solvency below the statutory floor -----------------------------
    sol = df[df["metric_id"].str.startswith("solvency", na=False)]
    for iid, g in sol.groupby("insurer_id"):
        fy = sorted(g["fy_canonical"].dropna().unique())[-1]
        last = _agg_by_year(g[g["fy_canonical"] == fy], "latest_q")
        if last is None or last.empty:
            continue
        v = float(last.iloc[-1])
        if v < SOLVENCY_FLOOR:
            sev = "high" if v < 1.0 else "medium"
            why = (f"Solvency {v:.2f} is below the {SOLVENCY_FLOOR:.2f} regulatory floor"
                   + (f" — capital deficit (negative)" if v < 0 else ""))
            add(meta[iid], "threshold", sev, "solvency", "Solvency Ratio", round(v, 2), fy, why,
                {"threshold": SOLVENCY_FLOOR})

    # ---- 2. Loss-making ----------------------------------------------------
    pat = df[df["metric_id"] == "profit_after_tax__inr"]
    for iid, g in pat.groupby("insurer_id"):
        by = _agg_by_year(g, "sum")
        if by is None or by.empty:
            continue
        fy = sorted(by.index)[-1]
        v = float(by[fy])
        # Materiality floor: a sub-crore loss rounds to "₹0 Cr" on screen and is not
        # worth a supervisor's queue slot on its own.
        if v < -1e7:
            # Consecutive loss years are a materially worse signal than one bad year.
            yrs = sorted(by.index)
            streak = 0
            for f in reversed(yrs):
                if float(by[f]) < 0:
                    streak += 1
                else:
                    break
            sev = "high" if streak >= 3 else "medium"
            why = (f"Loss of ₹{abs(v)/1e7:,.0f} Cr"
                   + (f" — {streak} consecutive loss-making years" if streak > 1 else ""))
            add(meta[iid], "loss", sev, "profit_after_tax__inr", "Profit After Tax",
                round(v, 2), fy, why, {"streak": streak})

    # ---- 3. Conduct outlier vs class median --------------------------------
    # Grievances per lakh policies: the size-normalised measure. Raw counts just
    # rank by size, so an outlier test on them would only ever flag the biggest.
    rep = df[df["metric_id"] == "reported_during_the_year__count"]
    pol = df[df["metric_id"] == "no_of_policies__count"]
    # Two guards, both learned from false positives this engine produced:
    #   * MIN_POLICIES — Go Digit Life reports 6 policies against 206 grievances,
    #     which yields a rate of millions per lakh. A denominator that small is a
    #     filing artefact, not conduct; below the floor no rate is computed.
    #   * IMPLAUSIBLE_RATE — Zuno reports 4,503 grievances on 11,402 policies (39%).
    #     A rate that high means the two lines are on different bases (policies is a
    #     segment, grievances the whole book). That is a DATA QUALITY finding, and
    #     is reported as one instead of being laundered into a conduct ranking.
    MIN_POLICIES = 50_000
    IMPLAUSIBLE_RATE = 5_000        # >5% of policies generating a grievance
    rates = {}
    for iid in set(rep["insurer_id"]) & set(pol["insurer_id"]):
        r = _agg_by_year(rep[rep["insurer_id"] == iid], "sum")
        p = _agg_by_year(pol[pol["insurer_id"] == iid], "sum")
        common = sorted(set(r.index) & set(p.index))
        if not common:
            continue
        fy = common[-1]
        pv, rv = float(p[fy]), float(r[fy])
        if pv < MIN_POLICIES:
            continue
        rate = rv / pv * 1e5
        if rate > IMPLAUSIBLE_RATE:
            add(meta[iid], "data_quality", "low", "_grievances_per_lakh",
                "Grievances vs policies", round(rate, 1), fy,
                f"{rv:,.0f} grievances against {pv:,.0f} policies — the two lines "
                f"appear to be on different bases, not a conduct signal")
            continue
        rates[iid] = (fy, rate)
    by_class = {}
    for iid, (fy, rate) in rates.items():
        by_class.setdefault(meta[iid]["class"], []).append(rate)
    for iid, (fy, rate) in rates.items():
        peers = by_class.get(meta[iid]["class"], [])
        if len(peers) < 4:
            continue
        med = sorted(peers)[len(peers) // 2]
        if med > 0 and rate > med * 1.5:
            ratio = rate / med
            sev = "high" if ratio >= 2.5 else "medium"
            add(meta[iid], "outlier", sev, "_grievances_per_lakh",
                "Grievances per lakh policies", round(rate, 1), fy,
                f"{ratio:.1f}x the {meta[iid]['class']} median of {med:,.0f} per lakh policies",
                {"peer_median": round(med, 1), "ratio": round(ratio, 2)})

    # ---- 4. Sharp adverse year-on-year swing -------------------------------
    SWING = {"gross_direct_premium_within_india__inr": ("Gross Direct Premium", True, 25),
             "reported_during_the_year__count": ("Grievances Reported", False, 50)}
    for mid, (label, higher_better, pct_gate) in SWING.items():
        for iid, g in df[df["metric_id"] == mid].groupby("insurer_id"):
            by = _agg_by_year(g, "sum")
            if by is None or len(by) < 2:
                continue
            yrs = sorted(by.index)
            cur, prev = float(by[yrs[-1]]), float(by[yrs[-2]])
            if prev <= 0:
                continue
            chg = (cur - prev) / abs(prev) * 100.0
            adverse = chg < -pct_gate if higher_better else chg > pct_gate
            if adverse:
                add(meta[iid], "swing", "medium", mid, label, round(cur, 2), yrs[-1],
                    f"{'fell' if chg < 0 else 'rose'} {abs(chg):.0f}% vs {yrs[-2]}"
                    + ("" if higher_better else " — rising grievances"),
                    {"change_pct": round(chg, 1), "prev_fy": yrs[-2]})

    rank = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda e: (rank.get(e["severity"], 3), e["stale"], -abs(e.get("ratio") or 0)))

    summary = {}
    for e in out:
        summary[e["severity"]] = summary.get(e["severity"], 0) + 1
    flagged = len({e["insurer_id"] for e in out})

    return jsonify({
        "ok": True, "latest_fy": latest_fy,
        "exceptions": out, "summary": summary,
        "insurers_flagged": flagged,
        "insurers_total": int(df["insurer_id"].nunique()),
    })


@api_bp.get("/documents")
def api_documents():
    """Document tree (Act → Regulation → Circular) + download links + status, for
    the Downloads page."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    is_admin = bool(getattr(_app.current_user, "is_admin", False))
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
            "id": d["id"], "title": d.get("title", d["id"]), "type": _norm_doc_type(d.get("type", "Document")),
            "category": d.get("category", ""), "parent": d.get("parent"),
            "status": d.get("status", "Active"),
            "effective_date": d.get("effective_date"), "repealed_on": d.get("repealed_on"),
            "repealed_by": d.get("repealed_by"),
            "clauses": int(counts.get(d["id"], 0)),
            # Prefer a bundled static PDF; otherwise fall back to an uploaded PDF
            # asset (Doc Studio "Attach/Replace PDF") so registry docs can carry a
            # user-supplied PDF too — previously they showed "No file".
            "download_url": ("/static/" + pdf) if pdf else _doc_pdf_url(str(d["id"])),
            "children": [],
        }
    # KB documents not in the registry (e.g. imported via Studio) with a
    # downloadable PDF — add them as nodes too, carrying their stored hierarchy
    # (parent_doc), status and effective date so they slot into the tree.
    imported_ids = set()
    reg_ids = set(nodes.keys())
    try:
        kb = brain.load_knowledge_base()
        meta = {}
        if kb is not None and not kb.empty:
            for src, g in kb.groupby("Source_Doc"):
                meta[str(src)] = (str(g["Doc_Type"].iloc[0]) if "Doc_Type" in g.columns else "Document",
                                  str(g["Doc_Category"].iloc[0]) if "Doc_Category" in g.columns else "")
        assets = {a.source_doc: a for a in _app.DocumentAsset.query.all()}
        for src in sorted(counts):
            src = str(src)
            if src in reg_ids:
                continue
            url = _doc_pdf_url(src)
            if not url:
                continue
            a = assets.get(src)
            dt, dc = meta.get(src, ("Document", ""))
            nodes[src] = {
                "id": src, "title": src, "type": _norm_doc_type(dt or "Document"), "category": dc,
                "parent": (a.parent_doc if a else None),
                "status": (a.status if a and a.status else "Active"),
                "effective_date": (a.effective_date if a else None), "repealed_on": None,
                "repealed_by": None, "clauses": int(counts.get(src, 0)),
                "download_url": url, "children": [],
            }
            imported_ids.add(src)
    except Exception:
        pass
    # Link the unified set. Imported docs with a resolvable parent slot into the
    # main hierarchy; those without go to a separate 'Imported' catch-all.
    roots, repealed, imported = [], [], []
    for n in nodes.values():
        if str(n["status"]).lower() == "repealed":
            repealed.append(n)
        elif n["parent"] and n["parent"] in nodes:
            nodes[n["parent"]]["children"].append(n)
        elif n["id"] in imported_ids:
            imported.append(n)
        else:
            roots.append(n)
    # Loose "imported/stray" docs (not slotted into the registry hierarchy) are an
    # admin housekeeping view — regular users shouldn't see stray uploads.
    return jsonify({"tree": roots, "repealed": repealed,
                    "imported": imported if is_admin else []})


def _pq_body_proper(body):
    """The reply itself — letterhead (Ref/date/addressee/Subject) and the standard
    covering intro stripped, whitespace flattened."""
    if not body:
        return ""
    m = re.search(r"Subject\s*:.*?(?:\n|$)", body, re.I) or re.search(r"Dear\s+Sir.*?(?:\n|$)", body, re.I)
    rest = body[m.end():] if m else body
    rest = re.sub(r"\s+", " ", rest).strip()
    # Drop the standard covering-letter intro ("This has reference … seriatim for the same.")
    return re.sub(r"^This has reference.*?seriatim for the same\.\s*", "", rest, flags=re.I).strip()


def _pq_lede(body, limit=220):
    """Skip the letterhead (Ref/date/address/Subject) and snippet the actual content."""
    rest = _pq_body_proper(body)
    return rest[:limit] + ("…" if len(rest) > limit else "")


def _pq_snippet(body, roots=None, limit=220, radius=200):
    """A PQ preview. Browsing shows the lede; searching shows the match.

    Without roots this is the lede — the opening of the reply proper. That is the
    right preview for browse/tag/number, and the wrong one for a search: the reply
    is thousands of characters long and the term that matched is almost never in
    its opening. In PQ 9000 "cashless" sits at char 3,004 and the lede ends by ~508.

    With roots, windows around the earliest match instead, snapping to word
    boundaries. Falls back to the lede when no root is in the body (a tag-only hit
    has nothing to centre on).

    Both search the reply PROPER, never the letterhead: every PQ is addressed to the
    "Insurance Section" of DFS, so a search for "insurance" would otherwise match the
    envelope at char ~100 and every snippet would show the address block.
    """
    flat = _pq_body_proper(body)
    if not flat:
        return ""
    lc = flat.lower()
    at = None
    for r in (roots or []):
        m = re.search(rf"\b{re.escape(r)}\w*", lc)
        if m and (at is None or m.start() < at):
            at = m.start()
    if at is None:
        return flat[:limit] + ("…" if len(flat) > limit else "")
    start, end = max(0, at - radius), min(len(flat), at + radius)
    if start > 0:                       # snap forward to a word start
        sp = flat.find(" ", start)
        if sp != -1 and sp < start + 40:
            start = sp + 1
    if end < len(flat):                 # snap back to a word end
        sp = flat.rfind(" ", start, end)
        if sp > start + 40:
            end = sp
    # A window landing inside a data table would otherwise open/close on a bare "|",
    # which the client's markdown-table renderer reads as a table row.
    slice_ = flat[start:end].strip().strip("|").strip()
    return ("… " if start > 0 else "") + slice_ + (" …" if end < len(flat) else "")


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


def _pq_newest_first():
    """Every PQ, most recently answered first.

    Ordering used to be by id — upload order. With a handful of PQs that looked
    like recency; over a back-catalogue it is arbitrary, and "the most recent
    reply on this topic" is the question the PQ view exists to answer. Sorts on
    doc_date_iso (NULLs last, since an unparsed date is not a new one), then id
    to break ties between replies filed the same day.
    """
    m = _app
    return m.PqDocument.query.order_by(
        (m.PqDocument.doc_date_iso.is_(None)).asc(),
        m.PqDocument.doc_date_iso.desc(),
        m.PqDocument.id.desc(),
    ).all()


def _pq_card(r, roots=None):
    return {
        "id": r.id, "pq_no": r.pq_no, "house": r.house, "title": r.title,
        "subject": r.subject, "date": r.doc_date, "date_iso": r.doc_date_iso,
        "tags": [t.strip() for t in (r.tags or "").split(",") if t.strip()],
        "departments": _dept_list(r),
        "snippet": _pq_snippet(r.body_text or "", roots=roots),
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
    num = (request.args.get("num") or "").strip()
    tag = (request.args.get("tag") or "").strip()
    q = (request.args.get("q") or "").strip()
    deep = (request.args.get("deep") or "").strip()
    chips = []
    deep_flag = False
    hl_source = ""      # the text whose terms get highlighted on the cards
    if deep:
        # Deep Scan — read every reply body and return all that contain the query.
        mode = (request.args.get("mode") or "all").strip().lower()
        if mode not in {"word", "all", "phrase"}:
            mode = "all"
        items = _deep_scan_pqs(deep, mode=mode)
        deep_flag = True
        hl_source = deep
        try:
            email = _app.current_user.email if _app.current_user.is_authenticated else None
            _app._record_search(email, "pq", "[Deep Scan] " + deep, len(items))
        except Exception:
            pass
    elif num:
        items = _search_pq_numbers(num)
    elif tag:
        # Exact-tag filter (case/space-insensitive) — not a body word search.
        key = re.sub(r"\s+", "", tag.lower())
        rows = _pq_newest_first()
        items = [_pq_card(r) for r in rows
                 if key in {re.sub(r"\s+", "", t.strip().lower()) for t in (r.tags or "").split(",") if t.strip()}]
    elif q:
        items = _search_pqs(q, limit=100)
        chips = _pq_chips(q)   # offer Deep-Scan suggestions alongside headline hits
        hl_source = q
        try:
            email = _app.current_user.email if _app.current_user.is_authenticated else None
            _app._record_search(email, "pq", q, len(items))
        except Exception:
            pass
    else:
        rows = _pq_newest_first()
        items = [_pq_card(r) for r in rows]
    # Department facet — narrows whatever the base set is (a PQ matches if it
    # carries the requested department; multi-dept PQs match any of theirs).
    depts = _norm_departments(request.args.get("dept"))
    if depts:
        want = set(depts)
        items = [it for it in items if want & set(it.get("departments") or [])]
    # Highlight terms, same contract as /api/search: `highlight` is per-word (amber),
    # `highlight_phrase` is the verbatim query as one unit (green). Drawn from the
    # PQ stoplist, so what lights up is exactly what could have matched.
    phrase_l = " ".join(hl_source.lower().split())
    return jsonify({
        "items": items, "chips": chips, "deep": deep_flag,
        "highlight": _word_highlights(hl_source, stop=_PQ_STOP) if hl_source else [],
        "highlight_phrase": [phrase_l.split()] if (hl_source and " " in phrase_l) else [],
    })


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
        "filename": r.docx_filename or "",
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
    is_pdf = (r.docx_filename or "").lower().endswith(".pdf")
    return send_file(io.BytesIO(data),
                     mimetype=("application/pdf" if is_pdf
                               else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
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
    if not f or not f.filename.lower().endswith((".docx", ".pdf")):
        return jsonify({"ok": False, "message": "Please upload a .docx or .pdf file."}), 400
    tags = (request.form.get("tags") or "").strip()
    departments = ",".join(_norm_departments(request.form.get("departments")))
    date_override = (request.form.get("date") or "").strip()[:40]
    force = str(request.form.get("force") or "").lower() in {"1", "true", "yes"}
    raw = f.read()
    try:
        parsed = parse_pq(raw, filename=f.filename)
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
    shown_date = date_override or parsed["doc_date"]
    row = m.PqDocument(
        pq_no=parsed["pq_no"], house=parsed["house"], title=parsed["title"],
        subject=parsed["subject"], doc_date=shown_date,
        doc_date_iso=_iso_date(shown_date) or None, tags=tags,
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
        if not f.filename.lower().endswith((".docx", ".pdf")):
            skipped.append(f.filename); continue
        raw = f.read()
        try:
            parsed = parse_pq(raw, filename=f.filename)
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
            subject=parsed["subject"], doc_date=parsed["doc_date"],
            doc_date_iso=parsed["doc_date_iso"] or None, tags="",
            html=parsed["html"], body_text=parsed["text"], docx_filename=safe,
            created_at=datetime.utcnow())
        m.db.session.add(row)
        m.db.session.commit()
        created.append({"id": row.id, "title": row.title, "pq_no": row.pq_no,
                        "house": row.house, "date": row.doc_date, "filename": f.filename})
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
# The clause stoplist plus words that carry no signal in THIS corpus specifically:
# every reply is titled "... Starred Question ... regarding X" and answers under
# "Reply:", so those are letterhead, not content. Kept as the single source for
# _pq_terms / _pq_vocab / _pq_chips / PQ highlighting, so that every term that can
# drive a match can also be highlighted, and nothing is highlighted that cannot.
# (Universal's _HL_STOP would be wrong here — it would light up "reply" on every
# card while "reply" drove no result.)
_PQ_STOP = brain.FUNCTION_WORDS | {
    "regarding", "respect", "question", "answer", "reply", "starred", "unstarred",
    "sabha", "lok", "rajya", "seriatim", "captioned",
}


def _pq_terms(text):
    """Significant lowercase words in a query (drops stopwords + tiny tokens)."""
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower())
            if len(t) > 2 and t not in _PQ_STOP]


def _pq_roots(query):
    """Distinct word-roots the user typed — the PQ twin of api_search's core_roots.

    Uses brain.search_root (Porter + the letter-restoration guard) rather than raw
    tokens, so "penetration" finds "penetrate". Deliberately does NOT go through
    brain.get_clean_keywords: that reads vocab globals which _rebuild_vocab builds
    from the CLAUSE corpus and clears, so a PQ query would be spell-corrected
    against clause vocabulary. PQ has its own vocabulary in _pq_vocab().
    """
    out = []
    for t in _pq_terms(query):
        r = brain.search_root(t) if t.isalpha() else t
        if r and r not in out:
            out.append(r)
    return out


_PQ_NORM_CACHE = {}


def _pq_norm(r):
    """Cached (lowercase, hyphens-stripped) body for a PQ row.

    Keyed on (id, len(body_text)) so a re-extract invalidates it — deliberately not
    _pq_vocab's (row count, newest id), which never moves when a row is edited in
    place and so goes stale on re-tag/re-extract.
    """
    body = r.body_text or ""
    key = (r.id, len(body))
    hit = _PQ_NORM_CACHE.get(key)
    if hit is None:
        hit = brain._derive_search_columns(body)
        _PQ_NORM_CACHE[key] = hit
    return hit


def _search_pqs(query, limit=100):
    """Headline (precise) tier — match a PQ by its TAGS only (the curated
    keywords an editor deliberately assigned). Titles, subjects and full reply
    bodies are reserved for Deep Scan, so a typed word never matches just
    because it happens to appear in a subject line. Every query term must be
    present in the tags (AND)."""
    return _pq_tiered_search(query, limit=limit)


# Candidate caps, applied BEFORE scoring. _run_len is O(n^2) in the query and scans
# the whole text, and a PQ body is ~13x a clause: a 13-word query costs ~27ms per
# 25k body, so an uncapped back-catalogue would spend ~8s in scoring alone.
# Universal gets away without this because TAG_CAND_CAP/CONTENT_CAP already bound it.
PQ_TAG_CAP, PQ_BODY_CAP = 60, 40
# Above this many content words, skip the consecutive-run scan (see _make_scorer).
PQ_RUN_LEN_CAP = 8


def _pq_tiered_search(query, limit=100):
    """PQ full-text search in two tiers: curated tags first, reply bodies second.

    Tags-only search was a dead end — a typed word could not find an untagged PQ at
    all. Bodies now match too, but as a SEPARATE tier rather than a merged list: a
    tag is an editor's deliberate judgement about what a reply is about, and an
    incidental mention in 25k characters of prose is not. So a body hit never
    outranks a tag hit, and _score only ever orders WITHIN a tier.

    That tier split — not word-boundary matching — is also what fixes searching
    "Star Health" returning an unrelated PMJAY reply: "star" genuinely does stem to
    the same root as the "Starred" in its title, so no matcher can separate them.
    But PMJAY is tagged "PMJAY", so it cannot be a tag hit; it lands under bodies,
    where an incidental match belongs.
    """
    roots = _pq_roots(query)
    if not roots:
        return []
    phrase_l = " ".join((query or "").lower().split())
    multi = " " in phrase_l or bool(re.search(r"[A-Za-z0-9][/\-][A-Za-z0-9]", phrase_l))
    tag_cand, body_cand = [], []
    # Newest-first in + stable sorts throughout => recency survives as the tiebreak
    # within every tier, which is what "the best previous answer" actually means.
    for r in _pq_newest_first():
        tags_lc, tags_lcj = brain._derive_search_columns(r.tags or "")
        if tags_lc and all(brain._root_present(x, tags_lc, tags_lcj) for x in roots):
            tag_cand.append((r, tags_lc, tags_lcj))
            continue
        body_lc, body_lcj = _pq_norm(r)
        head_lc, head_lcj = brain._derive_search_columns(
            " ".join([r.title or "", r.subject or ""]))
        if all(brain._root_present(x, body_lc, body_lcj)
               or brain._root_present(x, head_lc, head_lcj) for x in roots):
            body_cand.append((r, body_lc, body_lcj))

    def _cheap_rank(cands, cap):
        # Pre-rank so the cap keeps the best candidates rather than the first N.
        # Must be genuinely cheap: this runs over EVERY candidate's full body, while
        # the real scoring only runs over the capped set. str.count is a C-level
        # substring scan; brain._root_count would be re.findall over 25k chars per
        # root per candidate, which profiled at 77% of total query time. Counting
        # substrings over-counts mid-word hits ("care" in "healthcare"), which is
        # fine for choosing WHICH candidates to score — _score then ranks them on
        # word boundaries. Stable, so newest-first survives ties.
        ranked = sorted(cands, key=lambda c: (
            0 if phrase_l in c[1] else 1,
            -sum(c[1].count(x) for x in roots),
        ))
        return ranked[:cap]

    out = []
    for cands, cap, tier in ((tag_cand, PQ_TAG_CAP, "tag"),
                             (body_cand, PQ_BODY_CAP, "body")):
        picked = _cheap_rank(cands, cap)
        # Materialise before scoring: _make_scorer memoises on id(m), and CPython
        # recycles the id of a dict that nothing holds a reference to.
        scored = [{"raw_text": lc, "_row": r} for r, lc, _lcj in picked]
        score, _promote = _make_scorer(phrase_l, roots, multi, run_len_cap=PQ_RUN_LEN_CAP)
        scored.sort(key=lambda m: tuple(-v for v in score(m)))
        for m in scored:
            card = _pq_card(m["_row"], roots=roots)
            card["tier"] = tier
            out.append(card)
    return out[:limit]


def _search_pq_numbers(num, limit=100):
    """Find PQs by number, tiered exact > prefix > loose.

    Mirrors brain.search_by_clause_number (which can't be called directly — it walks
    a clause DataFrame), reusing its _num_key so "S6487" / "6487" / "6 487" collapse
    alike. The old path stripped non-digits off BOTH sides and did a substring test,
    so a PQ numbered S6487 was unfindable and "/900" ranked 9000 above 900.
    """
    q = brain._num_key(str(num).lstrip("/"))
    if not q:
        return []
    exact, prefix, loose = [], [], []
    for r in _pq_newest_first():          # stable: newest first within each tier
        k = brain._num_key(r.pq_no or "")
        if not k:
            continue
        if k == q:
            exact.append(r)
        elif k.startswith(q):
            prefix.append(r)
        elif len(q) >= 2 and q in k:
            loose.append(r)
    prefix.sort(key=lambda r: len(brain._num_key(r.pq_no or "")))   # shortest = closest
    out = exact + prefix + loose
    return [_pq_card(r) for r in out[:limit]]


def _deep_scan_pqs(text, mode="all", limit=300):
    """Deep Scan (recall) tier — scan the FULL rendered reply text of every PQ
    and return each one that contains the query. mode:
      word   -> a single term anywhere in the reply
      all    -> every significant word present (AND)
      phrase -> the exact phrase appears verbatim"""
    phrase = " ".join((text or "").lower().split())
    roots = _pq_roots(text)
    out = []
    # Newest-first in, stable sort by score => equal scores stay newest-first.
    for r in _pq_newest_first():
        body_lc, body_lcj = _pq_norm(r)
        head_lc, head_lcj = brain._derive_search_columns(
            " ".join([r.title or "", r.subject or "", r.tags or ""]))
        if mode == "phrase":
            if not phrase or (phrase not in body_lc and phrase not in head_lc):
                continue
            score = body_lc.count(phrase) + head_lc.count(phrase)
        else:  # word / all  (a single-word chip is just AND over one term)
            # Word-boundary matching, not substring: "care" must not match
            # "healthcare", nor "star" match "restart".
            if not roots or not all(
                    brain._root_present(x, body_lc, body_lcj)
                    or brain._root_present(x, head_lc, head_lcj) for x in roots):
                continue
            score = sum(brain._root_count(x, body_lc, body_lcj)
                        + brain._root_count(x, head_lc, head_lcj) for x in roots)
        out.append((score, _pq_card(r, roots=roots)))
    out.sort(key=lambda x: -x[0])
    return [p for _, p in out[:limit]]


_PQ_VOCAB_CACHE = {"key": None, "words": set()}


def _pq_vocab():
    """Cached set of meaningful words across all PQ tags + reply text — used to
    spell-correct typed queries (e.g. 'insrance' -> 'insurance'). Rebuilt only
    when the PQ set changes (keyed on row count + newest id)."""
    rows = _app.PqDocument.query.order_by(_app.PqDocument.id.desc()).all()
    key = (len(rows), rows[0].id if rows else 0)
    if _PQ_VOCAB_CACHE["key"] == key:
        return _PQ_VOCAB_CACHE["words"]
    words = set()
    for r in rows:
        blob = " ".join([r.tags or "", r.title or "", r.subject or "", r.body_text or ""]).lower()
        for w in re.findall(r"[a-z]{4,}", blob):
            if w not in _PQ_STOP:
                words.add(w)
    _PQ_VOCAB_CACHE["key"] = key
    _PQ_VOCAB_CACHE["words"] = words
    return words


def _pq_chips(query):
    """Smart Deep-Scan suggestion chips, mirroring the KB modules: a spell-fix
    chip when a word looks misspelled ('Did you mean "insurance"?'), the exact
    word as typed, an exact-phrase chip, and a 'Search All' chip."""
    vocab = _pq_vocab()
    seen, words, chips = set(), [], []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/]*", query or ""):
        wl = w.lower()
        if len(wl) <= 2 or wl in _PQ_STOP or wl in seen:
            continue
        seen.add(wl)
        words.append(w)
        # Spelling correction against the PQ vocabulary (deterministic, no LLM).
        if wl not in vocab and len(wl) > 3:
            m = difflib.get_close_matches(wl, vocab, n=1, cutoff=0.86)
            if m and m[0] != wl:
                chips.append({"label": f'Did you mean "{m[0]}"?', "mode": "word",
                              "text": m[0], "kind": "fix"})
        chips.append({"label": f'Search "{w}"', "mode": "word", "text": w, "kind": "keyword"})
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


@api_bp.post("/pq/reextract")
def api_pq_reextract():
    """Admin-only: re-derive body_text for every PQ from its stored original.

    body_text is the column search actually scans, and it is written once at
    upload time — so an extraction fix leaves every existing row stale. Rewrites
    body_text only; html, tags, departments and metadata are left untouched
    (nothing here is user-authored — api_pq_update never writes these two).
    Idempotent, so it can be re-run after any future extraction change.
    """
    if not _require_role("admin"):
        return jsonify({"message": "Admin access required"}), 403
    out, changed = [], 0
    for r in _app.PqDocument.query.order_by(_app.PqDocument.id).all():
        row = {"id": r.id, "pq_no": r.pq_no, "before": len(r.body_text or "")}
        # Derive the sort key from the date already on the row, before anything
        # that can bail out — it needs no source file, and a PQ whose original
        # has gone missing must still sort by date rather than sink to the end.
        iso = _iso_date(r.doc_date)
        if iso and iso != r.doc_date_iso:
            r.doc_date_iso = iso
            row["date_iso"] = iso
            changed += 1
        data = storage.load_pq(r.docx_filename) if r.docx_filename else None
        if data is None:
            row["status"] = "original file not found"
            out.append(row)
            continue
        try:
            parsed = parse_pq(data, filename=r.docx_filename)
        except Exception as e:
            row["status"] = f"{type(e).__name__}: {e}"
            out.append(row)
            continue
        if parsed["text"] != (r.body_text or ""):
            r.body_text = parsed["text"]
            changed += 1
        # Only fall back to the document's own date when the row carries none —
        # an admin's date_override on the row wins over what the file says.
        if not r.doc_date_iso and parsed["doc_date_iso"]:
            r.doc_date_iso = row["date_iso"] = parsed["doc_date_iso"]
            if not r.doc_date:
                r.doc_date = parsed["doc_date"]
            changed += 1
        row["after"], row["status"] = len(r.body_text or ""), "ok"
        out.append(row)
    _app.db.session.commit()
    # Keyed on (row count, newest id) — neither moves when body_text changes,
    # so the spell-check vocabulary has to be invalidated by hand.
    _PQ_VOCAB_CACHE["key"] = None
    _audit(f"Re-extracted body text for {changed} PQ(s)", "PQ re-extract")
    return jsonify({"ok": True, "changed": changed, "results": out})


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
            "category": str(g["Doc_Category"].iloc[0]) if "Doc_Category" in g.columns else "",
            "parent": (asset.parent_doc if asset else None),
            "status": (asset.status if asset and asset.status else "Active"),
            "effective_date": (asset.effective_date if asset else None),
            "clauses": int(len(g)),
            "edited": edited,
            "pdf_url": _doc_pdf_url(str(src)),
            "bundle": _doc_bundle(str(src)),
            "has_uploaded_pdf": bool(asset and asset.pdf_filename),
        })
    docs.sort(key=lambda d: d["source"].lower())
    return jsonify({"docs": docs})


@api_bp.post("/clause/doc-meta")
def api_clause_doc_meta():
    """Admin: rename a document and/or change its type-band / department."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    d = request.get_json(silent=True) or {}
    source = (d.get("source") or "").strip()
    new_source = (d.get("new_source") or "").strip()
    doc_type = (d.get("doc_type") or "").strip().upper()
    category = (d.get("category") or "").strip().upper()
    if not source:
        return jsonify({"ok": False, "message": "Missing document."}), 400
    ok, err = brain.update_document_meta(source, new_source or None, doc_type or None, category or None)
    if not ok:
        return jsonify({"ok": False, "message": err}), 409
    final = new_source or source
    # Persist doc-level hierarchy/status on the DocumentAsset (create if needed).
    m = _app
    asset = m.DocumentAsset.query.filter_by(source_doc=final).first()
    has_meta = any(k in d for k in ("parent", "status", "effective_date"))
    if not asset and has_meta:
        asset = m.DocumentAsset(source_doc=final, uploaded_at=datetime.utcnow())
        m.db.session.add(asset)
    if asset:
        if "parent" in d:
            p = (d.get("parent") or "").strip()
            asset.parent_doc = (p if p and p != final else None)
        if "status" in d:
            asset.status = ((d.get("status") or "").strip() or "Active")
        if "effective_date" in d:
            asset.effective_date = ((d.get("effective_date") or "").strip() or None)
        m.db.session.commit()
    renamed = bool(new_source and new_source != source)
    _audit(f"Edited document '{source}'" + (f" → '{final}'" if renamed else ""), "Document meta")
    return jsonify({"ok": True, "source": final})


@api_bp.get("/clause/specs")
def api_clause_specs():
    """Admin: available document-type specs for PDF import."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    import ingest
    return jsonify({"specs": ingest.list_specs()})


# PDF extraction (pdfplumber) is memory-heavy and doesn't return its memory to the
# OS, so it runs in a fresh short-lived SUBPROCESS whose memory is fully reclaimed on
# exit (see ingest_worker.py). A lock serialises imports so only one ~650 MB child
# exists at a time.
import threading as _threading
_INGEST_LOCK = _threading.Lock()
_INGEST_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingest_worker.py")


def _run_ingest_worker(args, timeout=240):
    """Run the PDF ingestion worker in a clean subprocess and return its JSON result.
    Raises RuntimeError on timeout, crash, or a structured worker error."""
    import subprocess, sys, tempfile, json
    out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    out.close()
    cmd = [sys.executable, _INGEST_WORKER, *args, out.name]
    try:
        with _INGEST_LOCK:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                                  cwd=os.path.dirname(_INGEST_WORKER))
        try:
            with open(out.name, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (ValueError, OSError):
            tail = proc.stderr[-500:].decode("utf-8", "replace") if proc.stderr else ""
            raise RuntimeError(f"PDF worker crashed (exit {proc.returncode}). {tail}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("PDF processing timed out — the file may be very large.")
    finally:
        try: os.unlink(out.name)
        except OSError: pass
    if not data.get("ok"):
        raise RuntimeError(data.get("error") or "PDF worker error")
    return data


@api_bp.post("/clause/detect-spec")
def api_clause_detect_spec():
    """Admin: rank existing specs by how well they segment the uploaded PDF."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    import tempfile
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "message": "Upload a PDF"}), 400
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(f.read()); tmp.close()
        ranked = _run_ingest_worker(["detect", tmp.name])["ranked"]
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
        _res = _run_ingest_worker(["segment", tmp.name, spec_id])
        rows = _res.get("rows") or []
        report = _res.get("report") or {}
        spec_errors = _res.get("spec_errors") or []
    except Exception as e:
        print(f"PDF import error: {e}")
        return jsonify({"ok": False, "message": "Could not segment that PDF."}), 400
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass

    if not rows:
        return jsonify({"ok": False, "message": "No clauses were found — wrong spec for this document?"}), 400

    # Insert the produced clauses as a new document. Wrapped so a DB error surfaces
    # as a clear message (and rolls back) instead of a bare 500 "import failed".
    conn = _sql.connect(brain.DB_NAME)
    try:
        for i, r in enumerate(rows, 1):
            clause = str(r.get("clause", ""))
            header = clause.split("\n", 1)[0].rstrip(":")[:120]
            is_header = 1 if clause.strip().endswith(":") and "\n" not in clause.strip() else 0
            conn.execute(
                "INSERT INTO regulatory_clauses (source_doc, doc_category, doc_type, clause_id, "
                "clause_text, context_header, regulatory_tags, priority, is_header, sort_order) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (source, category, doc_type, str(r.get("id", "")), clause, header,
                 str(r.get("tag", "")).replace("_", " "), 99, is_header, i * 10))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"PDF import DB error: {e}")
        return jsonify({"ok": False, "message": f"Could not save clauses: {e}"}), 400
    finally:
        try: conn.close()
        except Exception: pass

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
    dup_ids = report.get("duplicate_ids", [])
    return jsonify({"ok": True, "source": source, "clauses": len(rows),
                    "orphans": report.get("orphan_lines", 0),
                    "duplicates": len(dup_ids), "duplicate_ids": dup_ids,
                    "spec_errors": spec_errors})


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
        html = brain.clause_html(cid, source) or _clause_text_to_html(str(r.get("Clause_Text", "")), bold_headings=False)
        edited = bool(has_upd and pd.notna(r.get("updated_by")) and str(r.get("updated_by")).strip())
        out.append({"id": cid, "html": html, "edited": edited,
                    "tags": [t.strip() for t in str(r.get("Regulatory_Tags", "")).split(",") if t.strip()]})
    return jsonify({"source": source, "clauses": out, "rev": brain.doc_revision(source)})


def _clause_toc_title(html, text):
    """A short, plain-text label for a clause in the reader's table of contents."""
    raw = (text or "").strip()
    if not raw:
        raw = re.sub(r"<[^>]+>", " ", html or "")
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:90]


@api_bp.get("/clause/readable")
def api_clause_readable():
    """Any signed-in user: documents available to read in full (from the KB)."""
    if not _app.current_user.is_authenticated:
        return jsonify({"docs": []}), 401
    df = brain.load_knowledge_base()
    if df is None or df.empty:
        return jsonify({"docs": []})
    docs = []
    for src, g in df.groupby("Source_Doc"):
        asset = _app.DocumentAsset.query.filter_by(source_doc=str(src)).first()
        docs.append({
            "source": str(src),
            "type": str(g["Doc_Type"].iloc[0]) if "Doc_Type" in g.columns else "",
            "category": str(g["Doc_Category"].iloc[0]) if "Doc_Category" in g.columns else "",
            "status": (asset.status if asset and asset.status else "Active"),
            "effective_date": (asset.effective_date if asset else None),
            "clauses": int(len(g)),
            "pdf_url": _doc_pdf_url(str(src)),
            "bundle": _doc_bundle(str(src)),
        })
    docs.sort(key=lambda d: d["source"].lower())
    return jsonify({"docs": docs})


@api_bp.get("/clause/doc-read")
def api_clause_doc_read():
    """Any signed-in user: every clause of a document in order, for the full-doc
    reader (read-only; the editor uses /clause/doc-full)."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    source = (request.args.get("source") or "").strip()
    df = brain.load_knowledge_base()
    if df is None or df.empty:
        return jsonify({"source": source, "clauses": []})
    sub = df[df["Source_Doc"].astype(str) == source]
    if sub.empty:
        return jsonify({"ok": False, "message": "Document not found."}), 404
    if "sort_order" in sub.columns:
        sub = sub.sort_values("sort_order", kind="stable", na_position="last")
    out = []
    for _, r in sub.iterrows():
        cid = str(r.get("Clause_ID", "")).strip()
        text = str(r.get("Clause_Text", ""))
        html = brain.clause_html(cid, source) or _clause_text_to_html(text)
        out.append({"id": cid, "title": _clause_toc_title(html, text), "html": html,
                    "tags": [t.strip() for t in str(r.get("Regulatory_Tags", "")).split(",") if t.strip()]})
    asset = _app.DocumentAsset.query.filter_by(source_doc=source).first()
    return jsonify({
        "source": source,
        "type": str(sub["Doc_Type"].iloc[0]) if "Doc_Type" in sub.columns else "",
        "category": str(sub["Doc_Category"].iloc[0]) if "Doc_Category" in sub.columns else "",
        "status": (asset.status if asset and asset.status else "Active"),
        "effective_date": (asset.effective_date if asset else None),
        "pdf_url": _doc_pdf_url(source),
        "bundle": _doc_bundle(source),
        "clauses": out,
    })


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
            storage.delete_doc_bundle(asset.bundle_filename)
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


@api_bp.post("/clause/doc-bundle")
def api_doc_bundle_upload():
    """Editor: attach/replace the supplementary bundle for a document (the
    annexure ZIP, a form workbook, etc.). Kept apart from the source PDF, which
    the viewer renders inline."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    source = (request.form.get("source") or "").strip()
    f = request.files.get("file")
    ext = os.path.splitext((f.filename if f else "") or "")[1].lower()
    if not source or not f or ext not in _BUNDLE_TYPES:
        return jsonify({"ok": False, "message":
                        "Provide a document and a %s file." % ", ".join(sorted(_BUNDLE_TYPES))}), 400
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", source) + "_annexures" + ext
    m = _app
    asset = m.DocumentAsset.query.filter_by(source_doc=source).first()
    if asset is None:
        asset = m.DocumentAsset(source_doc=source)
        m.db.session.add(asset)
    # Replacing with a DIFFERENT extension would otherwise orphan the old blob.
    old = asset.bundle_filename
    storage.save_doc_bundle(safe, f.read(), _BUNDLE_TYPES[ext])
    if old and old != safe:
        storage.delete_doc_bundle(old)
    asset.bundle_filename = safe
    asset.bundle_uploaded_at = datetime.utcnow()
    m.db.session.commit()
    g._iris_assets = None            # this request just changed it
    _audit(f"Attached annexure bundle to '{source}'", "Document bundle")
    return jsonify({"ok": True, "bundle": _doc_bundle(source)})


@api_bp.delete("/clause/doc-bundle")
def api_doc_bundle_delete():
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    source = (request.args.get("source") or "").strip()
    asset = _app.DocumentAsset.query.filter_by(source_doc=source).first()
    if asset and asset.bundle_filename:
        storage.delete_doc_bundle(asset.bundle_filename)
        asset.bundle_filename = None
        asset.bundle_uploaded_at = None
        _app.db.session.commit()
        _audit(f"Removed annexure bundle from '{source}'", "Document bundle")
    return jsonify({"ok": True})


@api_bp.get("/doc-bundle/<int:aid>/download")
def api_doc_bundle_serve(aid):
    """Stream a document's annexure bundle as a download (never inline)."""
    if not _app.current_user.is_authenticated:
        return jsonify({"ok": False}), 401
    a = _app.DocumentAsset.query.get_or_404(aid)
    if not a.bundle_filename:
        return jsonify({"ok": False, "message": "No bundle attached."}), 404
    data = storage.load_doc_bundle(a.bundle_filename)
    if data is None:
        return jsonify({"ok": False, "message": "Bundle not found."}), 404
    ext = os.path.splitext(a.bundle_filename)[1].lower()
    return send_file(io.BytesIO(data),
                     mimetype=_BUNDLE_TYPES.get(ext, "application/octet-stream"),
                     as_attachment=True, download_name=a.bundle_filename)


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


def _clause_text_to_html(text, bold_headings=True):
    """Render a legacy plain-text/markdown clause: lines -> <p>, a line ending in
    ':' -> bold heading, GFM table blocks -> <table>. Pass bold_headings=False to
    seed the Studio editor plain (so bold is then controlled manually there)."""
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
        if bold_headings and s.endswith(":"):
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
        existing = _clause_text_to_html(txt, bold_headings=False)
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
    """Admin: edit a PQ's title, date, tags and/or departments."""
    if not _require_role("editor"):
        return jsonify({"message": "Editor access required"}), 403
    r = _app.PqDocument.query.get_or_404(pid)
    data = request.get_json(silent=True) or {}
    if "title" in data and (data.get("title") or "").strip():
        r.title = data["title"].strip()[:400]
    if "date" in data:
        r.doc_date = (data.get("date") or "").strip()[:40] or None
        r.doc_date_iso = _iso_date(r.doc_date) or None   # keep the sort key in step
    if "tags" in data:
        r.tags = (data.get("tags") or "").strip()
    if "departments" in data:
        r.departments = ",".join(_norm_departments(data.get("departments")))
    _app.db.session.commit()
    _audit(f"Edited PQ {r.pq_no or pid}", "PQ edit")
    return jsonify({"ok": True, "title": r.title, "date": r.doc_date,
                    "tags": [t.strip() for t in (r.tags or "").split(",") if t.strip()],
                    "departments": _dept_list(r)})


@api_bp.get("/clause-suggest")
def api_clause_suggest():
    """Typeahead for the '/'-prefixed clause-number search."""
    q = request.args.get("q", "")
    module = (request.args.get("module") or "universal").strip()
    sources = request.args.getlist("source") or None
    KB_DF = brain.load_knowledge_base()
    rows = brain.search_by_clause_number(q, brain.filter_df_by_module(KB_DF, module), sources=sources, limit=30)
    out = [{"id": r["id"], "source": r["source"], "type": r["type"],
            "snippet": (str(r["raw_text"])[:90] + ("…" if len(str(r["raw_text"])) > 90 else ""))}
           for r in rows]
    return jsonify({"suggestions": out})


def _highlight_phrases(tuples):
    """Searched terms as phrases (each a list of word stems) for client-side
    highlighting. A multi-word tag like 'free look period' stays one phrase so it
    is matched as a unit (not standalone 'period'); stems let it light up word
    families (migration->migrate). Digit/symbol tokens like '10%' are kept verbatim."""
    phrases, seen = [], set()
    for (raw, _clean) in tuples:
        words = [w for w in re.findall(r"[A-Za-z0-9%]+", str(raw)) if len(w) >= 2]
        # search_root, not the bare stem: Porter can restore letters so the stem
        # isn't a prefix of the word ("pricing" -> "price"), and the client matches
        # by `root[a-z]*` — so "price" would fail to highlight "pricing".
        stems = [s for s in (brain.search_root(w.lower()) for w in words) if s]
        key = " ".join(stems)
        if stems and key not in seen:
            seen.add(key)
            phrases.append(stems)
    return phrases


# Function words that must NOT be highlighted on their own — lighting up "to",
# "be", "as", "shall" in every result is noise. Shared with the search-keyword
# extractor (brain.FUNCTION_WORDS) so a word that can't be highlighted also can't
# drive results (and vice-versa) — the green whole-phrase highlight still covers
# the verbatim phrase regardless.
_HL_STOP = brain.FUNCTION_WORDS


def _word_highlights(query, stop=None):
    """The query's individual CONTENT words as single-word highlight phrases
    (stopwords + tiny tokens dropped), so each meaningful word lights up on its own
    wherever it appears — even when the user typed a multi-word tag like
    'mental illness'. Distinct from the green whole-phrase highlight.

    `stop` lets a corpus supply its own stoplist (PQ passes _PQ_STOP), so that what
    lights up is always exactly what could have matched in that corpus.
    """
    stop = _HL_STOP if stop is None else stop
    out, seen = [], set()
    for w in re.findall(r"[A-Za-z0-9%]+", str(query).lower()):
        if w in stop:
            continue
        if len(w) < 3 and w.isalpha():          # tiny function words ("be", "as")
            continue
        s = brain.search_root(w) if w.isalpha() else w
        if s and s not in seen:
            seen.add(s)
            out.append([s])
    return out




_PHRASE_TYPE_RANK = {"ACT": 0, "REGULATION": 1, "MASTER": 2, "GUIDELINE": 3, "CIRCULAR": 4}

def _phrase_sort_key(m, phrase, phrase_join):
    """Rank exact-phrase matches so the most relevant clause leads:
      1. the clause actually CONTAINS the verbatim phrase (the literal source) —
         this trumps everything, so copy-pasting a sentence finds its own clause;
      2. then a clause TAGGED with the phrase (curated as being about it);
      3. then the phrase in the clause HEADING;
      4. then document authority (Act > Regulation > …);
      5. then the shorter/more-focused clause over a mention in a long schedule."""
    def _hit(field):
        # tags are stored underscore-joined ("Free_Look_Period") — normalize to
        # spaces so the phrase matches; also hyphen-insensitive.
        v = str(m.get(field, "")).lower().replace("_", " ")
        return phrase in v or phrase_join in v.replace("-", "")
    verbatim = 0 if _hit("raw_text") else 1
    in_tag = 0 if _hit("tag") else 1
    in_head = 0 if _hit("header") else 1
    return (verbatim, in_tag, in_head,
            _PHRASE_TYPE_RANK.get(str(m.get("type", "")).upper(), 9),
            m.get("priority") or 99,
            len(str(m.get("raw_text", ""))),
            str(m.get("id", "")))


def _spell_correct_phrase(query):
    """Word-by-word spelling correction of a phrase against the KB vocabulary.
    Returns the corrected phrase when something changed, else None. Conservative:
    a word is only rewritten when it's genuinely unknown (not in the vocab AND its
    root doesn't already prefix a known word), so valid words are never mangled."""
    vocab = getattr(brain, "KNOWN_VOCAB", None)
    if not vocab:
        return None
    out, changed = [], False
    for w in query.split():
        wl = w.lower()
        if len(wl) < 3 or not wl.isalpha() or wl in vocab:
            out.append(w); continue
        rs = brain.get_stem(wl)
        if len(rs) >= 3 and any(v.startswith(rs) for v in vocab):
            out.append(w); continue                       # already findable as typed
        mm = difflib.get_close_matches(wl, list(vocab), n=1, cutoff=0.84)
        out.append(mm[0] if (mm and mm[0] != wl) else w)
        changed = changed or bool(mm and mm[0] != wl)
    return " ".join(out) if changed else None


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
        is_raw = not keyword_tuples
        if is_raw:
            # Raw free-text deep scan (the "/deep <terms>" command sends plain text,
            # not the chip's word|stem||… payload). Build (word, root) tuples from the
            # literal words so it scans for exactly those terms.
            _stop = getattr(brain, "STOP_WORDS", set())
            keyword_tuples = [(w, brain.search_root(w.lower()))
                              for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", raw_payload)
                              if len(w) > 1 and w.lower() not in _stop]
        display_kws = [t[0] for t in keyword_tuples]
        tag_matches = brain.search_tags_only(keyword_tuples, KB_DF, module=module)
        # The "/deep" command is a strict standalone scan (search ALL clauses); the
        # chip variant is "additional to the tag matches already on screen".
        exclude_ids = [] if is_raw else [m["id"] for m in tag_matches]
        matches = _scope(brain.deep_scan_brain(keyword_tuples, KB_DF, exclude_ids=exclude_ids, module=module))
        # STRICT — a multi-word deep scan is a PHRASE, not scattered terms: the words
        # must appear TOGETHER (adjacent), never separately across the clause. So
        # "projected yield" won't match a clause that merely has "projected" and
        # "yield" in different places. No fallback — no phrase, no results.
        _dphrase = re.sub(r"\s+", " ",
                          (raw_payload if is_raw else " ".join(display_kws)).strip().lower())
        _deep_hl = _highlight_phrases(keyword_tuples)
        _deep_phrase_hl = []
        if " " in _dphrase:
            _dj = _dphrase.replace("-", "")
            matches = [m for m in matches
                       if _dphrase in str(m.get("raw_text", "")).lower()
                       or _dj in str(m.get("raw_text", "")).lower().replace("-", "")]
            display_kws = [_dphrase]   # show the phrase, not comma-split words
            # Highlight ONLY the consecutive phrase (not every occurrence of each
            # word), in its own colour. Split on WHITESPACE so punctuated tokens
            # survive intact — e.g. "40.1.4.2.2" must stay one token, or the
            # consecutive regex can't match it. wordPattern() on the client matches
            # digit/symbol tokens literally.
            _deep_phrase_hl = [_dphrase.split()]
            _deep_hl = []
            # Rank DEDICATED clauses first — the phrase in a clause's heading (a
            # clause ABOUT it) beats a passing mention buried in a long schedule/
            # table; then by document authority, then shorter (more focused) first.
            matches.sort(key=lambda m: _phrase_sort_key(m, _dphrase, _dj))
        # Record deep scans too (with a readable label + result count).
        try:
            email = _app.current_user.email if _app.current_user.is_authenticated else None
            _app._record_search(email, module, "[Deep Scan] " + ", ".join(display_kws), len(matches))
        except Exception:
            pass
        return jsonify({
            "ok": True, "module": module, "kind": "deep_scan",
            "query_label": "Deep Scan",
            "keywords": display_kws, "highlight": _deep_hl, "highlight_phrase": _deep_phrase_hl,
            "matches": [_match_payload(m) for m in matches],
            "chips": [],
            "note": None if matches else f"No results found for: {', '.join(display_kws)}.",
        })

    # --- Clause-number lookup (query starts with "/") ---
    if query.lstrip().startswith("/"):
        matches = _scope(brain.search_by_clause_number(query, brain.filter_df_by_module(KB_DF, module), sources=sources))
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

    # --- Regulatory citation / reference code (e.g. "IRDAI/Actl/IBNR/AIC/2009-10") ---
    # A slash-joined code is ONE reference, not a bag of keywords. Tokenising it
    # ("irdai", "aic", "10", …) explodes into unrelated tag hits, so match the literal
    # code in the clause text instead; if it isn't present, say so (no random noise).
    _cite = query.strip()
    if ("/" in _cite and " " not in _cite
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/().\-]*", _cite)
            and (any(ch.isdigit() for ch in _cite) or _cite.count("/") >= 2)):
        matches = _scope(brain.search_citation(_cite, brain.filter_df_by_module(KB_DF, module)))
        try:
            email = _app.current_user.email if _app.current_user.is_authenticated else None
            _app._record_search(email, module, query, len(matches))
        except Exception:
            pass
        return jsonify({
            "ok": True, "module": module, "kind": "tags",
            "query_label": query, "keywords": [_cite],
            "highlight": [], "highlight_phrase": [[_cite]], "phrase": _cite,
            "phrase_matches": [_match_payload(m) for m in matches],
            "matches": [], "content_matches": [],
            "chips": [], "note": None if matches else f"No clause references “{_cite}”.",
        })

    # --- Standard search ---
    kw_tuples = brain.get_clean_keywords(query)
    display_kws = [t[0] for t in kw_tuples]
    if not kw_tuples:
        return jsonify({"ok": True, "module": module, "kind": "rejected",
                        "query_label": query, "matches": [], "chips": [], "keywords": [],
                        "note": "Query rejected. Please use regulatory terms."})

    phrase_l = re.sub(r"\s+", " ", (query or "").strip().lower())
    phrase_join = phrase_l.replace("-", "")
    # The distinct word-roots the user typed (skip stopwords + synthesized tags).
    core_roots = []
    for raw, clean in kw_tuples:
        if " " in str(clean) or clean in brain.STOPWORDS_STRONG:
            continue
        r = brain.search_root(str(raw).lower(), clean)
        if r and r not in core_roots:
            core_roots.append(r)
    # "Multi-token" isn't just whitespace: a slash/hyphen compound like "actl/ibnr"
    # or "free-look" is several tokens too, and the clause that contains it verbatim
    # must be promoted to the top tier (not left ranked by score). Treat any
    # separator BETWEEN word characters as multi.
    multi = " " in phrase_l or bool(re.search(r"[A-Za-z0-9][/\-][A-Za-z0-9]", phrase_l))

    # Rank the candidate pools by relevance BEFORE capping them. The caps below
    # exist to bound the O(n^2) run scoring, but whatever they cut is gone for
    # good — so cutting in the upstream scan's order silently discards the best
    # answers. Searching "charged to shareholders account", 251 of 383 candidates
    # matched a single word and crowded out the 115 matching two; the clause that
    # answered the question ("charged ... from shareholder's fund", two of three
    # words in ONE sentence) sat at #116 and never survived to be scored at all.
    # run_len_cap=0 disables the expensive consecutive-run scan for this pass, so
    # pre-ranking every candidate stays cheap; the survivors are scored in full
    # (runs included) by the promotion loop below.
    # Resolve the prefix index ONCE per request, into the (source, clause_id) keys
    # the scorer already uses to memoize. Both scorers below share it, so the cost
    # is one pass over the postings of ~10 roots rather than a regex per
    # (root, candidate) — and the pre-ranking pass runs over every candidate.
    # Built with the SAME normalisation brain uses for its match dicts
    # (Source_Doc verbatim, Clause_ID str()+strip()), or the keys would not meet.
    _root_keys, _known_keys, _root_masks = None, None, None
    try:
        _lab_key = {lab: (src, str(cid).strip()) for lab, src, cid
                    in zip(KB_DF.index, KB_DF["Source_Doc"], KB_DF["Clause_ID"])}
        _known_keys = set(_lab_key.values())
        _root_keys, _root_masks = {}, {}
        for r in core_roots:
            labs = brain.clauses_with_root(r)
            _root_keys[r] = None if labs is None else {_lab_key[l] for l in labs if l in _lab_key}
            _sm = brain.sentence_masks(r)
            _root_masks[r] = None if _sm is None else {
                _lab_key[l]: b for l, b in _sm.items() if l in _lab_key}
    except Exception:
        _root_keys, _known_keys, _root_masks = None, None, None   # any doubt -> regex

    _prescore, _ = _make_scorer(phrase_l, core_roots, multi, run_len_cap=0,
                                root_keys=_root_keys, known_keys=_known_keys,
                                root_masks=_root_masks)

    def _by_relevance(matches):
        return sorted(matches, key=lambda m: tuple(-v for v in _prescore(m)))

    # A common-word query can match hundreds of tagged clauses; cap the candidate
    # pool so the O(n^2) run-length scoring in the promotion loop runs over a
    # bounded set (not ~350 clauses).
    TAG_CAND_CAP = 60
    tag_matches = _by_relevance(
        _scope(brain.search_tags_only(kw_tuples, KB_DF, module=module)))[:TAG_CAND_CAP]

    # Windows-Explorer-style: also auto-scan the clause TEXT (no manual "deep scan"
    # step), returned as a separate, clearly-subordinate tier.
    #
    # Two different caps, because they answer two different questions. CONTENT_SCAN
    # is how far down the promotion loop below may look for a clause that actually
    # answers the query; CONTENT_CAP is how many unpromoted clauses this subordinate
    # tier renders. Collapsing them into one number is what hid the "shareholder's
    # fund" clause: it pre-ranked 49th, so a single cap of 20 discarded it before
    # promotion could ever see it.
    #
    # CONTENT_SCAN scales INVERSELY with query length, because cost and usefulness
    # run in opposite directions. A short query is unselective — half the corpus
    # contains "premium" — so promotion reaches deep and every slot earns its place,
    # but each candidate is cheap to test. A conversational query is selective:
    # _promote needs ALL the roots, so almost nothing qualifies past the first
    # handful, yet each candidate costs proportionally more to evaluate (one
    # _root_present pass per root). Measured over 17 queries: the promotion loop
    # spends 2-103ms at <=5 roots but 284-418ms at 8+, while the deepest rank it
    # ever promotes at 8+ roots is 27. A flat 60 pays the expensive case in full for
    # depth it cannot use.
    #
    # Keep expectations honest: this trims the PROMOTION LOOP by ~25% on long
    # queries, but that loop is only ~5% of a universal-module request (the content
    # scan and the pre-rank dominate at ~2s). End-to-end this is worth ~3-5% on
    # conversational queries and exactly 0 on short ones. It is a free tidy-up, NOT
    # a fix for long-query latency — that lives in the scan.
    #
    # The floor is 40, not 20 or 30, for two independent reasons:
    #   - "premium payment options available to a policyholder under a health
    #     insurance policy issued by a general insurer" has 11 roots but promotes as
    #     deep as rank 27 — the one long query that behaves like a short one.
    #   - The subordinate content tier is drawn from what SURVIVES promotion, so it
    #     starves if the pool is too small: at 30, a 19-word universal query lost 2
    #     of its 20 content cards (CONTENT_CAP 20 + ~12 promoted out needs >32).
    #     Verified: 35 and 40 both reproduce the pre-change output byte-for-byte on a
    #     42-response suite; 40 keeps headroom for queries outside it, and costs ~1%
    #     against 35 — noise.
    # And do NOT lower the 60: at <=5 roots it is near-free, and 25 loses results on
    # 7 of 12 short/mid queries ("cashless claim settlement timeline" promotes all 60,
    # i.e. it is already saturated).
    CONTENT_SCAN, CONTENT_CAP = (60 if len(core_roots) <= 6 else 40), 20
    content_matches = []
    if module != "data":
        try:
            content_matches = _by_relevance(_scope(brain.deep_scan_brain(
                kw_tuples, KB_DF, exclude_ids=[m["id"] for m in tag_matches],
                module=module, phrase=query)))[:CONTENT_SCAN]
        except Exception:
            content_matches = []

    # Verbatim-phrase promotion: when the user types a multi-word phrase that
    # appears literally in a clause (e.g. "premium rate"), that clause should rank
    # FIRST — above concept/tag matches — instead of being buried in the text tier.
    # Pull such clauses out of both tiers into a dedicated top "exact phrase" tier,
    # ordered by regulatory hierarchy (Act > Regulation > Circular).
    # Best-match promotion: a clause that contains the verbatim phrase the user
    # typed, OR all of the distinct words they typed, is more relevant than a
    # concept/tag match on just one (often common) word — e.g. for "empanelment
    # process" the clause with BOTH should beat clauses tagged only "process".
    # Pull such clauses out of both tiers into a dedicated top tier.
    phrase_matches = []
    _phrase_note = None
    # phrase_l, phrase_join, core_roots and multi are computed further up — the
    # candidate pre-ranking needs them before the caps are applied.
    # run_len_cap 10: the consecutive-run scan is O(n^2) in the query and now runs
    # over a larger candidate pool, which took a 12-content-word query to 2.8s. Past
    # ~10 content words the run adds nothing anyway — a verbatim or all-words match
    # already decides those queries — so it is skipped there and kept for the short
    # and mid-length queries whose typo tolerance depends on it.
    _score, _promote = _make_scorer(phrase_l, core_roots, multi, run_len_cap=10,
                                    root_keys=_root_keys, known_keys=_known_keys,
                                    root_masks=_root_masks)

    # Run for multi-word phrases AND single content words: a clause containing the
    # exact query text is a "best match" and must rank above stem-family / loosely-
    # tagged clauses that don't contain the word at all (e.g. searching "promotion"
    # should surface the clause whose heading is "Promotion" before clauses that only
    # match the stem "promote" or a related tag).
    if multi or core_roots:
        tag_keep, content_keep = [], []
        for m in tag_matches:
            (phrase_matches if _promote(m) else tag_keep).append(m)
        for m in content_matches:
            (phrase_matches if _promote(m) else content_keep).append(m)
        tag_matches, content_matches = tag_keep, content_keep
        # Order the best-match (phrase) tier:
        #   verbatim phrase > tagged with it > longest CONSECUTIVE run > in heading >
        #   most words in ONE sentence > more words covered overall > document
        #   authority > more focused clause.
        # The consecutive-run term makes ranking degrade gracefully to typos: one
        # wrong letter ("accordance"->"accordam") still keeps the near-verbatim
        # clause on top instead of scattering to unrelated single-word matches.
        # Sentence coverage outranks whole-clause coverage because people describe
        # what they want in their own words, and the clause that answers them says
        # it in one place — whereas a long omnibus clause can contain every word
        # they typed and answer none of it.
        def _reg_key(m):
            pk = _phrase_sort_key(m, phrase_l, phrase_join)  # (verbatim,tag,head,type,pri,len,id)
            _v, line_cover, _aw, run, cover = _score(m)
            return (pk[0], pk[1], -run, pk[2], -line_cover, -cover,
                    pk[3], pk[4], pk[5], pk[6])
        phrase_matches.sort(key=_reg_key)

    # Bound the rendered result set. Hundreds of full clause cards make the results
    # view render for many seconds (huge DOM + thousands of highlight <mark>s) and jank
    # the whole page. Keep the most relevant of each tier — all three are sorted
    # best-first by now, and the content tier is only cut HERE, after promotion has
    # had the chance to pull a genuine answer out of it.
    phrase_matches = phrase_matches[:30]
    tag_matches = tag_matches[:20]
    content_matches = content_matches[:CONTENT_CAP]

    # Record the query for admin usage visibility (best-effort, non-blocking).
    try:
        email = _app.current_user.email if _app.current_user.is_authenticated else None
        _app._record_search(email, module, query,
                            len(phrase_matches) + len(tag_matches) + len(content_matches))
    except Exception:
        pass

    note = _phrase_note
    if not note and not phrase_matches and not tag_matches and not content_matches:
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
        # Yellow per-word highlight: each meaningful query word on its own (drops
        # stopwords + multi-word-tag grouping, so "mental" and "illness" both light
        # up, and filler like "to"/"be"/"as" never does).
        "highlight": _word_highlights(query),
        # Green "verbatim" highlight: the exact phrase (multi-word) or exact word
        # (single) as ONE unit — one highlight pattern. We deliberately do NOT emit
        # contiguous sub-runs: a long query (e.g. a 27-word sentence) produced ~90
        # patterns, and building/applying that regex per clause per render froze the
        # results view. Individual query words still light up yellow via `highlight`,
        # so a paraphrase isn't left un-highlighted — it just doesn't get the green
        # run. Stem-family cousins (e.g. "price" for "pricing") stay yellow too.
        "highlight_phrase": [phrase_l.split()] if phrase_l else [],
        "phrase": phrase_l if " " in phrase_l else None,
        "phrase_matches": [_match_payload(m) for m in phrase_matches],
        "matches": [_match_payload(m) for m in tag_matches],
        "content_matches": [_match_payload(m) for m in content_matches],
        "chips": _build_chips(kw_tuples, query),
        "note": note,
    })


@api_bp.get("/vocab")
def api_vocab():
    brain.load_knowledge_base()
    return jsonify(brain.get_autocomplete_data())


# --- Departments (Knowledge Base modules) ----------------------------------
_RESERVED_DEPT = {"GENERAL", "UNIVERSAL", "ALL", "DATA"}


@api_bp.get("/departments")
def api_departments():
    """Department list (drives the sidebar modules + category pickers)."""
    if not _app.current_user.is_authenticated:
        return jsonify({"departments": []}), 401
    depts = brain.load_departments()
    return jsonify({"departments": depts,
                    "categories": [d["key"] for d in depts] + ["GENERAL"]})


@api_bp.post("/admin/departments")
def api_departments_save():
    """Admin: add / edit / remove a department (persists to departments.json)."""
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    d = request.get_json(silent=True) or {}
    action = (d.get("action") or "").strip().lower()
    depts = brain.load_departments()
    by_key = {x["key"]: x for x in depts}
    order = [x["key"] for x in depts]
    if action in ("add", "edit"):
        key = re.sub(r"[^A-Z0-9]", "", (d.get("key") or "").strip().upper())
        label = (d.get("label") or "").strip()
        icon = (d.get("icon") or "").strip() or "fa-folder"
        if not key:
            return jsonify({"ok": False, "message": "A department code is required (letters/numbers only)."}), 400
        if key in _RESERVED_DEPT:
            return jsonify({"ok": False, "message": f"'{key}' is a reserved name."}), 400
        if action == "add" and key in by_key:
            return jsonify({"ok": False, "message": f"Department '{key}' already exists."}), 409
        general = bool(d.get("general"))
        by_key[key] = {"key": key, "label": label or key.title(), "icon": icon, "general": general}
        if key not in order:
            order.append(key)
        new_list = [by_key[k] for k in order]
    elif action == "delete":
        key = (d.get("key") or "").strip().upper()
        if key not in by_key:
            return jsonify({"ok": False, "message": "Department not found."}), 404
        new_list = [x for x in depts if x["key"] != key]
    else:
        return jsonify({"ok": False, "message": "Unknown action."}), 400
    saved = brain.save_departments(new_list)
    _audit(f"Department {action}: {d.get('key')}", "Departments")
    return jsonify({"ok": True, "departments": saved})


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
    tag_clauses = {}   # normalised tag -> [{id, source}] for the suggestion dropdown
    if scoped is not None and not scoped.empty:
        for _, row in scoped.iterrows():
            src = str(row.get("Source_Doc") or "").strip()
            if not src:
                continue
            src_type.setdefault(src, str(row.get("Doc_Type") or "UNKNOWN").strip().upper() or "UNKNOWN")
            cid = str(row.get("Clause_ID") or "").strip()
            raw_tags = str(row.get("Regulatory_Tags") or "")
            if raw_tags:
                tset = src_tags.setdefault(src, set())
                for t in raw_tags.split(","):
                    clean = t.strip().replace("_", " ").lower()
                    if len(clean) >= 2:
                        tset.add(clean)
                        lst = tag_clauses.setdefault(clean, [])
                        if cid and not any(e["id"] == cid and e["source"] == src for e in lst):
                            lst.append({"id": cid, "source": src})

    by_type = {}
    for src, typ in src_type.items():
        by_type.setdefault(typ, []).append(src)
    for docs in by_type.values():
        docs.sort()

    ordered_types = [t for t in _DOC_TYPE_ORDER if t in by_type]
    ordered_types += [t for t in by_type if t not in _DOC_TYPE_ORDER]
    groups = [{"type": t, "label": _DOC_TYPE_LABELS.get(t, t.title()), "docs": by_type[t]} for t in ordered_types]
    doc_tags = {src: sorted(tags) for src, tags in src_tags.items()}
    return jsonify({"module": module, "groups": groups, "doc_tags": doc_tags, "tag_clauses": tag_clauses})


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
            "add_total": bool(data.get("add_total", False)),
        }
    return {
        "dimension": request.form.get("dimension", "Insurer"),
        "entities": request.form.getlist("entities"),
        "metrics": request.form.getlist("metrics"),
        "years": request.form.getlist("years"),
        "quarters": request.form.getlist("quarters"),
        "lobs": request.form.getlist("lobs"),
        "classes": request.form.getlist("classes"),
        "add_total": request.form.get("add_total") in ("1", "true", "True", "on"),
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
    pw_err = _password_error(password, password)
    if pw_err:
        return jsonify({"message": pw_err}), 400
    if m.User.query.filter(m.db.func.lower(m.User.email) == email).first():
        return jsonify({"message": "User already exists."}), 409
    m.db.session.add(m.User(email=email, password_hash=m.hash_password(password),
                            is_active=True, is_admin=(role == "admin"), role=role))
    m.db.session.commit()
    m._record_admin_audit(email, f"user_create ({role})", "success")
    return jsonify({"ok": True, "message": "User created successfully."}), 201


def _active_admin_count():
    return _app.User.query.filter_by(is_admin=True, is_active=True).count()

def _would_orphan_admin(user):
    """True if removing/demoting/deactivating this user would leave zero active
    admins — guards against accidentally locking everyone out of the admin panel."""
    return bool(user.is_admin and user.is_active and _active_admin_count() <= 1)


@api_bp.post("/admin/user/<int:user_id>/role")
def api_admin_set_role(user_id):
    if not _require_admin():
        return jsonify({"message": "Admin access required"}), 403
    m = _app
    role = ((request.get_json(silent=True) or {}).get("role") or "").strip().lower()
    if role not in _ROLE_RANK:
        return jsonify({"message": "Invalid role"}), 400
    user = m.User.query.get_or_404(user_id)
    if role != "admin" and _would_orphan_admin(user):
        return jsonify({"message": "Cannot demote the last active admin."}), 400
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
    if user.is_active and _would_orphan_admin(user):
        return jsonify({"ok": False, "message": "Cannot deactivate the last active admin."}), 400
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
    if _would_orphan_admin(user):
        return jsonify({"ok": False, "message": "Cannot delete the last active admin."}), 400
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
    err = _password_error(new_password, confirm_password)
    if err:
        return jsonify({"ok": False, "message": err}), 400
    user = m.db.session.get(m.User, m.current_user.id)
    user.password_hash = m.hash_password(new_password)
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
    if kind not in {"clause", "financial", "pq"}:
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
    # Admins/editors are alerted here when users submit feedback or flags, so they
    # don't have to poll the admin panel. Synthetic ids are namespaced well above
    # announcement ids so they never collide (the bell tracks seen ids as a set).
    if _require_role("admin"):
        admin_items = []
        try:
            fb = (m.db.session.query(m.FeedbackEntry, m.User.email)
                  .join(m.User, m.User.id == m.FeedbackEntry.user_id)
                  .filter(m.FeedbackEntry.status == "Open")
                  .order_by(m.FeedbackEntry.id.desc()).limit(10).all())
            for f, email in fb:
                snip = " ".join((f.message or "").split())
                if len(snip) > 90:
                    snip = snip[:90] + "…"
                admin_items.append({"id": 2_000_000_000 + f.id,
                                    "title": f"New feedback · {f.category}",
                                    "body": f"{email or 'A user'}: {snip}", "level": "info",
                                    "created_at": m._format_dt_local(f.created_at)})
            fl = (m.Flag.query.filter_by(status="Open")
                  .order_by(m.Flag.id.desc()).limit(10).all())
            for f in fl:
                snip = " ".join((f.description or f.reason or "").split())
                if len(snip) > 90:
                    snip = snip[:90] + "…"
                admin_items.append({"id": 3_000_000_000 + f.id,
                                    "title": f"New flag · {f.reason}",
                                    "body": f"{f.user_email or 'A user'} flagged {f.kind}: {snip}",
                                    "level": "warning",
                                    "created_at": m._format_dt_local(f.created_at)})
        except Exception as e:
            print(f"admin notif feed error: {e}")
        items = admin_items + items
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
