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
import io
import re
import time
from datetime import datetime

import pandas as pd
from flask import Blueprint, request, jsonify, session, send_file

import iris_brain as brain

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
        "device_count": _app._get_active_device_count(cu.id),
        "display_name": getattr(cu, "display_name", None),
        "avatar": getattr(cu, "avatar", None),
    }


def _require_admin():
    return _app.current_user.is_authenticated and getattr(_app.current_user, "is_admin", False)


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
def _match_payload(m):
    pdf_path = _app.resolve_pdf_path(str(m.get("source", "")).strip().upper())
    return {
        "source": m.get("source", "UNKNOWN"),
        "type": m.get("type", "UNKNOWN"),
        "id": str(m.get("id", "")).strip(),
        "header": m.get("header", ""),
        "raw_text": str(m.get("raw_text", "")),
        "pdf_url": ("/static/" + pdf_path) if pdf_path else None,
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
                  "is_admin": u.is_admin, "created_at": m._format_dt_local(u.created_at),
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
    is_admin = str(data.get("is_admin") or "").lower() in {"1", "true", "yes", "on"}
    if not m._is_allowed_email(email):
        return jsonify({"message": "Email must use @irdai.gov.in domain."}), 400
    if len(password) < 8:
        return jsonify({"message": "Password must be at least 8 characters."}), 400
    if m.User.query.filter(m.db.func.lower(m.User.email) == email).first():
        return jsonify({"message": "User already exists."}), 409
    m.db.session.add(m.User(email=email, password_hash=m.generate_password_hash(password),
                            is_active=True, is_admin=is_admin))
    m.db.session.commit()
    m._record_admin_audit(email, "user_create", "success")
    return jsonify({"ok": True, "message": "User created successfully."}), 201


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
    m.db.session.add(m.Flag(user_email=m.current_user.email, kind=kind, reason=reason,
                            description=description, target=target, detail=detail,
                            status="Open", created_at=datetime.utcnow()))
    m.db.session.commit()
    return jsonify({"ok": True, "message": "Thanks — flag submitted for review."}), 201


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
