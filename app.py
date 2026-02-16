from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from typing import List
import re
import pandas as pd
import io
import os
import json
import time
import traceback
import sqlite3
import threading # Required for Background Sync
from datetime import datetime
import iris_brain as brain

app = Flask(__name__)

# ==========================================
# CRITICAL FIX: FORCE DATA LOAD ON STARTUP
# ==========================================
# This ensures that as soon as you run python app.py, 
# the system loads the data from SQL/Files into memory.
print("--- IRIS: Initializing Data Engine ---")
try:
    # 1. Load Knowledge Base (Text Search)
    brain.load_knowledge_base()
    # 2. Load Master Data Engine (Financial Data)
    brain.load_master_data_engine()
except Exception as e:
    print(f"[!] Warning: Data Engine load failed on startup: {e}")

CHAT_HISTORY = []
JUST_REDIRECTED = False
DB_NAME = "iris.db"

# ==========================================
# 0. SYSTEM ANALYTICS (MIDDLEWARE)
# ==========================================

def log_interaction(status_code, error_msg=None):
    """
    Records every request to the SQLite database (system_logs table).
    Filters out static assets and favicons to keep analytics clean.
    """
    # --- FILTER: Ignore static files AND favicon ---
    if request.path.startswith('/static') or request.path == '/favicon.ico': 
        return
    
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Ensure table exists
        c.execute('''CREATE TABLE IF NOT EXISTS system_logs 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, endpoint TEXT, method TEXT, ip TEXT, status INTEGER, error_msg TEXT)''')
        
        c.execute('''
            INSERT INTO system_logs (timestamp, endpoint, method, ip, status, error_msg)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            request.path,
            request.method,
            request.remote_addr,
            status_code,
            error_msg
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Logging Failed: {e}") 

@app.after_request
def record_success(response):
    """Logs successful requests (200, 302, 404, etc.)"""
    # We only log here if it's NOT a 500 (500s are handled by handle_crash)
    if response.status_code < 500:
        log_interaction(response.status_code)
    return response

@app.errorhandler(Exception)
def handle_crash(e):
    """
    Catches CRASHES (500 errors), logs them with the traceback, 
    and keeps IRIS alive instead of crashing the server.
    """
    # 1. Capture the full traceback to know EXACTLY where it failed
    error_trace = str(traceback.format_exc())
    print(f"🔥 IRIS CRASHED: {error_trace}") # Print to terminal for debugging
    
    # 2. Extract the specific error line for the UI log (last non-empty line)
    detailed_error = error_trace.strip().split('\n')[-1]
    
    log_interaction(500, error_msg=detailed_error) # Log detailed error
    return "<h3>IRIS System Error</h3><p>The system encountered an error. It has been logged for the admin.</p>", 500

# --- CLEAR LOGS ROUTE ---
@app.route("/clear_logs", methods=["POST"])
def clear_logs():
    """
    Wipes ONLY crashes and favicon noise from the database.
    Keeps legitimate user traffic stats intact.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Delete logs where status is 500+ OR endpoint is favicon
        c.execute("DELETE FROM system_logs WHERE status >= 500 OR endpoint = '/favicon.ico'")
        
        conn.commit()
        conn.close()
            
    except Exception as e:
        print(f"Error clearing logs: {e}")
        
    return redirect(url_for('analytics_dashboard'))

# ==========================================
# CONFIGURATION
# ==========================================
PDF_MAP = {
    "HEALTH MASTER CIRCULAR 2024": ["documents/health/health_master_circular_2024.pdf", "documents/health/HEALTH_MC_2024.pdf"],
    "HEALTH MC 2024": ["documents/health/HEALTH_MC_2024.pdf", "documents/health/health_master_circular_2024.pdf"],
    "PRODUCT REGULATIONS 2024": ["documents/health/product_regulations_2024.pdf", "documents/health/PRODUCT_REGS_2024.pdf"],
    "PRODUCT REGS 2024": ["documents/health/PRODUCT_REGS_2024.pdf", "documents/health/product_regulations_2024.pdf"],
    "PPHI REGULATIONS 2024": ["documents/health/PPHI_REGS_2024.pdf"],
    "PPHI REGS 2024": ["documents/health/PPHI_REGS_2024.pdf"],
    "PPHI MASTER CIRCULAR 2024": ["documents/health/PPHI_MC_2024.pdf"],
    "PPHI MC 2024": ["documents/health/PPHI_MC_2024.pdf"],
    "INSURANCE ACT 1938": ["documents/health/INSURANCE_ACT_1938.pdf"],
    "IRDAI ACT 1999": ["documents/health/IRDAI_ACT_1999.pdf"]
}


def resolve_pdf_path(doc_name_key):
    candidates = PDF_MAP.get(doc_name_key, [])
    for rel_path in candidates:
        abs_path = os.path.join(app.static_folder, rel_path)
        if os.path.exists(abs_path):
            return rel_path
    return None

TYPE_STYLES = {
    "ACT": {"color": "#856404", "bg": "#fff3cd", "border": "#ffeeba", "label": "ACT (The Law)"},
    "REGULATION": {"color": "#004085", "bg": "#cce5ff", "border": "#b8daff", "label": "REGULATION"},
    "MASTER": {"color": "#155724", "bg": "#d4edda", "border": "#c3e6cb", "label": "MASTER CIRCULAR"},
    "CIRCULAR": {"color": "#0c5460", "bg": "#d1ecf1", "border": "#bee5eb", "label": "CIRCULAR"},
    "GUIDELINE": {"color": "#383d41", "bg": "#e2e3e5", "border": "#d6d8db", "label": "GUIDELINE"},
    "UNKNOWN": {"color": "#666", "bg": "#f2f2f2", "border": "#ddd", "label": "DOCUMENT"}
}

# ==========================================
# ASYNC SYNC ENGINE (BACKGROUND THREADS)
# ==========================================
# Global state to track the background job.
# The Admin UI polls this variable.
SYNC_STATE = {
    "status": "idle",       # idle, running, complete, error
    "message": "System ready.",
    "timestamp": None
}

def run_background_sync():
    """Executes the heavy data aggregation logic in a separate thread."""
    global SYNC_STATE
    try:
        print("--- BACKGROUND SYNC STARTED ---")
        SYNC_STATE["status"] = "running"
        SYNC_STATE["message"] = "Syncing financial + regulatory data from knowledge_base..."

        financial_msg = brain.aggregate_submissions()
        regulatory_msg = brain.aggregate_regulatory_documents()
        result_msg = f"{financial_msg} | {regulatory_msg}"

        SYNC_STATE["status"] = "complete"
        SYNC_STATE["message"] = result_msg
        SYNC_STATE["timestamp"] = time.strftime("%H:%M:%S")
        print("--- BACKGROUND SYNC FINISHED ---")
        
    except Exception as e:
        print(f"--- SYNC ERROR: {e} ---")
        SYNC_STATE["status"] = "error"
        SYNC_STATE["message"] = f"Error: {str(e)}"

# ==========================================
# ADMIN ROUTES (SYNC CONTROL)
# ==========================================

@app.route("/admin", methods=["GET"])
def admin_panel():
    """Renders the Admin UI, passing initial sync state."""
    return render_template("admin.html", sync_state=SYNC_STATE)

@app.route("/admin/sync_start", methods=["POST"])
def sync_start():
    """Kicks off the background sync thread."""
    global SYNC_STATE
    
    if SYNC_STATE["status"] == "running":
        return jsonify({"status": "error", "message": "Sync already in progress."})

    # Reset State & Start
    SYNC_STATE["status"] = "starting"
    SYNC_STATE["message"] = "Initializing background process..."
    
    thread = threading.Thread(target=run_background_sync)
    thread.daemon = True # Ensures thread dies if app restarts
    thread.start()
    
    return jsonify({"status": "started"})

@app.route("/admin/sync_status", methods=["GET"])
def sync_status():
    """Frontend polls this to update progress bars."""
    return jsonify(SYNC_STATE)

# ==========================================
# ROUTES (MODULES)
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():
    return handle_search("universal")

@app.route("/health", methods=["GET", "POST"])
def health_module():
    return handle_search("health")

@app.route("/life", methods=["GET", "POST"])
def life_module():
    return handle_search("life")

# --- DATA MODULE (UPDATED) ---
@app.route("/data", methods=["GET", "POST"])
def data_module():
    filter_options = brain.get_filter_options()
    
    if request.method == "POST":
        # Capture all filters from the form
        filters = {
            "dimension": request.form.get("dimension", "Insurer"),
            "entities": request.form.getlist("entities"),
            "metrics": request.form.getlist("metrics"),
            "years": request.form.getlist("years"),
            "quarters": request.form.getlist("quarters"),
            "lobs": request.form.getlist("lobs"),
            "classes": request.form.getlist("classes")
        }
        
        # Process filters through the brain
        report_data = brain.filter_data(filters)
        
        # Return only the table fragment (HTMX-style update)
        return render_template("components/data_table.html", report=report_data)
        
    # GET Request: Render the full dashboard
    return render_template("data_dashboard.html", options=filter_options, active_module="data")

@app.route("/download_data", methods=["POST"])
def download_data():
    """Generates and downloads the Excel report based on active filters."""
    filters = {
        "dimension": request.form.get("dimension", "Insurer"),
        "entities": request.form.getlist("entities"),
        "metrics": request.form.getlist("metrics"),
        "years": request.form.getlist("years"),
        "quarters": request.form.getlist("quarters"),
        "lobs": request.form.getlist("lobs"),
        "classes": request.form.getlist("classes")
    }
    
    excel_file = brain.generate_excel(filters)
    
    if excel_file:
        return send_file(
            excel_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='IRIS_Financial_Report.xlsx'
        )
    return "No data found for these filters.", 400

# --- COMPLIANCE ROUTE (UPDATED FOR YEAR FILTER) ---
@app.route("/compliance", methods=["GET"])
def compliance_dashboard():
    # 1. Get available years for the dropdown
    available_years = brain.get_compliance_years()
    
    # 2. Get selected year from URL (e.g. ?year=2023-24)
    selected_year = request.args.get("year")
    
    # 3. Get Dashboard Data based on selection
    compliance_data = brain.get_compliance_dashboard(target_year=selected_year)
    
    return render_template("compliance.html", 
                           companies=compliance_data, 
                           years=available_years,
                           active_year=selected_year if selected_year else "Latest",
                           active_module="compliance")

# --- ANALYTICS ROUTE (SQL INTEGRATED) ---
@app.route("/analytics")
def analytics_dashboard():
    logs = []
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row # This enables column access by name
        c = conn.cursor()
        c.execute("SELECT * FROM system_logs ORDER BY id DESC")
        rows = c.fetchall()
        
        # Convert sqlite3.Row objects to dicts for easy processing
        logs = [dict(row) for row in rows]
        conn.close()
    except Exception as e:
        print(f"DB Error in Analytics: {e}")
        logs = []

    # --- 1. PRE-PROCESS LOGS ---
    # Filter out favicon.ico
    valid_logs = [l for l in logs if l.get('endpoint') != '/favicon.ico']

    # --- 2. CALCULATE BASIC STATS ---
    total_requests = len(valid_logs)
    unique_users = len(set(l['ip'] for l in valid_logs))
    errors = [l for l in valid_logs if l['status'] >= 500]
    error_count = len(errors)
    
    # --- 3. CALCULATE ENDPOINT USAGE (PIE CHART) ---
    endpoints = {}
    
    # Define internal endpoints to hide from the pie chart
    ignored_routes = ['/clear_logs', '/sync_data', '/download_data', '/admin/sync_status', '/admin/sync_start']
    
    for l in valid_logs:
        ep = l['endpoint']
        
        # Skip hidden routes
        if ep in ignored_routes: 
            continue
        
        # Clean Labels (Remove slash & Format)
        if ep == "/": label = "Home"
        elif ep == "/data": label = "Data Explorer"
        elif ep == "/compliance": label = "Compliance Cockpit"
        elif ep == "/analytics": label = "Analytics"
        elif ep == "/health": label = "Health Dept"
        elif ep == "/life": label = "Life Dept"
        elif ep == "/admin": label = "Admin Panel"
        else:
            # Fallback: remove slash and title case (e.g., "/some_page" -> "Some Page")
            label = ep.lstrip("/").replace("_", " ").title()
        
        endpoints[label] = endpoints.get(label, 0) + 1
        
    chart_labels = list(endpoints.keys())
    chart_data = list(endpoints.values())

    # --- 4. CALCULATE MONTHLY & YEARLY TRENDS ---
    current_year = datetime.now().year
    
    # Initialize the buckets for Jan-Dec
    monthly_labels = [datetime(current_year, m, 1).strftime('%b %Y') for m in range(1, 13)]
    monthly_data = [0] * 12
    yearly_stats = []

    try:
        if valid_logs:
            df = pd.DataFrame(valid_logs)
            df['dt'] = pd.to_datetime(df['timestamp'])
            
            # -- MONTHLY LOGIC --
            # Filter for CURRENT YEAR ONLY
            df_this_year = df[df['dt'].dt.year == current_year]
            
            # Group by Month Number (1-12)
            counts = df_this_year['dt'].dt.month.value_counts()
            
            # Populate array (index 0 is Jan)
            for month_num, count in counts.items():
                if 1 <= month_num <= 12:
                    monthly_data[month_num - 1] = int(count)

            # -- YEARLY LOGIC --
            yearly_counts = df['dt'].dt.year.value_counts().sort_index()
            yearly_stats = [{"year": y, "count": c} for y, c in yearly_counts.items()]

    except Exception as e:
        print(f"Analytics Data Processing Error: {e}")

    # Render template with all processed data
    return render_template("analytics.html", 
                           stats={"total": total_requests, "users": unique_users, "errors": error_count},
                           logs=errors,
                           chart={"labels": chart_labels, "data": chart_data},
                           monthly={"labels": monthly_labels, "data": monthly_data},
                           yearly=yearly_stats,
                           active_module="analytics")

# ==========================================
# CORE SEARCH LOGIC (TEXT ONLY)
# ==========================================

def handle_search(active_module):
    """
    Central handler for Universal, Health, and Life search logic.
    Manages Keywords, Deep Scans, and Rendering Results.
    """
    global CHAT_HISTORY, JUST_REDIRECTED

    # Brain now handles SQL retrieval internally
    KB_DF = brain.load_knowledge_base()
    vocab = brain.get_autocomplete_data()

    # GET Request: Just show the search page (with history if applicable)
    if request.method == "GET":
        if not JUST_REDIRECTED: CHAT_HISTORY = []
        JUST_REDIRECTED = False
        return render_template("index.html", history=CHAT_HISTORY, vocab=vocab, active_module=active_module)

    # POST Request: Process the query
    query = request.form.get("query", "").strip()
    
    if not query: 
        if active_module == "universal": return redirect(url_for("index"))
        return redirect(url_for(f"{active_module}_module"))

    # --- 1. DEEP SCAN HANDLER (Triggered by Chips) ---
    if query.startswith("__DEEP_SCAN__:"):
        raw_payload = query.replace("__DEEP_SCAN__:", "")
        pairs = raw_payload.split("||")
        # Reconstruct the tuple list: [('hospital', 'hospit'), ('cashless', 'cashless')]
        keyword_tuples = [(p.split("|")[0], p.split("|")[1]) for p in pairs if len(p.split("|"))==2]
        
        display_kws = [t[0] for t in keyword_tuples]
        
        # First, check tags again to get context IDs
        tag_matches = brain.search_tags_only(keyword_tuples, KB_DF, module=active_module)
        exclude_ids = [m['id'] for m in tag_matches]
        
        # Then, perform the expensive full-text scan
        matches = brain.deep_scan_brain(keyword_tuples, KB_DF, exclude_ids=exclude_ids, module=active_module)
        
        response = ""
        if matches:
            response = f"<div class='analysis-text'>Deep Scan results in <strong>{active_module.upper()}</strong>: <strong>{', '.join(display_kws)}</strong></div>"
            response += build_results_html(matches, display_kws)
        else:
            response = f"No additional matches found in <strong>{active_module.capitalize()}</strong> module."
            
        CHAT_HISTORY.append({"query": "Deep Scan", "response": response})
        JUST_REDIRECTED = True
        
        if active_module == "universal": return redirect(url_for("index"))
        return redirect(url_for(f"{active_module}_module"))

    # --- 2. GREETING CHECK ---
    if brain.check_greeting(query):
        response = "<strong>Hello!</strong> I am <strong>IRIS</strong>. Ask me anything related to IRDAI Acts, Regulations, Circulars, or Guidelines."
    
    else:
        # --- 3. STANDARD SEARCH ---
        # Step A: NLP Processing (Cleaning & Stemming)
        kw_tuples = brain.get_clean_keywords(query)
        display_kws = [t[0] for t in kw_tuples]
        
        if not kw_tuples:
            response = "Query rejected. Please use regulatory terms."
        else:
            # Step B: Tag-Based Search (Fast & Precise)
            tag_matches = brain.search_tags_only(kw_tuples, KB_DF, module=active_module)
            highlight_kws = [raw for (raw, clean) in kw_tuples if clean in brain.ALL_UNIQUE_TAGS]

            if tag_matches:
                response = f"<div style='font-size:12px; color:#888; margin-bottom:10px;'>Found via <strong>Tags</strong>: {', '.join(display_kws)}</div>"
                response += build_results_html(tag_matches, highlight_kws)
            else:
                if active_module == "life":
                    response = "<div><strong>Life Department:</strong> No documents currently loaded.</div>"
                elif active_module == "data":
                    response = "<div><strong>Data Module:</strong> Text search is disabled here.</div>"
                else:
                    response = f"<div>No matches found in {active_module.capitalize()} for: <strong>{', '.join(display_kws)}</strong></div>"

            # Step C: Offer Deep Scan Chips
            response += build_chips_html(kw_tuples, query)

    CHAT_HISTORY.append({"query": query, "response": response})
    JUST_REDIRECTED = True
    
    if active_module == "universal": return redirect(url_for("index"))
    return redirect(url_for(f"{active_module}_module"))

# =========================================================
# UTILS (Rendering Helpers)
# =========================================================

def build_chips_html(kw_tuples, original_query):
    """
    Generates the 'Deep Scan' buttons.
    FIX: Removes redundant 'Phrase Search' if it matches a single keyword.
    """
    if not kw_tuples: return ""

    html = """<hr><div style="font-size:12px; color:#666; margin-bottom:8px;">
            Not finding what you need? <strong>Deep Scan specific terms:</strong></div>
            <div style="display: flex; flex-wrap: wrap; gap: 6px;">"""
    
    # Track what we have shown to avoid duplicates
    shown_labels = set()

    # 1. Chips for individual keywords
    for raw, clean in kw_tuples:
        label = f'Search "{raw}"'
        if label in shown_labels: continue
        
        payload = f"{raw}|{clean}"
        html += f"""<form method="POST" style="margin:0;"><input type="hidden" name="query" value="__DEEP_SCAN__:{payload}">
                <button type="submit" style="background:#e8eaf6; border:1px solid #3f51b5; color:#1a237e; padding:6px 12px; border-radius:16px; font-size:11px; cursor:pointer;">
                {label}</button></form>"""
        shown_labels.add(label)
    
    # 2. Chip for the Exact Phrase (ONLY if multi-word AND not already shown)
    clean_original = " ".join(original_query.split()).strip()
    phrase_label = f'Search Phrase "{clean_original}"'
    
    if len(clean_original.split()) > 1 and phrase_label not in shown_labels:
        html += f"""<form method="POST" style="margin:0;"><input type="hidden" name="query" value="__DEEP_SCAN__:{clean_original}|{clean_original}">
                <button type="submit" style="background:#e3f2fd; border:1px solid #2196f3; color:#0d47a1; padding:6px 12px; border-radius:16px; font-weight:700; font-size:11px; cursor:pointer;">
                {phrase_label}</button></form>"""
    
    # 3. Chip for 'Search All' (combined) - Only if we have multiple distinct keywords
    if len(kw_tuples) > 1:
        all_payload = "||".join([f"{t[0]}|{t[1]}" for t in kw_tuples])
        html += f"""<form method="POST" style="margin:0;"><input type="hidden" name="query" value="__DEEP_SCAN__:{all_payload}">
                <button type="submit" style="background:#fff; border:1px solid #999; color:#666; padding:6px 12px; border-radius:16px; font-size:11px; cursor:pointer;">
                Search All</button></form>"""
                
    html += "</div>"
    return html

def highlight_keywords(text, keywords):
    """Wraps found keywords in a highlighting span."""
    if not keywords: return text
    expanded = set(keywords)
    # Simple stemming for highlight matching (e.g., 'insurer' -> 'insurers')
    for k in keywords:
        k = k.lower()
        expanded.add(k + "s"); expanded.add(k + "ed"); expanded.add(k + "ing")
    
    for kw in sorted(list(expanded), key=len, reverse=True):
        if len(kw) < 3: continue
        pattern = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
        text = pattern.sub(r"<span class='iris-highlight'>\1</span>", text)
    return text

def convert_markdown_to_html(text):
    """Simple parser to handle basic Markdown tables and line breaks."""
    lines = text.splitlines()
    new_lines: List[str] = []
    in_table = False
    table_rows: List[str] = []
    
    def flush_table(rows):
        if not rows: return ""
        html = '<table border="1" style="border-collapse: collapse; width: 100%; border-color: #ddd; font-size: 11px; margin-top:5px; margin-bottom:5px;">'
        for i, row in enumerate(rows):
            # Skip separator lines like |---|---|
            if re.match(r"^\|[\s\-:\|]+\|$", row): continue
            tag = "th" if i == 0 else "td"
            bg = 'style="background-color: #f2f2f2; padding: 5px;"' if i == 0 else 'style="padding: 5px;"'
            cells = [c.strip() for c in row.strip("|").split("|")]
            html += "<tr>"
            for c in cells: html += f"<{tag} {bg}>{c}</{tag}>"
            html += "</tr>"
        html += "</table>"
        return html

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True; table_rows.append(stripped)
        else:
            if in_table: 
                new_lines.append(flush_table(table_rows))
                table_rows = []; in_table = False
            new_lines.append(line)
            
    if in_table: new_lines.append(flush_table(table_rows))
    return "\n".join(new_lines)

def format_verbatim(raw_text, keywords):
    """Formats raw text for display while preserving heading style without highlights."""
    if not raw_text: return ""
    text_with_tables = convert_markdown_to_html(raw_text)

    lines = text_with_tables.splitlines()
    formatted_lines: List[str] = []

    for line in lines:
        stripped = line.strip()

        # Keep table/html lines untouched
        if stripped.startswith("<table") or stripped.startswith("<tr") or stripped.startswith("<th") or stripped.startswith("<td") or stripped.startswith("</"):
            formatted_lines.append(line)
            continue

        # Headings keep original formatting (bold only), no keyword highlighting
        if stripped.endswith(":"):
            formatted_lines.append(f"<strong>{line}</strong>")
            continue

        # Regular lines get keyword highlighting
        formatted_lines.append(highlight_keywords(line, keywords))

    final_text = "\n".join(formatted_lines)
    return f'<div style="white-space: pre-wrap; font-family: inherit;">{final_text}</div>'

def build_results_html(matches, keywords):
    """
    Constructs the HTML card for each search result.
    FIX: Makes PDF lookup case-insensitive.
    """
    html = ""
    current_doc_type = None
    
    for i, m in enumerate(matches):
        doc_type = m['type']
        # Header for new document types
        if doc_type != current_doc_type:
            style = TYPE_STYLES.get(doc_type, TYPE_STYLES["UNKNOWN"])
            html += f"""<div style="margin-top:15px;margin-bottom:10px;background-color:{style['bg']};border-left:5px solid {style['color']};padding:8px 12px;font-family:'Segoe UI';color:{style['color']};font-weight:bold;font-size:13px;text-transform:uppercase;border-radius:4px;">{style['label']}</div>"""
            current_doc_type = doc_type
        else: 
            html += "<hr style='border: 0; border-top: 1px dashed #999; margin: 15px 0;'>"
            
        formatted_body = format_verbatim(m['raw_text'], keywords)
        
        # --- FIX: Case-Insensitive PDF Lookup ---
        doc_name_key = m['source'].strip().upper()
        pdf_path = resolve_pdf_path(doc_name_key)

        pdf_btn = ""
        if pdf_path:
            pdf_btn = f"""<a href='/static/{pdf_path}' target='_blank' 
                        style='float:right; margin-left:8px; font-size:10px; font-weight:bold; text-decoration:none; color:#d32f2f; background:#fff; padding:2px 6px; border:1px solid #d32f2f; border-radius:3px;'>
                        <i class="fas fa-file-pdf"></i> PDF</a>"""
        
        # Copy Button Logic
        content_id = f"clause_text_{i}"
        copy_btn = f"""<button onclick="copyToClipboard('{content_id}')" title="Copy Clause" style="float:right; margin-right: 8px; background:none; border:none; color:#666; cursor:pointer; font-size:14px;"><i class="far fa-copy"></i></button>"""
        
        html += f"""<div style="margin-bottom:6px; font-size:11px; color:#555;"><span style="font-weight:800; color:#333;">{m['source']}</span> | <span style="color:#0056b3;">{m['header']}</span> | Clause: {m['id']} {pdf_btn} {copy_btn}</div><div id="{content_id}" style="line-height:1.5; color:#222; font-size:14px;">{formatted_body}</div>"""
    return html

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8080)