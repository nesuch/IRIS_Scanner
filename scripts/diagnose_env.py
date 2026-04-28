#!/usr/bin/env python3
"""Local environment drift diagnostic for IRIS Scanner."""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REQUIRED_PACKAGES = [
    "flask",
    "flask-login",
    "flask-sqlalchemy",
    "alembic",
    "pandas",
    "openpyxl",
    "nltk",
    "gunicorn",
    "psycopg2-binary",
]

KEY_ENV_VARS = [
    "DATABASE_URL",
    "IRIS_AUTH_DATABASE_URL",
    "IRIS_DB_PATH",
    "IRIS_SESSION_SECRET",
    "ADMIN_EMAIL",
    "ADMIN_PASSWORD",
    "K_SERVICE",
    "GOOGLE_CLOUD_PROJECT",
    "GAE_ENV",
]


def _masked(value: str) -> str:
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}...{value[-3:]}"


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for pkg in REQUIRED_PACKAGES:
        key = pkg.lower().replace("_", "-")
        try:
            result[pkg] = metadata.version(key)
        except metadata.PackageNotFoundError:
            result[pkg] = "MISSING"
    return result


def _env_summary() -> dict[str, str]:
    data: dict[str, str] = {}
    for key in KEY_ENV_VARS:
        value = (os.getenv(key) or "").strip()
        if not value:
            data[key] = "<unset>"
        elif "PASSWORD" in key or "SECRET" in key or "URL" in key:
            data[key] = _masked(value)
        else:
            data[key] = value
    return data


def _sqlite_status(db_path: Path) -> dict[str, str]:
    info: dict[str, str] = {
        "db_path": str(db_path),
        "exists": str(db_path.exists()),
        "readable": str(os.access(db_path, os.R_OK)),
        "writable": str(os.access(db_path, os.W_OK)),
    }
    if not db_path.exists():
        info["connection"] = "skipped (file missing)"
        return info

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1")
        table_names = [name for (name,) in cur.fetchall()]
        info["connection"] = "ok"
        info["table_count"] = str(len(table_names))
        info["tables"] = ", ".join(table_names[:12]) + (" ..." if len(table_names) > 12 else "")

        if "alembic_version" in table_names:
            cur.execute("SELECT version_num FROM alembic_version LIMIT 1")
            row = cur.fetchone()
            info["alembic_version"] = row[0] if row else "<empty>"
        else:
            info["alembic_version"] = "<missing>"

        conn.close()
    except Exception as exc:  # noqa: BLE001
        info["connection"] = f"failed: {exc}"
    return info


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    default_db = root / "iris.db"
    db_path = Path(os.getenv("IRIS_DB_PATH", str(default_db))).resolve()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "venv": os.getenv("VIRTUAL_ENV", "<not in venv>"),
        },
        "env": _env_summary(),
        "packages": _package_versions(),
        "sqlite": _sqlite_status(db_path),
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
