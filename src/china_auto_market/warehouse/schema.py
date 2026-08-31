"""Safe MySQL schema inspection and repeatable DDL application."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from china_auto_market.paths import PROJECT_ROOT


TARGET_DATABASES = ("auto_ops", "auto_raw", "auto_staging", "auto_mart")
DDL_DIR = PROJECT_ROOT / "sql" / "ddl"


def resolve_mysql_binary(explicit: str | None = None) -> str:
    """Resolve the MySQL client without installing anything system-wide."""
    candidates = [explicit, os.environ.get("MYSQL_CLIENT"), shutil.which("mysql")]
    candidates.append("/usr/local/mysql/bin/mysql")
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError("MySQL client was not found; set MYSQL_CLIENT to its path")


def mysql_command(login_path: str, mysql_binary: str | None = None) -> list[str]:
    """Build a client command that uses an encrypted MySQL login path."""
    if not login_path or any(character.isspace() for character in login_path):
        raise ValueError("MySQL login path must be a non-empty name without whitespace")
    return [resolve_mysql_binary(mysql_binary), f"--login-path={login_path}", "--batch"]


def query(login_path: str, sql: str, mysql_binary: str | None = None) -> str:
    """Run a read-only metadata query and return its tab-separated output."""
    try:
        result = subprocess.run(
            [*mysql_command(login_path, mysql_binary), "--skip-column-names", "--execute", sql],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "unknown MySQL error"
        raise RuntimeError(f"MySQL command failed: {detail}") from error
    return result.stdout


def query_json_rows(
    login_path: str,
    sql: str,
    mysql_binary: str | None = None,
) -> list[dict[str, object]]:
    """Return one JSON object per result row without exposing a plaintext password.

    Callers must make the SELECT return exactly one JSON document per row. The
    MySQL client's raw batch mode preserves JSON escaping, including review
    newlines and tabs, so each physical output line remains independently
    parseable.
    """
    try:
        result = subprocess.run(
            [
                *mysql_command(login_path, mysql_binary),
                "--raw",
                "--skip-column-names",
                "--execute",
                sql,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "unknown MySQL error"
        raise RuntimeError(f"MySQL command failed: {detail}") from error
    rows: list[dict[str, object]] = []
    # Split only on the client's physical LF delimiter. ``str.splitlines``
    # would also split on valid Unicode separators that may occur in reviews.
    for line_number, line in enumerate(result.stdout.split("\n"), start=1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object on MySQL output line {line_number}")
        rows.append(value)
    return rows


def list_databases(login_path: str, mysql_binary: str | None = None) -> set[str]:
    """Return visible database names without reading table contents."""
    return {line.strip() for line in query(login_path, "SHOW DATABASES;", mysql_binary).splitlines() if line.strip()}


def existing_target_databases(login_path: str, mysql_binary: str | None = None) -> set[str]:
    """Return target names that already exist on the server."""
    return set(TARGET_DATABASES) & list_databases(login_path, mysql_binary)


def apply_ddl(
    login_path: str,
    *,
    allow_existing: bool = False,
    mysql_binary: str | None = None,
    ddl_dir: Path = DDL_DIR,
) -> list[Path]:
    """Apply ordered CREATE IF NOT EXISTS files after an explicit collision check."""
    existing = existing_target_databases(login_path, mysql_binary)
    if existing and not allow_existing:
        names = ", ".join(sorted(existing))
        raise RuntimeError(
            f"Target databases already exist: {names}. Re-run with allow_existing=True "
            "only after confirming they belong to this project."
        )
    files = sorted(ddl_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No DDL files found in {ddl_dir}")
    command = mysql_command(login_path, mysql_binary)
    for path in files:
        try:
            subprocess.run(
                command,
                input=path.read_text(encoding="utf-8"),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or error.stdout.strip() or "unknown MySQL error"
            raise RuntimeError(f"DDL failed in {path.name}: {detail}") from error
    return files
