"""Safe MySQL CLI helpers that reuse an encrypted login path."""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from china_auto_market.warehouse.schema import mysql_command


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_value(value: Any) -> Any:
    """Convert pandas/numpy scalars into JSON- and SQL-safe Python values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat(sep=" ") if isinstance(value, (pd.Timestamp, datetime)) else value.isoformat()
    return value


def json_payload(record: dict[str, Any], *, exclude: Iterable[str] = ()) -> str:
    """Serialize one source record without non-standard NaN values."""
    excluded = set(exclude)
    normalized = {key: normalize_value(value) for key, value in record.items() if key not in excluded}
    return json.dumps(normalized, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def sql_literal(value: Any) -> str:
    """Encode a value as a MySQL literal without relying on shell escaping."""
    value = normalize_value(value)
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, Decimal)):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = str(value).encode("utf-8").hex()
    return f"CONVERT(0x{encoded} USING utf8mb4)"


def quote_identifier(value: str) -> str:
    """Quote an identifier after a strict allow-list check."""
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe SQL identifier: {value}")
    return f"`{value}`"


def insert_statements(
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    chunk_size: int,
) -> Iterator[str]:
    """Yield bounded multi-row INSERT statements for a trusted table contract."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if "." not in table:
        raise ValueError("table must include its database name")
    database, table_name = table.split(".", 1)
    target = f"{quote_identifier(database)}.{quote_identifier(table_name)}"
    column_sql = ",".join(quote_identifier(column) for column in columns)
    chunk: list[str] = []
    for row in rows:
        if len(row) != len(columns):
            raise ValueError(f"Expected {len(columns)} values, received {len(row)}")
        chunk.append("(" + ",".join(sql_literal(value) for value in row) + ")")
        if len(chunk) == chunk_size:
            yield f"INSERT INTO {target} ({column_sql}) VALUES\n" + ",\n".join(chunk) + ";\n"
            chunk = []
    if chunk:
        yield f"INSERT INTO {target} ({column_sql}) VALUES\n" + ",\n".join(chunk) + ";\n"


def run_transaction(
    login_path: str,
    statements: Iterable[str],
    *,
    mysql_binary: str | None = None,
) -> None:
    """Stream one transaction to MySQL and fail closed on any SQL error."""
    process = subprocess.Popen(
        mysql_command(login_path, mysql_binary),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdin is None:
        raise RuntimeError("MySQL client stdin is unavailable")
    try:
        process.stdin.write("SET NAMES utf8mb4;\nSTART TRANSACTION;\n")
        for statement in statements:
            process.stdin.write(statement)
        process.stdin.write("COMMIT;\n")
        stdout, stderr = process.communicate()
    except BrokenPipeError as error:
        stdout, stderr = process.communicate()
        raise RuntimeError(stderr.strip() or stdout.strip() or "MySQL transaction failed") from error
    if process.returncode:
        raise RuntimeError(stderr.strip() or stdout.strip() or "MySQL transaction failed")


def relative_source_uri(path: Path, root: Path) -> str:
    """Return a stable project-relative source URI for batch metadata."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return f"file://external/{path.name}"
    return f"file://{relative.as_posix()}"
