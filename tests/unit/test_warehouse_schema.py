from __future__ import annotations

from pathlib import Path

import pytest

from china_auto_market.warehouse import schema


def test_mysql_command_uses_login_path_without_plaintext_password(tmp_path: Path) -> None:
    mysql = tmp_path / "mysql"
    mysql.write_text("", encoding="utf-8")

    command = schema.mysql_command("local-auto", str(mysql))

    assert command == [str(mysql), "--login-path=local-auto", "--batch"]
    assert not any("password" in part.lower() for part in command)


def test_apply_ddl_fails_closed_on_existing_target_database(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(schema, "existing_target_databases", lambda *args, **kwargs: {"auto_raw"})

    with pytest.raises(RuntimeError, match="auto_raw"):
        schema.apply_ddl("local-auto", ddl_dir=tmp_path)


def test_ordered_ddl_contains_all_four_layers() -> None:
    files = sorted(path.name for path in schema.DDL_DIR.glob("*.sql"))

    assert files == [
        "00_create_databases.sql",
        "10_auto_ops.sql",
        "20_auto_raw.sql",
        "30_auto_staging.sql",
        "40_auto_mart.sql",
        "45_forecast_precision.sql",
        "46_annual_sales_corrections.sql",
        "47_user_needs_precision.sql",
    ]
