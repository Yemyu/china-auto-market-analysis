from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ENTRY_POINTS = (
    "04_explore_eda.py",
    "06_make_splits.py",
    "27_build_local_sentiment_features.py",
    "29_config_attribution.py",
    "33_evaluate_review_features.py",
    "34_analyze_forecast_robustness.py",
    "35_build_user_needs_and_alerts.py",
    "36_build_cold_start_curve.py",
    "39_evaluate_naive_forecast_baselines.py",
    "48_evaluate_rolling_origin.py",
)


def test_formal_entry_points_do_not_dynamically_import_numbered_scripts() -> None:
    offenders: list[str] = []
    for name in FORMAL_ENTRY_POINTS:
        path = PROJECT_ROOT / "scripts" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        if any(module == "importlib" or module.startswith("importlib.") for module in imported):
            offenders.append(name)
    assert offenders == []


def test_formal_entry_points_do_not_use_wildcard_imports() -> None:
    offenders: list[str] = []
    for name in FORMAL_ENTRY_POINTS:
        path = PROJECT_ROOT / "scripts" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
            for node in ast.walk(tree)
        ):
            offenders.append(name)
    assert offenders == []
