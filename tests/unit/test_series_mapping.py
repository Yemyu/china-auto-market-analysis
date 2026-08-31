from __future__ import annotations

from china_auto_market.quality.series_mapping import build_series_name_mapping, normalize_series_name


def test_normalization_keeps_meaningful_plus_symbol() -> None:
    assert normalize_series_name("A + Pro") == "a+pro"
    assert normalize_series_name("A Pro") == "apro"


def test_mapping_accepts_exact_and_unambiguous_typography_only() -> None:
    mapping = build_series_name_mapping(
        ["Model A", "Model-B", "Twin X", "Twin-X"],
        ["Model A", "Model B", "Twin_X"],
    )

    pairs = set(zip(mapping["sales_series_name"], mapping["config_series_name"], strict=True))
    assert ("Model A", "Model A") in pairs
    assert ("Model-B", "Model B") in pairs
    assert not any(sales.startswith("Twin") for sales, _ in pairs)
