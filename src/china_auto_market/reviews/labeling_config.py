"""Configuration for the optional review-labeling API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from china_auto_market.paths import PROJECT_ROOT


DEFAULT_ENV_FILE = PROJECT_ROOT / "config" / ".env"


@dataclass(frozen=True)
class ReviewLabelingConfig:
    api_key: str
    model: str
    base_url: str
    timeout: int


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def load_review_labeling_config(
    env_file: Path = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> ReviewLabelingConfig:
    """Read optional API settings, with environment variables taking precedence."""
    file_values = _read_env_file(env_file)
    environment = os.environ if environ is None else environ

    def value(name: str, default: str = "") -> str:
        return environment.get(name, file_values.get(name, default))

    return ReviewLabelingConfig(
        api_key=value("REVIEW_LABEL_API_KEY"),
        model=value("REVIEW_LABEL_MODEL"),
        base_url=value("REVIEW_LABEL_BASE_URL"),
        timeout=int(value("REVIEW_LABEL_TIMEOUT", "60")),
    )
