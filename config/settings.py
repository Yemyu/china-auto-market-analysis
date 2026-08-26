"""Runtime settings for optional API labeling."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent.parent
_ENV_FILE: Path = BASE_DIR / "config" / ".env"


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and os.environ.get(key) is None:
                os.environ[key] = value


_load_env_file(_ENV_FILE)

REVIEW_LABEL_API_KEY: str = os.environ.get("REVIEW_LABEL_API_KEY", "")
REVIEW_LABEL_MODEL: str = os.environ.get("REVIEW_LABEL_MODEL", "")
REVIEW_LABEL_BASE_URL: str = os.environ.get("REVIEW_LABEL_BASE_URL", "")
REVIEW_LABEL_TIMEOUT: int = int(os.environ.get("REVIEW_LABEL_TIMEOUT", "60"))
