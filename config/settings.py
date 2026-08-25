"""Project settings.

All paths are resolved relative to this file.  Secrets (API keys) are read
from a `.env` file inside the `config/` directory.  `config/.env` is listed
in `.gitignore` and must never be committed.
"""
from __future__ import annotations

import os
from pathlib import Path

# Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
SENTIMENT_DIR: Path = DATA_DIR / "sentiment"
PROCESSED_DIR: Path = DATA_DIR / "processed"
STAGE3_DIR: Path = PROCESSED_DIR / "stage3"
STAGE4_DIR: Path = PROCESSED_DIR / "stage4"
FIGURES_DIR: Path = BASE_DIR / "figures"
NOTEBOOK_DIR: Path = BASE_DIR / "notebook"

# Ensure output directories exist when settings are imported.
STAGE4_DIR.mkdir(parents=True, exist_ok=True)

# Load .env file (no external dependency needed)
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
            # Do not overwrite real environment variables.
            if key and os.environ.get(key) is None:
                os.environ[key] = value


_load_env_file(_ENV_FILE)

# DeepSeek API
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_ABSA_MODEL: str = os.environ.get("DEEPSEEK_ABSA_MODEL", "deepseek-v4-flash")
DEEPSEEK_BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# LLM call behaviour
LLM_MAX_RETRIES: int = int(os.environ.get("LLM_MAX_RETRIES", "3"))
LLM_REQUEST_TIMEOUT: int = int(os.environ.get("LLM_REQUEST_TIMEOUT", "60"))
LLM_BATCH_SIZE: int = int(os.environ.get("LLM_BATCH_SIZE", "50"))
LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.1"))

# ABSA (Aspect-Based Sentiment Analysis)
ABSA_ASPECTS: list[str] = [
    "appearance",      # 外观
    "interior",        # 内饰
    "space",           # 空间
    "power",           # 动力
    "control",         # 操控
    "comfort",         # 舒适
    "fuel_consumption",  # 油耗
    "configuration",   # 配置
    "intelligence",    # 智能化
    "value",           # 性价比
]
ABSA_OUTPUT_DIR: Path = SENTIMENT_DIR / "absa"
ABSA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
