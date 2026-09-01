from __future__ import annotations

from china_auto_market.reviews.labeling_config import load_review_labeling_config


def test_labeling_config_reads_file_and_prefers_environment(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "REVIEW_LABEL_API_KEY=file-key\n"
        "REVIEW_LABEL_MODEL=file-model\n"
        "REVIEW_LABEL_BASE_URL=https://file.example/v1\n"
        "REVIEW_LABEL_TIMEOUT=45\n",
        encoding="utf-8",
    )

    config = load_review_labeling_config(
        env_file,
        environ={"REVIEW_LABEL_API_KEY": "environment-key"},
    )

    assert config.api_key == "environment-key"
    assert config.model == "file-model"
    assert config.base_url == "https://file.example/v1"
    assert config.timeout == 45


def test_labeling_config_has_safe_empty_defaults(tmp_path) -> None:
    config = load_review_labeling_config(tmp_path / "missing.env", environ={})

    assert config.api_key == ""
    assert config.model == ""
    assert config.base_url == ""
    assert config.timeout == 60
