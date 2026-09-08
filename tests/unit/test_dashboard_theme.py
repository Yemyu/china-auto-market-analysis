from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_inner_pages_use_the_approved_paper_theme():
    for name in ("attribution", "absa", "alerts", "drilldown", "forecast"):
        html = (ROOT / "app" / f"{name}.html").read_text(encoding="utf-8")
        assert "static/css/paper.css" in html
        assert "static/js/paper-theme.js" in html
        assert "paper-theme" in html.split("<body", 1)[1].split(">", 1)[0]
