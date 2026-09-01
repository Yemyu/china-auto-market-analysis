"""Matplotlib defaults for charts with Chinese labels."""

import matplotlib.pyplot as plt


CJK_FONT_FALLBACKS = [
    "PingFang SC",
    "Heiti SC",
    "Hiragino Sans GB",
    "SimHei",
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "DejaVu Sans",
]


def configure_chinese_fonts() -> None:
    """Apply the font fallback used by the analysis charts."""
    plt.rcParams["font.sans-serif"] = CJK_FONT_FALLBACKS
    plt.rcParams["axes.unicode_minus"] = False
