"""Global matplotlib font setup for Chinese labels."""
import matplotlib.pyplot as plt

# Prefer macOS system CJK fonts, then common Windows/Linux fallbacks, then DejaVu
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',
    'Heiti SC',
    'Hiragino Sans GB',
    'SimHei',
    'Noto Sans CJK SC',
    'Microsoft YaHei',
    'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False  # avoid tofu minus signs
