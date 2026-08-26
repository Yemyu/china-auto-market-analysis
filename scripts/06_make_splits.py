#!/usr/bin/env python3
"""Create the fixed chronological train, validation, and test splits.

Train ends at 2025-06, validation covers 2025-07 through 2025-12, and the
six-month test window starts at 2026-01.
"""
import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import _feature_join as fj

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES = os.path.join(BASE, "data", "processed", "sales_filtered_24m.csv")
OUTDIR = os.path.join(BASE, "data", "processed", "splits")
SCHEMA_VERSION = "v1"

TRAIN_END = "2025-06"   # train: 2022-01 .. 2025-06
VAL_END = "2025-12"     # val:   2025-07 .. 2025-12 ; test: 2026-01 .. 2026-06

LAG_COLS = ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6"]
CAL = ["month_sin", "month_cos", "year"]
FEAT_COLS = LAG_COLS + CAL + fj.CFG_COLS
META_COLS = ["series_name", "series_id", "date", "year", "month",
             "brand", "category", "category_en", "monthly_sales"]


def engineer_features(sm: pd.DataFrame) -> pd.DataFrame:
    """在完整排序面板上算日历 + lag/滚动特征 (因果: shift 只用过去)。"""
    sm = sm.sort_values(["series_name", "date"]).copy()
    g = sm.groupby("series_name")["monthly_sales"]
    sm["lag_1"] = g.shift(1)
    sm["lag_2"] = g.shift(2)
    sm["lag_3"] = g.shift(3)
    sm["roll_mean_3"] = g.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    sm["roll_mean_6"] = g.shift(1).rolling(6).mean().reset_index(level=0, drop=True)
    moy = sm["date"].dt.month
    sm["month_sin"] = np.sin(2 * np.pi * moy / 12)
    sm["month_cos"] = np.cos(2 * np.pi * moy / 12)
    sm["year"] = sm["date"].dt.year
    return sm


def assign_split(sm: pd.DataFrame) -> pd.DataFrame:
    te = pd.to_datetime(TRAIN_END + "-01")
    ve = pd.to_datetime(VAL_END + "-01")
    split = np.where(sm["date"] <= te, "train",
             np.where(sm["date"] <= ve, "val", "test"))
    return sm.assign(split=split)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    sales = pd.read_csv(SALES)
    sales["date"] = pd.to_datetime(sales["date"])
    sales["series_name"] = sales["series_name"].astype(str)
    sales["year"] = sales["date"].dt.year
    sales["month"] = sales["date"].dt.month

    print(f"[splits] 月度面板: {sales['series_name'].nunique()} 车系 / {len(sales)} 行")
    print(f"[splits] 时间范围: {sales['date'].min().date()} .. {sales['date'].max().date()}")

    sm = fj.join_cfg(sales)
    print(f"[splits] 限定 feature 主表(有配置)后: "
          f"{sm['series_name'].nunique()} 车系 / {len(sm)} 行")

    sm = engineer_features(sm)
    sm = assign_split(sm)

    te = pd.to_datetime(TRAIN_END + "-01")
    ve = pd.to_datetime(VAL_END + "-01")
    tr = sm[sm["split"] == "train"]
    va = sm[sm["split"] == "val"]
    te_df = sm[sm["split"] == "test"]
    assert tr["date"].max() <= te, "train 含晚于 TRAIN_END 的月份"
    assert va["date"].min() > te and va["date"].max() <= ve, "val 区间错误"
    assert te_df["date"].min() > ve, "test 含早于 VAL_END 的月份"
    assert sm.assign(m=sm["date"].dt.to_period("M").astype(str)).duplicated(["series_name", "m"]).sum() == 0
    n_train_avail = int(tr[FEAT_COLS].notna().all(axis=1).sum())
    print(f"[splits] 切分行数: train={len(tr)} (可用{ n_train_avail }) "
          f"val={len(va)} test={len(te_df)}")

    tr_out = tr[tr[FEAT_COLS].notna().all(axis=1)].copy()
    va_out = va.copy()
    te_out = te_df.copy()

    cols = META_COLS + FEAT_COLS + ["split"]
    tr_out[cols].to_csv(os.path.join(OUTDIR, "train.csv"), index=False)
    va_out[cols].to_csv(os.path.join(OUTDIR, "val.csv"), index=False)
    te_out[cols].to_csv(os.path.join(OUTDIR, "test.csv"), index=False)

    split_idx = sm[["series_name", "date", "split"]].copy()
    split_idx["date"] = split_idx["date"].dt.strftime("%Y-%m-%d")
    split_idx.to_csv(os.path.join(OUTDIR, "split_index.csv"), index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sales": "data/processed/sales_filtered_24m.csv",
        "source_config": "data/raw/feature.csv (车系×年, 因果回退 join)",
        "population": "feature 主表车系 (有配置), 月度面板",
        "time_cutoffs": {
            "train_end": TRAIN_END,
            "val_end": VAL_END,
            "test_end": "2026-06",
        },
        "n_series": int(sm["series_name"].nunique()),
        "n_rows": {
            "train_total": int(len(tr)), "train_usable": int(len(tr_out)),
            "val": int(len(va)), "test": int(len(te_df)),
        },
        "feature_columns": FEAT_COLS,
        "target": "monthly_sales",
        "leakage_guarantees": [
            "切分按绝对时间 (全局切点), 非随机/非按车系打乱",
            "lag/roll 特征由 groupby(series).shift 计算, 仅用真实过去销量",
            "配置 join 因果回退: 只用 <= 行年份的最新规格, 不借未来年份",
            "训练仅用 train.csv; val/test 仅供评估, 不参与训练",
        ],
        "annual_attribution_note": "年度配置归因使用按 series_name 分组的 GroupKFold(5)，与月度预测的时间切分相互独立。",
    }
    with open(os.path.join(OUTDIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    readme = build_readme(manifest)
    with open(os.path.join(OUTDIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("[splits] 已写出:")
    for fn in ["train.csv", "val.csv", "test.csv", "split_index.csv",
               "manifest.json", "README.md"]:
        p = os.path.join(OUTDIR, fn)
        print(f"        {fn:16s} {os.path.getsize(p):>9,} bytes")
    print("[splits] 时间切分与完整性检查通过。")


def build_readme(m: dict) -> str:
    n = m["n_rows"]
    return f"""# 月度销量预测时间切分

本目录由 `scripts/06_make_splits.py` 生成。所有月度预测模型读取同一组
train、validation 和 test 文件，避免各模型自行定义测试区间。

## 文件

| 文件 | 内容 |
|---|---|
| `train.csv` | 滞后特征齐全的训练行 |
| `val.csv` | 参数与方案选择 |
| `test.csv` | 最终评价 |
| `split_index.csv` | `series_name, date, split` 的最小切分索引 |
| `manifest.json` | 时间边界、行数、特征列、来源和防泄漏约束 |

## 时间边界

| 数据段 | 目标月份 | 用途 |
|---|---|---|
| Train | 截至 {m['time_cutoffs']['train_end']} | 模型训练 |
| Validation | 2025-07—{m['time_cutoffs']['val_end']} | 参数与方案选择 |
| Test | 2026-01—{m['time_cutoffs']['test_end']} | 最终评价 |

切分按全局自然月完成，不随机打乱。一个车系可以出现在三份文件中，但同一个月份只属于一个数据段。

## 当前规模

- 目标车系：{m['n_series']}；
- Train：{n['train_total']} 个候选车系月，其中 {n['train_usable']} 行具备完整滞后特征；
- Validation：{n['val']} 行；
- Test：{n['test']} 行。

## 防泄漏约束

1. 销量滞后和滚动均值由车系内 `shift` 计算，只引用目标月以前的销量。
2. 配置按时间因果回退：缺少当年配置时，只使用不晚于该年份的最近配置。
3. Validation 用于选择参数和方案；Test 只报告最终结果。
4. 固定起点测试从 2026-01 开始递归六个月。第二个月起需要的销量滞后来自此前预测，不能读取测试期真实销量。
5. 用户评论特征在主实验中统一冻结于 2026-01-01 之前；每个预测月的可用范围由评论时间特征脚本生成并审计。

## 读取示例

```python
import pandas as pd
train = pd.read_csv("data/processed/splits/train.csv")
val = pd.read_csv("data/processed/splits/val.csv")
test = pd.read_csv("data/processed/splits/test.csv")
```

年度产品配置分析不使用这组时间切分。它在车系年数据上执行 `GroupKFold(5)`，并按车系分组。
"""


if __name__ == "__main__":
    main()
