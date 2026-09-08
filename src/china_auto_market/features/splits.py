"""Create chronological train, validation, and test splits shared by two protocols.

Train ends at 2025-06, validation covers 2025-07 through 2025-12, and the
six-month evaluation window starts at 2026-01. The rolling one-month protocol
uses these dates while revealing each previous realised month; the fixed-origin
protocol recursively withholds post-origin realised sales as a stress test.
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from china_auto_market.features import configuration as fj
from china_auto_market.paths import PROJECT_ROOT

BASE = str(PROJECT_ROOT)
SALES = os.path.join(BASE, "data", "processed", "sales_filtered_24m.csv")
COHORT = os.path.join(
    BASE, "data", "reviews", "processed", "target_371_review_coverage.csv"
)
OUTDIR = os.path.join(BASE, "data", "processed", "splits")
SCHEMA_VERSION = "sales-lags-v2"

TRAIN_END = "2025-06"   # train: 2022-01 .. 2025-06
VAL_END = "2025-12"     # val:   2025-07 .. 2025-12 ; test: 2026-01 .. 2026-06

LAG_COLS = ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6"]
CAL = ["month_sin", "month_cos", "year"]
FEAT_COLS = LAG_COLS + CAL + fj.CFG_COLS
STORED_FEATURE_COLS = LAG_COLS + CAL
# Optional long-memory features used by the rolling one-month candidate. They
# are written to the shared split files but are not part of the fixed-origin
# stress-test feature set.
SEASONAL_COLS = ["lag_12", "roll_mean_12"]
META_COLS = ["series_name", "series_id", "date", "year", "month",
             "brand", "category", "category_en", "monthly_sales"]


def engineer_features(sm: pd.DataFrame) -> pd.DataFrame:
    """在完整排序面板上算日历 + lag/滚动特征 (因果: shift 只用过去)。"""
    # Reset the index after sorting so rolling assignments align with rows.
    # Without this, a merge-generated non-contiguous index can silently place
    # rolling means on the wrong rows and discard otherwise valid training rows.
    sm = sm.sort_values(["series_name", "date"]).reset_index(drop=True)
    if sm["date"].isna().any() or sm.duplicated(["series_name", "date"]).any():
        raise ValueError("Sales features require valid unique series/month keys")
    months = sm["date"].dt.year * 12 + sm["date"].dt.month
    if not sm["date"].eq(sm["date"].dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("Sales dates must be calendar month starts")
    if months.groupby(sm["series_name"]).diff().dropna().ne(1).any():
        raise ValueError("Sales panel has missing calendar months; refusing row-offset lags")
    g = sm.groupby("series_name", sort=False)["monthly_sales"]
    sm["lag_1"] = g.shift(1)
    sm["lag_2"] = g.shift(2)
    sm["lag_3"] = g.shift(3)
    # The rolling operation must stay grouped as well as the lag operation.
    # Calling ``rolling`` directly on ``g.shift(...)`` would carry values over
    # from the previous series at each group boundary.
    sm["roll_mean_3"] = g.transform(lambda values: values.shift(1).rolling(3).mean())
    sm["roll_mean_6"] = g.transform(lambda values: values.shift(1).rolling(6).mean())
    sm["lag_12"] = g.shift(12)
    sm["roll_mean_12"] = g.transform(lambda values: values.shift(1).rolling(12).mean())
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(OUTDIR))
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(SALES)
    sales["date"] = pd.to_datetime(sales["date"])
    sales["series_name"] = sales["series_name"].astype(str)
    cohort = set(pd.read_csv(COHORT, usecols=["series_name"])["series_name"].astype(str))
    if len(cohort) != 371:
        raise ValueError(f"Expected the frozen 371-series evaluation cohort, found {len(cohort)}")
    sales = sales.loc[sales["series_name"].isin(cohort)].copy()
    missing_cohort = cohort - set(sales["series_name"])
    if missing_cohort:
        raise ValueError(f"Frozen evaluation series missing from sales panel: {sorted(missing_cohort)}")
    sales["year"] = sales["date"].dt.year
    sales["month"] = sales["date"].dt.month

    print(f"[splits] 月度面板: {sales['series_name'].nunique()} 车系 / {len(sales)} 行")
    print(f"[splits] 时间范围: {sales['date'].min().date()} .. {sales['date'].max().date()}")

    # Configuration fitting belongs to each model window, not shared splits.
    sm = engineer_features(sales)
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
    n_train_avail = int(tr[STORED_FEATURE_COLS].notna().all(axis=1).sum())
    print(f"[splits] 切分行数: train={len(tr)} (可用{ n_train_avail }) "
          f"val={len(va)} test={len(te_df)}")

    tr_out = tr[tr[STORED_FEATURE_COLS].notna().all(axis=1)].copy()
    va_out = va.copy()
    te_out = te_df.copy()

    if (len(tr_out), len(va_out), len(te_out)) != (13356, 2226, 2226):
        raise ValueError("Locked split row counts changed")
    cols = list(dict.fromkeys(META_COLS + STORED_FEATURE_COLS + SEASONAL_COLS + ["split"]))
    tr_out[cols].to_csv(output / "train.csv", index=False)
    va_out[cols].to_csv(output / "val.csv", index=False)
    te_out[cols].to_csv(output / "test.csv", index=False)

    split_idx = sm[["series_name", "date", "split"]].copy()
    split_idx["date"] = split_idx["date"].dt.strftime("%Y-%m-%d")
    split_idx.to_csv(output / "split_index.csv", index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sales": "data/processed/sales_filtered_24m.csv",
        "configuration_policy": "deferred-fit-window-v1",
        "source_config": "Loaded separately by each forecast evaluation; not preprocessed or stored in splits",
        "population": "frozen 371-series evaluation cohort with complete calendar months",
        "source_cohort": "data/reviews/processed/target_371_review_coverage.csv",
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
        "feature_columns": STORED_FEATURE_COLS,
        "deferred_configuration_columns": fj.CFG_COLS,
        "optional_feature_columns": SEASONAL_COLS,
        "target": "monthly_sales",
        "panel_policy": (
            "Retain every cohort series-month; store only sales, lags and calendar features. "
            "Configuration imputation and encoding are fitted by origin-specific consumers."
        ),
        "leakage_guarantees": [
            "切分按绝对时间 (全局切点), 非随机/非按车系打乱",
            "lag/roll 特征由 groupby(series).shift 计算, 仅用真实过去销量",
            "切分生成不读取、填充或编码配置；消费者必须按预测起点单独拟合配置规则",
            "评估车系固定为既有371车系，避免数据修复前后因新增名称映射改变样本",
            "开发阶段用 train 拟合、val 选型；选定后在 train+val 重新拟合；test 不参与参数选择或权重拟合",
        ],
        "availability_limitations": [
            "销量源未逐条恢复历史发布日期及修订版本；滚动协议假定上月销量可用",
            "配置缺少年内发布时间；消费者采用年度代理假设，不等于完整历史发布时间恢复",
        ],
        "annual_attribution_note": "年度配置归因使用按 series_name 分组的 GroupKFold(5)，与月度预测的时间切分相互独立。",
    }
    with open(output / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    readme = build_readme(manifest)
    with open(output / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print("[splits] 已写出:")
    for fn in ["train.csv", "val.csv", "test.csv", "split_index.csv",
               "manifest.json", "README.md"]:
        p = output / fn
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
| `train.csv` | 基础滞后特征齐全的训练行（季节特征在历史起始处可为空） |
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

月度面板保留每个目标车系的完整自然月。配置缺失不影响销量行保留；本目录只保存销量、日历和滞后，不保存统一填充或编码后的配置。缺月会阻止生成，不将更早月份误当成上月。

## 时间约束与尚存限制

1. 销量滞后和滚动均值由车系内 `shift` 计算，只引用目标月以前的销量；12 个月季节特征同样遵守该约束。
2. 训练消费者单独读取原始配置，按预测起点限制可用年份，并在实际训练行中拟合填充和类别规则。
3. 无配置记录的销售行不删除；训练窗口全缺数值列使用0占位，未知类别使用−1。这些占位不存入共享切分文件。
4. Validation用于选型；选定后最终模型使用train+val重新拟合至2025-12，测试期权重固定。
5. 固定起点压力测试从 2026-01 开始递归六个月。第二个月起需要的销量滞后来自此前预测，不能读取测试期真实销量；滚动主协议则在每月更新时使用已公布的上月真实销量。
6. 固定场景评论冻结于2026-01-01前；滚动评论辅助实验按各月截止日更新，销量滚动主模型不使用评论。

销量源未逐条恢复历史发布/修订版本，配置源也没有年内发布时间；年度代理规则不代表完整点时隔离。年度配置分析采用独立折内预处理。

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
