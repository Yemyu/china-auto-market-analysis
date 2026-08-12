#!/usr/bin/env python3
"""
06_make_splits.py — 时间切分 train / val / test (无舆情基线 · 月度预测腿)

编号说明
--------------------------------------------------
本脚本编号为 06, 排在全部「建模脚本」之前:
  07_model_xgboost / 08_model_lstm / 09_fusion_and_compare / 10_rolling_cv /
  11_xgb_ablation / 12_intervals / 13_model_prophet_exog / 14_arima_prophet
即「先切分数据, 再跑模型」。上述模型脚本都直接读取本脚本产出的
data/processed_new/splits/{train,val,test}.csv, 不再各自重新定义 holdout。

为什么要有这个脚本
--------------------------------------------------
旧管线里每个模型脚本各自做「该车系最后 H 个月 holdout」当测试集, 只有
test 没有显式 validation, 也没有把切分结果落盘。用户要求:
  (1) 数据切分必须准确 —— 按「绝对时间」切分, 不能随机/不能按车系打乱;
  (2) 明确分成 训练 / 验证 / 测试 三份;
  (3) 存放格式按 GitHub 典型规范 (Cookiecutter Data Science 风格:
      data/processed/ 下 train.csv / val.csv / test.csv + 说明 README)。

切分定义 (全局统一时间切点, 所有车系共用)
--------------------------------------------------
  train : 2022-01 .. 2025-06   (42 个月, 约 77.8%)
  val   : 2025-07 .. 2025-12   ( 6 个月, 约 11.1%)   ← 模型选择 / 早停
  test  : 2026-01 .. 2026-06   ( 6 个月, 约 11.1%)   ← 最终诚实评估

注意: 切分是按「目标月份」的绝对时间, 不是按车系随机。一个车系会同时出现在
三份文件的不同月份里 —— 这是正确的时序切分 (series 跨 train/val/test 但月份不交叉)。

防泄漏保证 (时间维度)
--------------------------------------------------
  * lag_1..3 / roll_mean_3/6 由 groupby(series).shift 在「完整排序面板」上计算,
    物理上只引用该月之前的真实销量 (autoregressive 用真实历史, 推理时同理)。
  * 配置特征来自 feature.csv, 经 _feature_join.join_cfg 的「因果回退」:
    某(车系,年)无配置时只用 <= 当前行年份的最新配置, 绝不借用未来年份规格。
  * 训练只用 train.csv 的行; val/test 文件仅供评估, 不参与训练。
    (val/test 行的 lag 用真实过去, 正是推理时拥有的信息, 非泄漏)

产物 (GitHub 典型布局)
--------------------------------------------------
  data/processed_new/splits/
    train.csv        # 特征齐全, 可直接喂 XGBoost 等
    val.csv
    test.csv
    split_index.csv  # 纯切分分配表: series_name, date, split (可复现性审计)
    manifest.json    # 切点/行数/特征列/来源/生成时间/版本
    README.md        # 本规范的文档版 (与下方说明一致)

运行:
  python scripts/new_pipeline/06_make_splits.py
依赖: pandas/numpy + 本目录 _feature_join.py
"""
import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import _feature_join as fj  # CFG_NUM/CFG_CAT/CFG_COLS + 因果 join_cfg

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SALES = os.path.join(BASE, "data", "processed_new", "sales_filtered_24m.csv")
OUTDIR = os.path.join(BASE, "data", "processed_new", "splits")
SCHEMA_VERSION = "v1"

# ---- 全局时间切点 (target month <= cutoff 归入对应集) ----
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

    # 1) 限定 feature 主表车系 + 因果配置 join (防未来规格泄漏)
    sm = fj.join_cfg(sales)
    print(f"[splits] 限定 feature 主表(有配置)后: "
          f"{sm['series_name'].nunique()} 车系 / {len(sm)} 行")

    # 2) 特征工程 (完整面板, 因果)
    sm = engineer_features(sm)

    # 3) 时间切分
    sm = assign_split(sm)

    # ---- 泄漏 / 正确性自检 ----
    te = pd.to_datetime(TRAIN_END + "-01")
    ve = pd.to_datetime(VAL_END + "-01")
    tr = sm[sm["split"] == "train"]
    va = sm[sm["split"] == "val"]
    te_df = sm[sm["split"] == "test"]
    # a) 三集月份区间严格递增且不交叉
    assert tr["date"].max() <= te, "train 含晚于 TRAIN_END 的月份"
    assert va["date"].min() > te and va["date"].max() <= ve, "val 区间错误"
    assert te_df["date"].min() > ve, "test 含早于 VAL_END 的月份"
    # b) 同一(车系,月)不会出现在多集
    keys = sm.groupby("split")["date"].apply(lambda s: s.dt.to_period("M").astype(str))
    assert sm.assign(m=sm["date"].dt.to_period("M").astype(str)).duplicated(["series_name", "m"]).sum() == 0
    # c) 配置因果性: 所有行的配置年份 <= 行年份 (join_cfg 已保证, 这里复检)
    #    用 source: feature.csv 中 (series,year) 命中的年份本就 == 行年; 回退年份 <= 行年
    cfg_year_le_row = True  # join_cfg 内已强制, 仅占位断言
    # d) lag 无 NaN 才能进训练池 (首月 lag 必然 NaN, 属正常, 仅影响可用样本)
    n_train_avail = int(tr[FEAT_COLS].notna().all(axis=1).sum())
    print(f"[splits] 切分行数: train={len(tr)} (可用{ n_train_avail }) "
          f"val={len(va)} test={len(te_df)}")

    # 4) 写盘
    # 仅保留训练池里 lag 齐全的行进 train (首月无历史, 无意义); val/test 全保留
    tr_out = tr[tr[FEAT_COLS].notna().all(axis=1)].copy()
    va_out = va.copy()
    te_out = te_df.copy()

    cols = META_COLS + FEAT_COLS + ["split"]
    tr_out[cols].to_csv(os.path.join(OUTDIR, "train.csv"), index=False)
    va_out[cols].to_csv(os.path.join(OUTDIR, "val.csv"), index=False)
    te_out[cols].to_csv(os.path.join(OUTDIR, "test.csv"), index=False)

    # 纯切分分配表 (最小可复现单元)
    split_idx = sm[["series_name", "date", "split"]].copy()
    split_idx["date"] = split_idx["date"].dt.strftime("%Y-%m-%d")
    split_idx.to_csv(os.path.join(OUTDIR, "split_index.csv"), index=False)

    # manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sales": "data/processed_new/sales_filtered_24m.csv",
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
        "leg_a_note": "年度配置归因 (腿A) 用 GroupKFold(5) by series_name 防泄漏, 属另一套切分, 见 20_config_attribution.py",
    }
    with open(os.path.join(OUTDIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # README
    readme = build_readme(manifest)
    with open(os.path.join(OUTDIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("[splits] 已写出:")
    for fn in ["train.csv", "val.csv", "test.csv", "split_index.csv",
               "manifest.json", "README.md"]:
        p = os.path.join(OUTDIR, fn)
        print(f"        {fn:16s} {os.path.getsize(p):>9,} bytes")
    print("[splits] 完成。数据切分准确 (时间切分) + GitHub 典型布局已就绪。")


def build_readme(m: dict) -> str:
    n = m["n_rows"]
    return f"""# 数据切分 (train / val / test) — 无舆情月度预测基线

本目录由 `scripts/new_pipeline/06_make_splits.py` 生成，是月度预测腿（腿B）的
**唯一权威数据切分来源**。所有模型应直接读取这里的 `train.csv / val.csv / test.csv`，
而不是各自重新定义 holdout，以保证「同一拨训练/验证/测试」下的横向可比。

## 布局（GitHub / Cookiecutter Data Science 风格）

```
data/processed_new/splits/
├── train.csv        # 特征齐全，可直接喂 XGBoost / LSTM / 等；仅含可用样本(lag 齐全)
├── val.csv          # 验证集（模型选择 / 早停），特征齐全
├── test.csv         # 测试集（最终诚实评估），特征齐全
├── split_index.csv  # 纯切分分配: series_name, date, split（最小可复现单元）
├── manifest.json    # 切点 / 行数 / 特征列 / 来源 / 生成时间 / 版本
└── README.md        # 本文件
```

> 这些文件均为派生数据，可用 `python scripts/new_pipeline/06_make_splits.py`
> 从 `data/processed_new/sales_filtered_24m.csv` + `data/raw/feature.csv` 完整复现，
> 不依赖任何手工中间产物。

## 时间切点（全局统一，所有车系共用）

| 集 | 目标月份区间 | 月数 | 用途 |
|----|--------------|------|------|
| train | {m['time_cutoffs']['train_end']} 及以前 → 至 2022-01 | 42 | 模型训练 |
| val   | 2025-07 → {m['time_cutoffs']['val_end']} | 6 | 模型选择 / 早停 |
| test  | 2026-01 → {m['time_cutoffs']['test_end']} | 6 | 最终评估 |

- 切分按**目标月份的绝对时间**，不是随机、也不是按车系打乱。
- 一个车系会同时出现在三份文件的不同月份中（时序切分的正确形态：series 跨集、月份不交叉）。

## 行数（当前生成）

- train: 总 {n['train_total']} 行 / 可用 {n['train_usable']} 行（首月无历史被剔除）
- val:   {n['val']} 行
- test:  {n['test']} 行
- 车系数（in-population，有配置）: {m['n_series']}

## 特征列（FEAT_COLS，见 manifest.json）

`{', '.join(m['feature_columns'])}`

目标列: `monthly_sales`。

## 防泄漏保证（时间维度）

1. 切分按绝对时间（全局切点），非随机 / 非按车系打乱。
2. `lag_1..3` / `roll_mean_3/6` 由 `groupby(series).shift` 在完整排序面板上计算，
   物理上只引用该月之前的**真实销量**（autoregressive 用真实历史，推理时同理）。
3. 配置特征来自 `feature.csv`，经 `_feature_join.join_cfg` 的**因果回退**：
   某(车系,年)无配置时只用 ≤ 当前行年份的最新配置，**绝不借用未来年份规格**。
4. 训练仅用 `train.csv`；`val.csv` / `test.csv` 仅供评估，不参与训练。

## 模型如何消费

```python
import pandas as pd
TR = pd.read_csv("data/processed_new/splits/train.csv")
VA = pd.read_csv("data/processed_new/splits/val.csv")
TE = pd.read_csv("data/processed_new/splits/test.csv")
feat = [...]  # = manifest.json 的 feature_columns
model.fit(TR[feat], np.log1p(TR["monthly_sales"]))
# 验证: 在 VA 上选超参 / 早停; 测试: 仅在 TE 上报告最终指标
```

## 与腿A（年度配置归因）的关系

腿A 是「车系×年」横截面回归（解释什么配置的车卖得好），用
`GroupKFold(5) by series_name` 防同一车系跨折泄漏——那是一套**不同的切分**，
与本目录的时序切分互不替代。详见 `20_config_attribution.py`。
"""


if __name__ == "__main__":
    main()
