# 数据切分 (train / val / test) — 无舆情月度预测基线

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
| train | 2025-06 及以前 → 至 2022-01 | 42 | 模型训练 |
| val   | 2025-07 → 2025-12 | 6 | 模型选择 / 早停 |
| test  | 2026-01 → 2026-06 | 6 | 最终评估 |

- 切分按**目标月份的绝对时间**，不是随机、也不是按车系打乱。
- 一个车系会同时出现在三份文件的不同月份中（时序切分的正确形态：series 跨集、月份不交叉）。

## 行数（当前生成）

- train: 总 12036 行 / 可用 9468 行（首月无历史被剔除）
- val:   2172 行
- test:  2226 行
- 车系数（in-population，有配置）: 371

## 特征列（FEAT_COLS，见 manifest.json）

`lag_1, lag_2, lag_3, roll_mean_3, roll_mean_6, month_sin, month_cos, year, official_price_wan, engine_max_power_kw, engine_max_torque_nm, battery_capacity_kwh, battery_range_km, length_mm, width_mm, height_mm, wheelbase_mm, curb_weight_kg, seat_count, door_count, trunk_volume_l, acceleration_0_100_s, fuel_consumption_l_100km, energy_type_enc, vehicle_class_enc, brand_name_enc, body_structure_enc, gearbox_type_enc, seat_material_enc`

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
