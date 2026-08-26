# 月度销量预测时间切分

本目录由 `scripts/new_pipeline/06_make_splits.py` 生成。所有月度预测模型读取同一组 train、validation 和 test 文件，避免各模型自行定义测试区间。

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
| Train | 截至 2025-06 | 模型训练 |
| Validation | 2025-07—12 | 参数与方案选择 |
| Test | 2026-01—06 | 最终评价 |

切分按全局自然月完成，不随机打乱。一个车系可以出现在三份文件中，但同一个月份只属于一个数据段。

## 当前规模

- 目标车系：371；
- Train：12,036 个候选车系月，其中 9,468 行具备完整滞后特征；
- Validation：2,172 行；
- Test：2,226 行。

## 防泄漏约束

1. 销量滞后和滚动均值由车系内 `shift` 计算，只引用目标月以前的销量。
2. 配置按时间因果回退：缺少当年配置时，只使用不晚于该年份的最近配置。
3. Validation 用于选择参数和方案；Test 只报告最终结果。
4. 固定起点测试从 2026-01 开始递归六个月。第二个月起需要的销量滞后来自此前预测，不能读取测试期真实销量。
5. 用户评论特征在主实验中统一冻结于 2026-01-01 之前；每个预测月的可用范围另由评论时间特征脚本生成并审计。

## 读取示例

```python
import json
import pandas as pd

train = pd.read_csv("data/processed_new/splits/train.csv")
val = pd.read_csv("data/processed_new/splits/val.csv")
test = pd.read_csv("data/processed_new/splits/test.csv")

with open("data/processed_new/splits/manifest.json", encoding="utf-8") as handle:
    features = json.load(handle)["feature_columns"]
```

年度产品配置分析不使用这组时间切分。它在车系年数据上执行 `GroupKFold(5)`，并按车系分组。
