# 历史评论资源

> **历史归档说明**：本页描述早期评论标签资源，服务于复现、抽样核验和扩展研究，不是当前销量主模型的独立结果。当前正式指标、预测口径和看板解释以根目录 [README.md](../../../README.md)、[数据说明](../../README.md) 和 `notebook/` 为准。

## 资源文件

`review_absa_reference.csv.gz` 是按 `review_id` 去重的本地压缩表，不随公开仓库上传。它包含：

- 39,496 条唯一汽车用户评论，覆盖 490 个车系；
- 评论原文、发布时间、车型、购车信息及用户提交的平台评分；
- 28,724 条十维评论标签；
- 标签状态、提示词版本和模型来源。

压缩文件可直接由 pandas 读取，不需要手动解压：

```python
import pandas as pd

reviews = pd.read_csv(
    "data/resources/historical_reviews/review_absa_reference.csv.gz",
    low_memory=False,
)
```

`manifest.json` 记录行数、时间范围、SHA-256 和标签语义。当前 371 车系语料复用了其中 16,538 条标签；其余记录可用于扩充车型覆盖、抽样核验或检索。

## 标签限制

历史标签取值为 `-1 / 0 / 1`：

- `-1`：负面；
- `1`：正面；
- `0`：提示词定义为“没有提及”，但字段缺失、解析异常或其他返回值也可能落为 0。

因此 `0` 不能作为“未提及”或“中性”的可靠真值。建模使用 `data/reviews/processed/review_aspect_labels.csv`，其中单独保存统一的维度提及标记。

历史调用使用 `deepseek-chat`。归档没有保留原始响应和 token 账单，只保留结构化标签及其来源字段。
