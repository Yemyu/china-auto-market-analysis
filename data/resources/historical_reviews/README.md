# 历史舆情标注资源

这个目录保留第一版项目中仍有复用价值的内容，不再保留旧版的整套数据目录和建模结果。

## 资源文件

`review_absa_reference.csv.gz` 是一张按 `review_id` 去重的本地压缩表，不随公开仓库上传。它包含：

- 39,496 条唯一汽车用户评论，覆盖 490 个车系；
- 评论原文、发布时间、车型、购车信息及用户提交的平台评分；
- 28,724 条旧版 DeepSeek 十维 ABSA 标签；
- 标签是否可用、旧 Prompt 版本和模型来源。

压缩文件可直接由 pandas 读取，不需要手动解压：

```python
import pandas as pd

reviews = pd.read_csv(
    "data/resources/legacy_sentiment/review_absa_reference.csv.gz",
    low_memory=False,
)
```

`manifest.json` 记录行数、时间范围、SHA-256 和标签语义，用于迁移与完整性检查。

## 为什么保留

新版 371 车系语料中的 16,538 条标签直接复用了这批历史 DeepSeek 结果。其余历史评论虽然不都进入当前预测样本，但仍可用于扩充车型覆盖、人工抽样、迁移学习或后续 Agent 检索，因此没有随旧版流程一起删除。

## 旧标签的限制

旧版标签取值为 `-1 / 0 / 1`：

- `-1`：负面；
- `1`：正面；
- `0`：原始 Prompt 设计为“没有提及”，但代码也会在字段缺失、解析异常或模型返回其他值时填入 0。

因此旧标签中的 0 不能作为“未提及”或“中性”的可靠真值。当前项目建模应使用 `data/sentiment_new/processed/unified_deepseek_absa_review_features.csv`，其中另外保存了统一的维度提及标记。

旧版调用使用 `deepseek-chat`，Prompt 的核心要求是：针对外观、内饰、空间、动力、操控、舒适、油耗、配置、智能化和性价比十个维度输出 JSON；未提及填 0，正面填 1，负面填 -1。旧流程没有保存原始模型响应和 token 账单，所以资源中只保留最终结构化标签。

## 重新生成

只有在旧源文件仍在本地时才需要执行：

```bash
conda run --no-capture-output -n nlp-sentiment \
  python scripts/new_pipeline/38_curate_legacy_sentiment_resource.py
```

生成过程不调用任何外部 API。
