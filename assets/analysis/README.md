# 分析图

图片来自 371 车系预测实验或 646 车系完整年度配置分析。该目录只保存报告使用的最终图。

| 文件 | 表达的结论 |
|---|---|
| `forecast_review_feature_ablation.png` | 固定六个月压力测试中的用户口碑消融，用于辅助分析，不纳入滚动主结果 |
| `forecast_robustness.png` | 固定压力测试的车型簇 Bootstrap 区间、特征重要性及逐月改善稳定性 |
| `product_config_attribution.png` | 配置对年度销量差异的增量解释力与重要属性 |
| `user_needs_and_alerts.png` | 用户讨论重点、负面反馈集中维度和预警历史 |
| `cold_start_launch_curve.png` | 固定压力测试中新上市车型的冷启动折中验证 |

滚动单月主结果的数值以 `data/processed/forecast/rolling_origin_test_predictions.csv` 和 `rolling_origin_summary.json` 为准；现有分析图用于固定压力测试、配置分析和需求监测的辅助解读。
