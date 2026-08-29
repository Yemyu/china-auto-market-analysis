<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

# Data documentation

This directory contains source tables, temporal splits, the review corpus, model artifacts, and data audits. Reusable historical reviews and labels are isolated under `resources/`.

## Data layers

| Path | Contents | Used by current pipeline |
|---|---|---|
| `raw/` | Monthly sales and vehicle-specification source tables | Yes |
| `processed/` | Cleaned data, temporal splits, forecasts, attribution, and monitoring outputs | Yes |
| `reviews/raw/` | Collected reviews, source detail pages, and crawl manifests | After quality filtering |
| `reviews/processed/` | Strict corpus, structured labels, temporal features, and audits | Yes |
| `resources/` | Historical review and label archive | Reference or future extension |

## Three analysis samples

| Analysis | Final sample | Main filters |
|---|---:|---|
| Monthly sales forecast | 371 series | Continuous monthly sales, aligned specifications, and a complete test period |
| Product-specification analysis | 736 series; 2,007 series-year records | Annual sales and specifications aligned during 2022–2026 |
| User needs and risk | 24,175 reviews; 345 series | Complete text, valid publication time, and traceable source |

These are not successive reductions of one table. Each analysis defines its own sample from the data it requires.

## 1. Source tables

### `raw/monthly_sales.csv`

A PCauto-based monthly sales panel.

| Item | Value |
|---|---:|
| Records | 54,918 |
| Series | 1,017 |
| Period | 2022-01—2026-06 |
| Negative sales | 0 |
| Grain | Series × calendar month |

Main modeling fields:

| Field | Description |
|---|---|
| `year`, `month`, `period` | Time fields |
| `series_id`, `series_name` | Source ID and series name |
| `brand`, `category` | Brand and vehicle category |
| `monthly_sales` | Monthly sales |
| `数据来源` | Source marker |

Rank, cumulative sales, display price, and related fields are source-platform metadata and are excluded from forecasting features.

### `raw/feature.csv`

Vehicle specifications at a series-model-year grain, not a trim-level product list.

| Item | Value |
|---|---:|
| Records | 2,084 |
| Series | 766 |
| Fields | 84 |
| Unique key | `series_name, year` |
| Annual-sales coverage | 760 / 766 series |

The table covers:

- basic attributes: brand, model year, energy type, vehicle class, and list price;
- powertrain: engine, motor, gearbox, acceleration, and consumption;
- battery and range: capacity, chemistry, range, and charging;
- body: dimensions, wheelbase, weight, seats, and body style;
- safety and cabin: airbags, displays, seats, audio, and climate control.

Engine, motor, and battery columns contain structural missingness. A BEV has no engine parameters, while an ICE model has no battery parameters; these missing values should not automatically be treated as data errors.

## 2. Forecast sample and temporal split

`processed/splits/` stores the fixed split for 371 series:

| File | Period | Purpose |
|---|---|---|
| `train.csv` | through 2025-06 | Model fitting |
| `val.csv` | 2025-07—12 | Hyperparameter and design selection |
| `test.csv` | 2026-01—06 | Final evaluation |
| `split_index.csv` | Full sample | Split membership by row |
| `manifest.json` | — | Series count, row count, time boundaries, and version metadata |

The headline experiment is a rolling one-month forecast: each test month predicts the next month and can use the latest published previous-month sales, with model parameters locked within the six-month window. The fixed-origin protocol remains as a stress test: it starts in January 2026 and recursively uses prior predictions as lagged sales, never actual post-origin sales. The protocols answer different questions and are not ranked directly.

## 3. Review corpus

### Raw collection layer

`reviews/raw/` stores collected owner reviews and crawl manifests locally. Full records contain platform user identifiers and are excluded from the public repository; crawl manifests and de-identified analysis artifacts remain versioned.

Main audit files:

| File | Purpose |
|---|---|
| `dongchedi_incremental_manifest.csv` | Dongchedi incremental-crawl manifest |
| `autohome_incremental_manifest.csv` | Autohome incremental-crawl manifest |
| `autohome_incremental_review_details.csv` | Full Autohome review text |
| `processed/review_collection/autohome_id_resolutions.csv` | Series mapping and resolution |
| `processed/review_collection/sentiment_resolution_exceptions.csv` | Unresolved cases and stop rules |

### Strict modeling corpus

The local file `reviews/processed/target_371_review_corpus.csv` includes candidate reviews, quality flags, and source audit fields for the 371 target series. A modeling record must have:

1. valid series identity;
2. parseable publication time;
3. complete, quality-eligible text;
4. publication before the relevant forecast origin.

The final corpus contains 24,175 reviews across 345 series. Another 103 Autohome list-page summaries could not be resolved to full detail text; they remain in the audit table with `eligible_for_temporal_model=False`.

| Coverage definition | Series |
|---|---:|
| Any eligible review | 345 |
| Review available before the January 2026 fixed origin | 330 |
| Review available in the preceding 180 days at that origin | 272 |

## 4. Review labels and temporal features

### Review-level labels

The [structured review labels](./reviews/processed/review_aspect_labels.csv) store ten product dimensions:

`appearance`, `interior`, `space`, `power`, `control`, `comfort`, `fuel_consumption`, `configuration`, `intelligence`, and `value`.

Two items are stored separately for every dimension:

- whether the review explicitly mentions the dimension;
- polarity after mention: `-1` negative, `0` neutral, `1` positive.

Platform star ratings and text evidence are not interchangeable. If a review does not mention a dimension, it does not enter that dimension's positive or negative share as a neutral observation.

### Leakage-safe monthly features

The [fixed-origin review feature panel](./reviews/processed/review_features_by_series_month_fixed_origin.csv) has 13,866 rows across 371 series and 51 forecast months. It contains:

- cumulative reviews available before the origin;
- review count and availability in the previous 180 days;
- historical means across ten dimensions;
- 180-day positive, negative, and mention shares;
- composite aspect score and any-positive/negative shares;
- `information_cutoff_exclusive`, the information cutoff for that row.

Fixed-origin test features are frozen before `2026-01-01` for the stress test and review ablation. The rolling headline uses sales history available at each month; rolling forecast artifacts are stored under `processed/forecast/rolling_origin_*.{json,csv}`.

## 5. Analysis artifacts

### Data quality and repair: `processed/data_quality/`

| Artifact | Contents |
|---|---|
| `data_repair_summary.json` | Phase summary for sales zeros and cross-source series mapping |
| `sales_correction_register.csv` | Reviewable month/source-level sales correction register |
| `sales_panel_integrity_summary.json` | Full 54,918-row structural, derived-field, and model-cohort audit summary |
| `sales_series_risk_audit.csv` | Positive-history, zero-window, and model-impact audit for all 1,017 series |
| `sales_manual_verification_queue.csv` | External-verification targets ranked by severity and model impact |
| `sales_cutoff_cluster_audit.csv` | Clusters of last-positive months for detecting batch source cutoffs |
| `sales_zero_audit.csv` | Per-series positive-month coverage, positive runs, test zeros, and audit flags |
| `sales_zero_audit_repaired.csv` | Like-for-like zero audit after verified corrections only |
| `sales_zero_status_register.csv` | Source-gap, discontinuation, and unresolved labels for high-risk zeros |
| `series_mapping_audit.csv` | Exact, safely normalized, and unmatched sales/config names |
| `verified_sales_overlay_audit.csv` | Corrections actually applied and their sales deltas |
| `pcauto_recrawl_pilot_manifest.csv` | Request, identity, and parse audit for 90 six-month anchor pages across ten series |
| `pcauto_recrawl_pilot_diff.csv` | Row-level comparison for 540 recollected series-months against the frozen snapshot |
| `pcauto_recrawl_pilot_summary.json` | Coverage and difference summary for the batch-recollection pilot |
| `pcauto_recrawl_high_remaining_manifest.csv` | Connection-failure and retry audit for the 108-page second batch |
| `pcauto_recrawl_high_remaining_summary.json` | Current `blocked_no_successful_pages` state for the second batch |
| `alternative_sales_source_map.csv` | Explicit alternative-source series, brand/manufacturer, and page mapping; conflicts are never auto-merged |
| `alternative_sales_crosscheck_*.csv/json` | Alternative-source identity checks, monthly observations, diffs, and metric calibration |

Phase one accepts only exact names and unambiguous one-to-one normalized matches; it does not use fuzzy matching. The safe overlap grows from 371 to 379 series.

Phase two classifies the nine series with high 2024 volume and an all-zero 2026 test window. Model Y and Model 3 are confirmed same-source snapshot anomalies. Same-source year-comparison tables supply five missing 2022 rows and the 38 missing June 2024–December 2025 rows, and correct four nonzero 2022 values; monthly pages supply 12 January–June 2026 rows. Repaired 2022 and 2025 totals match the source-page annual totals for both series. The iCAR 03 test zeros are supported by a discontinued status on the same source. The other six series remain unchanged and pending manual verification; the pipeline neither infers their state nor merges them into differently named series. Raw `monthly_sales.csv` is never overwritten, and the repair helper fails closed on duplicate keys or an original-value mismatch.

The known Model Y and Model 3 history gaps are now complete. After unique-key, source, and annual-total checks, the overlay is included in the reproducible panel. The remaining high-priority queue is retained as an audit item and does not change the locked test window or metric protocol. The repaired fixed stress-test combined method scores 39.07%, while the rolling one-month headline scores 31.34%; both are recomputed from saved artifacts.

Phase three scans all 54,918 rows and 1,017 series locally. Hard structural checks find no duplicate series-month keys, negative sales, name/ID conflicts, or missing calendar rows, but this does not establish source truth. Annual cumulative sales, brand-month totals, positive-series counts, zero labels, and rank availability all reproduce `monthly_sales` deterministically and therefore cannot serve as independent validation.

Phase four uses `scripts/42_recrawl_pcauto_sales.py` to recollect pages by exact source ID. June and December serve as six-month anchors, raw HTML is cached in a Git-ignored directory, and no source file is overwritten. The first ten-series pilot requests 90 pages successfully and parses 540 series-months: 186 exactly match the snapshot, 353 are shown as “—” by the source and are also zero in the snapshot, and the only definite difference is Wuling Hongguang S in June 2026—2,215 on the source versus zero in the snapshot. Exact ID, page-name, and manufacturer-table checks support adding that row to the overlay.

Phase five uses `scripts/43_crosscheck_alternative_sales.py` with an explicit alternative-source map. A page must pass series-name and brand/manufacturer checks before its monthly values are accepted. Model Y and Model 3 provide 72 repaired overlap months for metric calibration, all of which match exactly. The first nine mapped pages yield 324 monthly observations. MG HS in September 2022 and Weltmeister W6 in September 2022 differ by 16 and 4 units respectively; a second independent source confirms both corrections before they enter the overlay. Changan CS55PLUS matches in later months but is discontinuous at a generation boundary, so its original value is retained—demonstrating why a name match alone is insufficient. The overlay now repairs 62 rows across five series and restores a net 1,357,467 units; external review contains 21 high- and 33 medium-priority series.

A source dash is not automatically interpreted as a true zero. The 353 pilot dashes are positioned against the raw positive-sales span: 120 before the first positive month, 96 after the last positive month, 30 inside the positive span, and 107 with no positive reference anywhere. Boundary cases remain launch/discontinuation or source-coverage candidates, internal cases receive priority review, and no-reference series do not enter a normal history model automatically. The collector also separates a successful page containing a dash from a page that was never retrieved; the latter is labeled `not_retrieved` and cannot support a data conclusion.

The second batch targets the other twelve high-priority series and 108 pages. Every request in this run failed at the TLS connection layer, producing `blocked_no_successful_pages`; no new sales, differences, or corrections were inferred. The frozen roster can be retried when source connectivity returns.

The scan also confirms that the previous “at least 24 consecutive months” rule counts padded panel rows rather than positive-sales history. Every one of the 1,017 series has the same 54 rows from January 2022 through June 2026, so every series passes before the configuration join. The raw 371-series cohort contains two series with no positive sales anywhere; the Wuling Hongguang S repair reduces that count to one, while 32 still have no positive sales before the training cutoff. Eligibility must be rebuilt with lifecycle-aware positive-history rules after source gaps are resolved; rows should not be deleted and models rerun prematurely.

### Sales forecasting: `processed/forecast/`

| Artifact | Contents |
|---|---|
| `review_feature_ablation_summary.csv` | Model comparison on 371 series |
| `review_feature_predictions.csv` | Series-month test predictions |
| `review_feature_series_metrics.csv` | Per-series errors |
| `forecast_robustness_bootstrap.csv` | Series-cluster bootstrap |
| `review_feature_shap_importance.csv` | Feature contributions |
| `cold_start_launch_curve_summary.json` | Cold-start method and results |
| `forecast_benchmark_comparison.csv` | Naive baselines and final models on all 371 series |
| `rolling_origin_summary.json` | Rolling one-month headline, historical-origin validation, and locked-test summary |
| `rolling_origin_validation.csv` | Rolling versus fixed protocol across four historical origins |
| `rolling_origin_test_predictions.csv` | January–June 2026 rolling predictions and same-scenario naive baselines |
| `direct_multihorizon_validation.csv` | Rolling validation summary for direct multi-horizon candidates |
| `direct_multihorizon_summary.json` | Locked selection protocol and rejection decision for the direct experiment |

The rolling one-month headline reaches 31.34% global WMAPE versus 40.99% for the last-observed-value naive baseline, a 9.65-point (about 23.5%) reduction. In the fixed six-month stress test, the trailing-six-month mean is 69.31% and the reviews-plus-cold-start combined method is 39.07%, a 43.6% reduction; these are different forecast tasks.

Review enhancement in the fixed stress test improves the point estimate by 0.884 pp; the 5,000-replicate series-cluster bootstrap 95% interval is −0.284 to 2.108 pp and crosses zero, so review features remain supporting evidence. The cold-start statistical strategy handles nine history-poor series and does not replace the 371-series rolling headline.

The direct multi-horizon experiment selected blend weights on three earlier origins and reached 30.42% WMAPE at the July 2025 fixed-origin validation. Its locked specification then rose to 47.46% on the 362 history-bearing series in the 2026 test, underperforming the existing recursive model; it was therefore rejected under the predefined rule and does not replace the headline result. Row-level experimental predictions remain local and are ignored by Git.

### Product specifications: `processed/product/`

| Artifact | Contents |
|---|---|
| `config_attribution_ablation.csv` | Sequential year, brand, and specification ablation |
| `config_importance_annual.csv` | Annual feature importance |

The complete model reaches a five-fold grouped cross-validated R² of 0.301; annual cross-sectional WMAPE is supporting error and is not directly comparable with monthly sales-forecast WMAPE.

### User needs and risk: `processed/user_feedback/`

| Artifact | Contents |
|---|---|
| `user_need_aspect_summary.csv` | Mention and polarity distribution by dimension |
| `user_need_keywords.csv` | Dimension-level keywords |
| `user_need_topics.csv` | Topics within each dimension |
| `sentiment_monitoring_windows.csv` | Adjacent 180-day window statistics |
| `sentiment_alerts.csv` | Historical and current rule alerts |

An alert is a manual-review entry, not a confirmed product defect.

## 6. Curated historical resource

`resources/historical_reviews/` stores the reusable historical review archive:

| File | Contents |
|---|---|
| `review_absa_reference.csv.gz` | Local full-text archive: 39,496 deduplicated reviews, including 28,724 with historical ten-dimension labels |
| `manifest.json` | Counts, time range, SHA-256, and label semantics |
| `README.md` | Usage and limitations |

The current corpus reuses 16,538 previously generated historical labels. Full text, publication time, series, purchase context, and platform ratings stay in the local archive; the public repository contains de-identified labels and aggregates.

A historical label of `0` cannot reliably separate “not mentioned,” “neutral,” and parser fallback. It is not used directly as mention ground truth; the current feature table uses a separate uniform mention flag.

## 7. Reproduction

Install the dependencies from the root `requirements.txt`, then run the scripts under `scripts/` in order.

The main review and modeling sequence is:

```text
18  build target-series corpus
21  audit multi-source quality
25  validate temporal availability
27  build platform-rating and lexicon features
28  aggregate local review monthly features
30  backfill missing labels (optional, requires an API)
31  merge review-level labels
32  aggregate leakage-safe monthly features
33  run the 371-series forecast ablation
34  analyze robustness
35  build user-needs and risk monitoring outputs
36  validate the cold-start method
37  generate the report notebooks
```

Source-platform data remains subject to the respective owners' rights. Data in this directory is included only for learning, research, and project demonstration.
