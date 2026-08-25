<p align="center">
  <a href="./README.md">🇨🇳 中文</a> &nbsp;|&nbsp; <a href="./README_EN.md">🌐 English</a>
</p>

# Data Documentation

This file is the single source of truth for all China Automotive Market Analysis datasets, covering two core tables — monthly sales and vehicle configurations — plus user sentiment data (Phase B), for modeling, analysis, and the dashboard. All data come from public automotive platforms and are used for learning, research, and demonstration only.

## Data Sources

| Data category | Source platform | Collection method |
|---------------|----------------|-------------------|
| Monthly sales (raw) | 太平洋汽车 (pcauto) | Aggregated collection |
| Vehicle configs (raw) | 汽车之家 (autohome), 太平洋汽车 (pcauto) & other public platforms | Aggregated collection |
| User sentiment (Phase B) | 懂车帝 (dongchedi) public reviews | `scripts/01_crawl_reviews.py` automated crawler |

## Project Data Overview

The two tables are aligned directly by `series_name` (the new pipeline needs no legacy `series_mapping` bridge):

```
monthly_sales ──series_name──┐
                             ├──> align → 371 "monthly+config" series (Leg B) + 736 series with annual sales (Leg A)
feature       ──series_name──┘
```

For Leg B monthly forecasting: `monthly_sales` is aligned to `feature` by series (config joined on `(series_name, year)`) → filtered to 371 in-population series (with both monthly panel and configs, continuous coverage 2022-01~2026-06).

---

## 1. `monthly_sales.csv` — Monthly Sales

### Overview

| Item | Value |
|------|-------|
| Total records | 54,918 rows |
| Series | 1,017 |
| Brands | 162 |
| Time range | 2022-01 ~ 2026-06 |
| Total sales | ~70.29 million units |
| Data quality | 0 negative sales ✅; 70.6% are zero sales (suspension / pre-launch gaps, not anomalies) |
| Main source | pcauto (100%) |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Year (2022~2026) |
| `month` | int | Month (1~12) |
| `period` | str | Year-month label (e.g. `Jan-22`) |
| `series_id` | str | pcauto numeric series ID |
| `source_series_id` | str | Source-platform series ID (e.g. `sgXXXXX`) |
| `series_name` | str | Series name (cross-table join key) |
| `brand` | str | Brand |
| `category` | str | SUV / Sedan / MPV |
| `monthly_sales` | int | **Monthly sales (units)** |
| `数据来源` | str | pcauto |

> Other columns (`零销量类型`, `record_status`, `period_status`, `website_cumulative_sales`, `source_rank`, `source_last_rank`, `source_official_price`, `品牌月总销量`, `品牌车型数`) are platform metadata and are not used for modeling.

### Annual Sales Trend

| Year | Total sales (10k units) | Series covered |
|------|------------------------|----------------|
| 2022 | 1,506 | 1,017 |
| 2023 | 1,609 | 1,017 |
| 2024 | 1,656 | 1,017 |
| 2025 | 1,675 | 1,017 |
| 2026 (Jan–Jun) | 583 | 1,017 |

**Trend:** Across the sample (pcauto-covered series), annual sales grew steadily from 2022 to 2025, peaking at 16.75 million units in 2025; the NEV share (BEV + PHEV + EREV + HEV) now exceeds that of ICE vehicles.

### Monthly Sales Quantiles (non-zero valid months)

| Percentile | Monthly sales | Interpretation |
|-----------|---------------|---------------|
| **P10** | 68 units | Niche cars, sell only dozens per month |
| P25 | 360 units | Small-volume models |
| **P50 (median)** | 1,654 units | Half of valid models sell under ~2k/month |
| P75 | 5,586 units | Popular-model threshold |
| **P90** | 12,340 units | Selling >10k/month puts you in the top 10% |
| P95 | 17,123 units | Top 5% evergreen models |
| **P99** | 31,377 units | "Halo" models |

> Note: `monthly_sales` contains 70.6% zeros (suspension / pre-launch gaps); the table above is based on non-zero valid months (16,166 rows). Zero-inflation is handled in Leg B forecasting via a `log1p` target / two-stage (hurdle) approach.

### Brand Landscape (cumulative 2022–2026)

| Rank | Brand | Total sales (10k units) | Note |
|------|-------|------------------------|------|
| 1 | **BYD** | **1,165** | Dominant NEV leader |
| 2 | Volkswagen | 888 | Legacy ICE leader |
| 3 | Toyota | 666 | Steady |
| 4 | Honda | 402 | |
| 5 | Changan | 291 | Top domestic brand |
| 6 | Nissan | 282 | |
| 7 | Geely | 279 | 2nd domestic brand |
| 8 | BMW | 270 | Top luxury brand |
| 9 | Audi | 263 | |
| 10 | Mercedes-Benz | 229 | |
| 11 | Wuling | 203 | Micro-EV king |
| 12 | Chery | 191 | |

### Top 10 Series by Cumulative Sales

| Series | Brand | Cumulative (10k units) |
|--------|-------|------------------------|
| 秦PLUS (Qin PLUS) | BYD | 168 |
| 轩逸 (Sylphy) | Nissan | 156 |
| 宋PLUS新能源 (Song PLUS NEV) | BYD | 147 |
| 宏光MINIEV (Hongguang MINIEV) | Wuling | 142 |
| 朗逸 (Lavida) | Volkswagen | 140 |
| 速腾 (Sagitar) | Volkswagen | 109 |
| 海鸥 (Seagull) | BYD | 106 |
| 长安CS75 PLUS (CS75 PLUS) | Changan | 97 |
| 凯美瑞 (Camry) | Toyota | 92 |
| 元PLUS (Yuan PLUS) | BYD | 91 |

---

## 2. `feature.csv` — Vehicle Configurations

### Overview

| Item | Value |
|------|-------|
| Records | 2,084 rows (766 series × multiple model years) |
| Series | 766 |
| Brands | 126 |
| Feature columns | 84 |
| Time range | Model years 2015–2027 (annual sales cover 2022–2026) |
| Granularity | **series × year**, unique key `(series_name, year)` |
| Annual sales | `annual_sales` column (present for 760 / 766 series, 99.2% coverage) |
| Source | 汽车之家 (autohome), 太平洋汽车 (pcauto) & other public platforms |

> **Granularity note:** Unlike the legacy `vehicles.csv`, each row of `feature.csv` is a configuration snapshot of *one series in one model year* (one representative trim per series-year), not a trim-level breakdown. Configs update with model year (reflecting generation/facelift changes) but should not be used for "multi-trim supply mix" style analysis.

**Energy-type distribution (deduplicated by series):**

| Type | Series | Share |
|------|--------|-------|
| ICE | 337 | 44.0% |
| BEV | 249 | 32.5% |
| PHEV | 127 | 16.6% |
| EREV | 50 | 6.5% |
| HEV | 3 | 0.4% |

**Top 5 vehicle classes (deduplicated by series):** Compact SUV (153) > Mid-size SUV (119) > Compact car (98) > Mid-large SUV (79) > Mid-size car (73)

**Price distribution (official guide price, deduplicated by series):** median 15.48, mean 18.9, range 3.0 ~ 156.0 (10k CNY).

### Missing Values

Core spec fields (`official_price_wan`, `energy_type`, `vehicle_class`, `manufacturer`, `year`, dimensions) are **100% complete**. Missingness is confined to conditional fields:

- BEV / EREV models → no engine parameters (related columns NaN)
- ICE models → no motor / battery parameters (related columns NaN)
- Battery fields are heavily missing: `battery_capacity_kwh` 81.1% / `battery_type` 80.2% / `battery_warranty` 77.4% / `battery_range_km` 49.1% (conditional missingness, handled as needed during modeling)

### Key Column Reference

#### Basic specs (4 columns, no missing)

| Column | Description | Range |
|--------|-------------|-------|
| `official_price_wan` | Guide price (10k CNY) | 3.0 ~ 156.0, median 15.48 |
| `energy_type` | Energy type | ICE / BEV / PHEV / EREV / HEV |
| `vehicle_class` | Vehicle class | Compact SUV / Mid-size car / MPV, etc. (18 classes) |
| `manufacturer` | Maker | — |

#### Engine params (ICE / PHEV present, BEV / EREV NaN)

| Column | Description | Typical value |
|--------|-------------|---------------|
| `engine_displacement_l` | Displacement (L) | 1.0 ~ 4.0, mean 1.7 |
| `engine_max_horsepower_ps` | Max horsepower (PS) | mean 173, range 72 ~ 544 |
| `engine_max_torque_nm` | Max torque (N·m) | mean 254, range 93 ~ 725 |
| `engine_intake_type` | Intake type | Turbo / NA / Twin-turbo |
| `fuel_form` | Fuel form | Gasoline / BEV / PHEV |
| `gearbox_type` | Gearbox type | Fixed-ratio / Wet DCT / AT / Manual |

#### Motor params (BEV / hybrid present, ICE NaN)

| Column | Description | Typical value |
|--------|-------------|---------------|
| `motor_total_power_kw` | Total motor power (kW) | mean 171, max 880 |
| `motor_total_torque_nm` | Total motor torque (N·m) | mean 334, max 1,520 |
| `motor_front_power_kw` | Front motor power (kW) | FWD-dominated |
| `motor_rear_power_kw` | Rear motor power (kW) | Key for AWD / performance |

#### Battery params (BEV / hybrid partial, heavily missing)

| Column | Description | Note |
|--------|-------------|------|
| `battery_range_km` | Pure-electric range (km) | mean 362, ~51% coverage |
| `battery_capacity_kwh` | Battery capacity (kWh) | ⚠️ 81.1% missing, use with caution |
| `battery_type` | Battery type | ⚠️ 80.2% missing |
| `battery_warranty` | Battery warranty | ⚠️ 77.4% missing |

#### Body dimensions (no missing)

| Column | Description | Mean |
|--------|-------------|------|
| `length_mm` | Length | 4,723 mm |
| `width_mm` | Width | 1,865 mm |
| `height_mm` | Height | 1,628 mm |
| `wheelbase_mm` | **Wheelbase** (space core metric) | 2,813 mm |
| `curb_weight_kg` | Curb weight | 1,735 kg |
| `body_structure` | Body structure | Sedan / SUV / MPV |
| `acceleration_0_100_s` | 0–100 km/h (s) | mean 7.8, range 3.1 ~ 20.0 |

#### Safety config (no missing)

| Column | Description |
|--------|-------------|
| `driver_airbag` | Driver airbag |
| `side_airbag` | Side / curtain airbag |
| `knee_airbag` | Knee airbag (high-end, not standard) |

#### Comfort & tech config (no missing)

| Column | Description |
|--------|-------------|
| `center_screen` | Center display |
| `seat_material` | Seat material (synthetic / leather / fabric) |
| `sound_brand` | Audio brand (Harman Kardon / BOSE, etc.) |
| `seat_heating` | Seat heating |
| `seat_massage` | Seat massage |
| `seat_ventilation` | Seat ventilation |
| `aircon_control` | A/C control (auto / manual) |

> Note: `speaker_count` carries anomalous filler values in the source table and has been dropped from features.

---

## 3. Alignment & Modeling Subsets

Align `monthly_sales` (1,017 series) and `feature` (766 series) directly by `series_name`:

- **371 in-population series (Leg B):** have both a monthly sales panel and configs; enter Leg B monthly forecasting (continuous coverage 2022-01~2026-06).
- **736 series with annual sales (Leg A):** `feature.csv` rows with `annual_sales` falling in the 2022–2026 window; enter Leg A config→sales cross-sectional attribution.
- **646 sales-only series:** present in `monthly_sales` but lacking `feature` configs; excluded from the modeling population (to avoid diluting metrics with out-of-population series).

```
monthly_sales(1017) ─┐
                     ├─ intersection 371 ──> Leg B monthly forecasting
feature(766) ────────┘
feature series with annual_sales (736) ──> Leg A annual attribution
monthly_sales series without feature (646) ──> excluded
```

**Impact:** The 646 sales-only series are mostly long-tail / suspended models; excluding them brought Leg B volume-weighted WMAPE from ~51% down to 26.9% (the old figure was inflated by orphan series), confirming the correctness of "do not pad with out-of-population series".

---

## 4. Data Usage Guide

### For Leg B monthly forecasting (ARIMA / Prophet / XGBoost / LSTM)

1. Read `data/processed_new/splits/` (absolute-time split: `train` 2022-01~2025-06 / `val` 2025-07~2025-12 / `test` 2026-01~2026-06).
2. On the 371 in-population series, join `feature` by `series_name` (config taken at `(series_name, year)` ≤ row's model year, to prevent future-config leakage).
3. Target `monthly_sales` → `log1p` for zero-inflation; historical lag features dominate, config features are secondary.
4. Metrics: WMAPE (volume-weighted + per-series median, dual口径) + feature ablation + 90% prediction interval.

### For Leg A config attribution (XGBoost cross-section)

1. Take the 736 series in `feature.csv` with non-null `annual_sales` (~2,007 series×year rows).
2. Target `y = log1p(annual_sales)`; `GroupKFold(5) by series_name` (config is near-constant within a series across years, so grouping by series is required to prevent leakage).
3. Features = configs (price / size / energy / horsepower / battery…) + brand + year.
4. Note: configs explain **between-series** differences, not within-series year-over-year moves (see Stage 4 conclusion).

### For sentiment impact-factor regression (Phase B, pending)

1. Align `sentiment_summary.csv` to `monthly_sales` / `feature` by `series_name` (the new pipeline uses `series_name` uniformly, dropping the legacy `series_id` bridge).
2. Intersect sentiment with the 371 / 736 series first, then feed it as an exogenous variable into the Leg B / Leg A framework.
3. Note: legacy sentiment covers 490 dongchedi series under a different ID system; Phase B must re-align by `series_name` to the new 371 / 736 series scope.

---

## 5. User Sentiment (Phase B, Pending)

> **Stage note:** The sentiment data below come from an earlier dongchedi collection (490 series, completed 2026-07) and are **Phase B (pending new-data integration)** material. The new pipeline has switched to the "pcauto sales + autohome/pcauto configs" dual source aligned by `series_name`; Phase B will re-align sentiment by `series_name` to the new 371 / 736 series scope before integration. This section is retained as the Phase B reference.

**Source**: dongchedi public review API (`dongchedi.com/motor/pc/car/series/get_review_list`), plain `requests` crawler. Script `scripts/01_crawl_reviews.py`.

### File List

| File | Records | Time range | Description |
|------|---------|-----------|-------------|
| `data/sentiment/sentiment_reviews.csv` | 40,054 rows (490 series) | 2019-06 ~ 2026-07 | Review details |
| `data/sentiment/sentiment_summary.csv` | 490 series | — | Series-level sentiment aggregates |
| Chapter 6 of this doc | — | — | Coverage & quality report (merged here) |

> Regenerate aggregates / quality report: `python scripts/03_build_sentiment_summary.py` (reads `sentiment_reviews.csv`, writes `sentiment_summary.csv` and `data_quality_report.json` / `.md` under `data/sentiment/`, reproducible).

### Collection Scope (full crawl completed 2026-07-10)

- **Target:** all integer-ID series in the sales table (dongchedi API accepts integer IDs only), 502 in total
- **Completed:** **490 series / 40,054 rows**, covering **95 brands** (12 target series had no review data on the platform, marked empty)
- Review time span ~7 years (2019-06 ~ 2026-07), enabling long-term sentiment trend analysis

**Top 20 brands by coverage (by review count):**

| Brand | Series | Reviews | Avg rating |
|-------|--------|---------|------------|
| Volkswagen | 23 | 4,242 | 4.10 |
| Toyota | 18 | 1,548 | 4.11 |
| Honda | 16 | 1,461 | 3.99 |
| BYD | 19 | 1,444 | 4.09 |
| Audi | 17 | 1,425 | 4.18 |
| Changan | 14 | 1,264 | 4.14 |
| Hongqi | 17 | 1,169 | 4.27 |
| Mercedes-Benz | 13 | 1,120 | 4.15 |
| Chery | 11 | 915 | 4.04 |
| Xingtu | 10 | 884 | 4.32 |
| Leapmotor | 9 | 854 | 4.27 |
| GAC Trumpchi | 9 | 823 | 4.22 |
| Geely | 9 | 817 | 4.26 |
| Lynk | 8 | 800 | 4.32 |
| Cadillac | 10 | 787 | 4.24 |
| Chery Fengyun | 9 | 757 | 4.20 |
| Buick | 8 | 739 | 4.07 |
| Aion | 8 | 728 | 4.24 |
| Nissan | 8 | 725 | 4.06 |
| BMW | 9 | 713 | 4.24 |

> Full 95-brand detail in Chapter 6, "Sentiment Data Quality & Coverage Report".

### Data Quality

| Metric | Value |
|--------|-------|
| Duplicate `review_id` | 558 (1.39%) |
| Missing overall rating | 259 (0.65%) |
| Empty content | 0 |
| Series with no rating | 0 |

→ Good quality; duplicates / missing are negligible; `drop_duplicates(subset='review_id')` before analysis.

### Sentiment Polarity (scored subset, 39,795 rows)

- Positive (≥4.5): 11,953 (30.0%)
- Neutral (3.5–4.5): 24,696 (62.1%)
- Negative (<3.5): 3,146 (7.9%)

> Note: dongchedi reviews skew to 4–5 stars, so high neutral share is normal; text-level NLP sentiment complements star ratings to catch implicit negatives ("high score, low praise").

### `sentiment_reviews.csv` columns

| Column | Description |
|--------|-------------|
| `series_id` / `series_name` | Series join key + name |
| `review_id` | Unique review ID (for dedup) |
| `platform` | Source platform (dongchedi) |
| `user_nickname` / `user_id` | User nickname / ID |
| `publish_time` | Publish time |
| `content` / `content_len` | Review text / length |
| `rating_overall` | Overall rating (5-point) |
| `rating_appearance` ~ `rating_config` | 8-dimension ratings (appearance / space / interior / power / handling / comfort / fuel / config) |
| `digg_count` / `comment_count` | Likes / comments |
| `car_model` / `buy_location` / `buy_price` / `buy_time` / `fuel_type` / `consumption` | Purchase info |

> ⚠️ `sentiment_reviews.csv` has **no brand column**. Brand must be back-filled via `series_name` join to `monthly_sales.csv` (`brand`) or `feature.csv` (`brand_name`), see `attach_brand()` in `scripts/03_build_sentiment_summary.py`.

### `sentiment_summary.csv` columns

Series-level aggregates, one row per series:

- `review_count` / `avg_rating` / `median_rating` / `min_rating` / `max_rating`
- `avg_content_len` / `total_digg` / `total_comment`
- `earliest_review` / `latest_review`
- `positive_cnt` / `neutral_cnt` / `negative_cnt` + corresponding `_ratio`
- `avg_rating_appearance` … `avg_rating_config` (8-dimension means)

### Relationship to the other two tables

```
feature ──series_name──┐
                       ├──> sentiment_reviews / sentiment_summary
monthly_sales ──series_name──┘
```

- Sentiment × configs → explains "what kind of cars have good word-of-mouth"
- Sentiment × monthly sales → impact-factor regression (does sentiment drive sales, Phase B)

---

## 6. Sentiment Data Quality & Coverage Report (full, Phase B, Pending)

> The original generation logic is in `scripts/03_build_sentiment_summary.py`; re-running writes `data_quality_report.json` and `data_quality_report.md` under `data/sentiment/` (reproducible). Snapshot merged into this doc (Phase B reference).

- Total reviews: **40,054**
- Series covered: **490**
- Brands covered: **95**
- Review time range: 2019-06-24 00:04 ~ 2026-07-10 18:32

### Data Quality

- Duplicate review_id: 558 (1.39%)
- Missing overall rating: 259 (0.65%)
- Empty content: 0
- Series with no rating: 0

### Sentiment Polarity (scored subset)

- Positive (≥4.5): 11953 (30.0%)
- Neutral (3.5–4.5): 24696
- Negative (<3.5): 3146 (7.9%)

### Brand Coverage (full 95 brands)

| Brand | Series | Reviews | Avg rating |
|-------|--------|---------|------------|
| Volkswagen | 23 | 4,242 | 4.10 |
| Toyota | 18 | 1,548 | 4.11 |
| Honda | 16 | 1,461 | 3.99 |
| BYD | 19 | 1,444 | 4.09 |
| Audi | 17 | 1,425 | 4.18 |
| Changan | 14 | 1,264 | 4.14 |
| Hongqi | 17 | 1,169 | 4.27 |
| Mercedes-Benz | 13 | 1,120 | 4.15 |
| Chery | 11 | 915 | 4.04 |
| Xingtu | 10 | 884 | 4.32 |
| Leapmotor | 9 | 854 | 4.27 |
| GAC Trumpchi | 9 | 823 | 4.22 |
| Geely | 9 | 817 | 4.26 |
| Lynk | 8 | 800 | 4.32 |
| Cadillac | 10 | 787 | 4.24 |
| Chery Fengyun | 9 | 757 | 4.20 |
| Buick | 8 | 739 | 4.07 |
| Aion | 8 | 728 | 4.24 |
| Nissan | 8 | 725 | 4.06 |
| BMW | 9 | 713 | 4.24 |
| Deepal | 7 | 694 | 4.24 |
| Changan Nevo | 7 | 652 | 4.09 |
| NIO | 9 | 649 | 4.42 |
| Kia | 10 | 610 | 3.96 |
| Mazda | 6 | 600 | 4.26 |
| Ford | 8 | 591 | 4.20 |
| Geely Galaxy | 8 | 575 | 4.23 |
| Li Auto | 6 | 541 | 4.57 |
| Skoda | 5 | 500 | 3.87 |
| Haval | 5 | 500 | 4.28 |
| IM Motors | 6 | 499 | 4.44 |
| Qichen | 5 | 482 | 3.91 |
| Jetour | 5 | 474 | 4.06 |
| Roewe | 6 | 457 | 4.12 |
| Volvo | 6 | 436 | 4.29 |
| Chevrolet | 5 | 431 | 4.02 |
| Hyundai | 5 | 404 | 4.12 |
| Lincoln | 4 | 400 | 4.37 |
| Peugeot | 4 | 400 | 4.20 |
| Jetta | 5 | 332 | 3.86 |
| Dongfeng Aeolus | 4 | 326 | 3.98 |
| Dongfeng Forthing | 6 | 311 | 3.88 |
| Fang Cheng Bao | 3 | 300 | 4.47 |
| Bestune | 5 | 287 | 3.86 |
| Dongfeng Fengguang | 5 | 263 | 3.61 |
| Onvo | 3 | 239 | 4.45 |
| ORA | 3 | 234 | 4.13 |
| Tank | 3 | 229 | 4.35 |
| smart | 3 | 207 | 4.23 |
| Landian | 2 | 200 | 4.06 |
| Citroen | 2 | 200 | 4.22 |
| Tesla | 2 | 200 | 4.24 |
| Feifan | 2 | 200 | 4.27 |
| Infiniti | 2 | 200 | 4.14 |
| ARCFOX | 2 | 200 | 4.26 |
| Xiaomi Auto | 2 | 200 | 4.53 |
| Wuling | 6 | 191 | 3.59 |
| Maxus | 5 | 187 | 3.97 |
| Polestar | 3 | 171 | 4.06 |
| Geometry | 3 | 158 | 3.90 |
| Cowin | 3 | 153 | 3.80 |
| Jaguar | 2 | 145 | 4.25 |
| Mengshi | 2 | 140 | 4.38 |
| BAIC | 2 | 122 | 4.08 |
| Yangwang | 2 | 109 | 4.50 |
| Sihao | 3 | 106 | 3.70 |
| Zongheng | 1 | 100 | 4.29 |
| Land Rover | 1 | 100 | 4.07 |
| Denza | 1 | 100 | 4.38 |
| Changan Oushang | 1 | 100 | 4.15 |
| iCAR | 1 | 100 | 4.47 |
| firefly | 1 | 95 | 4.30 |
| Audi AUDI | 2 | 93 | 4.42 |
| Huajing | 1 | 90 | 4.58 |
| Jiangling NEV | 4 | 70 | 3.84 |
| Ruipee | 5 | 70 | 4.10 |
| Jianghuai Yiwei | 2 | 62 | 3.94 |
| Jianghuai Ruifeng | 2 | 55 | 3.74 |
| Shijie | 1 | 43 | 3.96 |
| Unknown | 1 | 42 | 4.51 |
| Changan Kaicheng | 2 | 36 | 3.50 |
| MINI | 2 | 33 | 4.14 |
| Haima | 1 | 32 | 4.00 |
| Dongfeng Fengdu | 1 | 23 | 4.13 |
| Zhidou | 1 | 22 | 4.15 |
| Lingbao | 2 | 15 | 3.08 |
| Caocao | 1 | 9 | 3.82 |
| Ruichi | 1 | 9 | 3.72 |
| Dayun | 2 | 9 | 3.59 |
| Skyworth Auto | 1 | 8 | 3.75 |
| Aishang | 1 | 5 | 4.36 |
| Xiaohu | 1 | 5 | 3.42 |
| Foton | 1 | 4 | 2.72 |
| Lingxi | 1 | 3 | 3.83 |
| Dongfeng Fukang | 1 | 1 | 3.50 |

---

## 7. Stage 6 Dashboard Data Bridge

The Stage 6 web dashboard (`app/`) does not produce raw data itself; it reads the CSVs described above and pre-bakes them into a JSON data bridge via `app/build_dashboard_data.py` for direct ECharts rendering:

- Input: products under `data/processed_new/stage3/`, `data/processed_new/stage4/`, and `data/sentiment/`.
- Output: `app/static/data/*.json` (overview, forecast, absa, attribution, relation, alerts, drilldown).
- Run: `python app/build_dashboard_data.py` to regenerate; `python app/app.py` to launch.

> The dashboard data bridge is currently based on an initial-pipeline snapshot; after Phase B integration it will be refreshed to the new 371 / 736 series scope.
