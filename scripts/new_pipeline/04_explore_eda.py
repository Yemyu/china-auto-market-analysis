# -*- coding: utf-8 -*-
"""
Stage 2: Data Filtering & Exploratory Visualization

Inputs (prepared in Stage 1):
  data/raw/sales.csv                         - (legacy) monthly sales (33,845 rows / 1,122 series)
  data/raw/vehicles.csv                      - vehicle specs (1,139 series x 92 cols)
  data/raw/monthly_sales.csv      - NEW monthly sales (54,918 rows / 1,017 series,
                                               2022-2026, 太平洋汽车)
  data/sentiment/analysis_input.csv          - sentiment universe (490 series); used as the
                                               series_name -> legacy series_id bridge so the new
                                               sales can join stage-4/5 (which key on legacy id).

NOTE: monthly sales are sourced from monthly_sales.csv (FULL 1,017 series, 2022-2026).
We keep every series with >=24 consecutive months (all 1,017 qualify) - NO restriction to the
sentiment universe, so the baseline models see the COMPLETE new sales. The native pacific
series_id is the primary key; legacy_series_id is carried where mappable (for Phase B sentiment
joins on the 261 overlapping series). See docs/车辆配置与销量对应方案.md.

Outputs:
  data/processed_new/sales_filtered_24m.csv   - series with >=24 consecutive months
  data/processed_new/timeseries_summary.csv   - full 1,122-series time-series summary
  figures_new/sales_trend.png                 - monthly sales trend
  figures_new/category_distribution.png       - category & vehicle class distribution
  figures_new/hardware_features.png           - hardware feature distributions
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _font_setup
from matplotlib.ticker import FuncFormatter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(BASE, 'data', 'raw')
PROC = os.path.join(BASE, 'data', 'processed_new')
FIG = os.path.join(BASE, 'figures_new')
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

MIN_RUN = 24

# Clean, publication-ready style (English-only to avoid font issues)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'savefig.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# Professional color palette
COLORS = {
    'blue': '#2E86AB',
    'orange': '#F18F01',
    'green': '#C73E1D',
    'purple': '#6A4C93',
    'teal': '#1B998B',
    'gray': '#8D99AE',
}

# Load data
# --- NEW sales source (2026-08, full coverage) ---
# monthly_sales.csv is the complete new monthly sales (1,017 series, 2022-2026).
# We keep ALL series with >=24 consecutive months (every series qualifies) and do NOT
# restrict to the sentiment universe, so baseline models see the full new sales.
# Native pacific series_id is the primary key; legacy_series_id carried where mappable.
vehicles_path = os.path.join(RAW, 'vehicles.csv')
if not os.path.exists(vehicles_path):
    vehicles_path = os.path.join(RAW, 'version1', 'vehicles.csv')
vehicles = pd.read_csv(vehicles_path)

new_sales = pd.read_csv(os.path.join(RAW, 'monthly_sales.csv'))
new_sales['series_name'] = new_sales['series_name'].astype(str)
sent_universe = pd.read_csv(os.path.join(BASE, 'data', 'sentiment', 'analysis_input.csv'))
bridge = (sent_universe.drop_duplicates('series_name')
          .set_index('series_name')['series_id'].astype(str))
sales = new_sales.copy()
sales['legacy_series_id'] = sales['series_name'].map(bridge)  # NaN for the 756 new series
print(f'[04] FULL all_sales kept: {sales["series_id"].nunique()} series '
      f'(legacy id mappable for {sales["legacy_series_id"].notna().sum()})')

sales['period'] = sales['year'] * 12 + (sales['month'] - 1)
sales['date'] = pd.to_datetime(dict(year=sales.year, month=sales.month, day=1))
sales['data_source'] = 'pcauto'

# English labels for Chinese categorical values (avoids font encoding issues in charts)
CATEGORY_MAP = {'SUV': 'SUV', '轿车': 'Sedan', 'MPV': 'MPV'}
VEHICLE_CLASS_MAP = {
    '中型车': 'Mid-size Sedan', '中大型车': 'Large Sedan', '中型SUV': 'Mid-size SUV',
    '紧凑型SUV': 'Compact SUV', '紧凑型车': 'Compact Sedan', '中大型SUV': 'Large SUV',
    '小型SUV': 'Small SUV', '大型SUV': 'Full-size SUV', '中大型MPV': 'Large MPV',
    '小型车': 'Small Sedan', '微型车': 'Mini Car', '中型MPV': 'Mid-size MPV',
    '紧凑型MPV': 'Compact MPV', '大型车': 'Full-size Sedan', '大型MPV': 'Full-size MPV',
    '微面': 'Mini Van', '轻客': 'Light Van', 'MPV': 'MPV',
}
ENERGY_TYPE_MAP = {
    '燃油': 'Gasoline', '纯电动': 'BEV', '插电混动': 'PHEV', '增程式': 'EREV',
    '油电混动': 'HEV', '插混+纯电': 'PHEV+BEV', '其他': 'Other',
}
sales['category_en'] = sales['category'].map(CATEGORY_MAP)
vehicles['vehicle_class_en'] = vehicles['vehicle_class'].map(VEHICLE_CLASS_MAP)
vehicles['energy_type_en'] = vehicles['energy_type'].map(ENERGY_TYPE_MAP)


def runs_info(periods):
    """Return (longest_run, interrupt_count, longest_gap, total_months)."""
    p = np.sort(np.unique(periods))
    if len(p) == 0:
        return 0, 0, 0, 0
    diffs = np.diff(p)
    runs = []
    gaps = []
    cur = 1
    for d in diffs:
        if d == 1:
            cur += 1
        else:
            runs.append(cur)
            gaps.append(d - 1)
            cur = 1
    runs.append(cur)
    longest = int(max(runs))
    n_interrupt = int(np.sum(diffs > 1))
    longest_gap = int(max(gaps)) if gaps else 0
    return longest, n_interrupt, longest_gap, len(p)


# full time-series summary
print('Computing full time-series summary ...')
summary_rows = []
for sid, g in sales.groupby('series_id'):
    longest, nint, gap, total = runs_info(g['period'].values)
    summary_rows.append({
        'series_id': sid,
        'series_name': g['series_name'].iloc[0],
        'brand': g['brand'].iloc[0],
        'category': g['category'].iloc[0],
        'total_months': total,
        'longest_run_months': longest,
        'interrupt_count': nint,
        'longest_gap_months': gap,
        'first_year': int(g['year'].min()),
        'last_year': int(g['year'].max()),
    })
summary = pd.DataFrame(summary_rows).sort_values('longest_run_months', ascending=False)
summary.to_csv(os.path.join(PROC, 'timeseries_summary.csv'), index=False, encoding='utf-8-sig')
print(f'          Total series: {len(summary)}; written to timeseries_summary.csv')

# filter series with >=24 consecutive months
qualified = summary[summary['longest_run_months'] >= MIN_RUN]['series_id'].tolist()
print(f'Series with >= {MIN_RUN} consecutive months: {len(qualified)}')
filt = sales[sales['series_id'].isin(qualified)].copy()
filt = filt[['year', 'month', 'series_id', 'legacy_series_id', 'series_name', 'brand', 'category',
             'monthly_sales', 'data_source', 'period', 'date', 'category_en']]
filt.to_csv(os.path.join(PROC, 'sales_filtered_24m.csv'), index=False, encoding='utf-8-sig')
print(f'          Filtered dataset: {len(filt)} rows; written to sales_filtered_24m.csv')

legacy_ids = sales.dropna(subset=['legacy_series_id'])['legacy_series_id'].astype(int).unique()
veh_filt = vehicles[vehicles['series_id'].isin(legacy_ids)].drop_duplicates('series_id')

# Sales trend chart
print('Sales trend chart ...')
mt = filt.groupby('date')['monthly_sales'].sum().reset_index()
mt = mt.sort_values('date')
mt['rolling_12m'] = mt['monthly_sales'].rolling(window=12, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.fill_between(mt['date'], mt['monthly_sales'], color=COLORS['blue'], alpha=0.12)
ax.plot(mt['date'], mt['monthly_sales'], color=COLORS['blue'], linewidth=2, label='Monthly sales')
ax.plot(mt['date'], mt['rolling_12m'], color=COLORS['orange'], linewidth=2, label='12-month moving average')
ax.set_title('Monthly Sales Trend of Mature Models (>=24 Consecutive Months)')
ax.set_xlabel('Month')
ax.set_ylabel('Total Sales (units)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
ax.legend(loc='upper left', frameon=False)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'sales_trend.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# Category distribution chart
print('Category distribution chart ...')
cat = filt.groupby('category_en')['series_id'].nunique().sort_values(ascending=False)
vclass = veh_filt['vehicle_class_en'].value_counts().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: sales category
bars1 = axes[0].bar(cat.index, cat.values, color=[COLORS['blue'], COLORS['teal'], COLORS['purple']])
axes[0].set_title('Sales Category Distribution (Number of Series)')
axes[0].set_ylabel('Number of Series')
axes[0].set_xlabel('Category')
for bar in bars1:
    height = bar.get_height()
    axes[0].annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 4),
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

# Right: vehicle class
x_pos = np.arange(len(vclass))
bars2 = axes[1].bar(x_pos, vclass.values, color=COLORS['orange'])
axes[1].set_title('Vehicle Class Distribution (Number of Series)')
axes[1].set_ylabel('Number of Series')
axes[1].set_xlabel('Vehicle Class')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(vclass.index, rotation=30, ha='right', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    axes[1].annotate(f'{int(height)}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4),
                     textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(FIG, 'category_distribution.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# Hardware feature distributions
print('Hardware feature distributions ...')
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Price
price = veh_filt['official_price_wan'].dropna()
axes[0, 0].hist(price, bins=45, color=COLORS['blue'], edgecolor='white', alpha=0.85)
axes[0, 0].axvline(price.median(), color=COLORS['orange'], linestyle='--', linewidth=2, label=f'Median: {price.median():.1f}')
axes[0, 0].set_title('Official Price Distribution')
axes[0, 0].set_xlabel('Price (10k CNY)')
axes[0, 0].set_ylabel('Number of Series')
axes[0, 0].legend(loc='upper right', frameon=False)

# Energy type
et = veh_filt['energy_type_en'].value_counts().head(8)
axes[0, 1].barh(np.arange(len(et)), et.values, color=COLORS['teal'])
axes[0, 1].set_yticks(np.arange(len(et)))
axes[0, 1].set_yticklabels(et.index, fontsize=10)
axes[0, 1].invert_yaxis()
axes[0, 1].set_title('Energy Type Distribution (Top 8)')
axes[0, 1].set_xlabel('Number of Series')
for i, v in enumerate(et.values):
    axes[0, 1].text(v + 2, i, f'{int(v)}', va='center', fontsize=10, fontweight='bold')

# Range
rng = veh_filt['battery_range_km'].dropna()
axes[1, 0].hist(rng, bins=40, color=COLORS['purple'], edgecolor='white', alpha=0.85)
axes[1, 0].axvline(rng.median(), color=COLORS['orange'], linestyle='--', linewidth=2, label=f'Median: {rng.median():.0f} km')
axes[1, 0].set_title('BEV / PHEV Range Distribution')
axes[1, 0].set_xlabel('Range (km)')
axes[1, 0].set_ylabel('Number of Series')
axes[1, 0].legend(loc='upper right', frameon=False)

# Acceleration
acc = veh_filt['acceleration_0_100_s'].dropna()
axes[1, 1].hist(acc, bins=40, color=COLORS['green'], edgecolor='white', alpha=0.85)
axes[1, 1].axvline(acc.median(), color=COLORS['orange'], linestyle='--', linewidth=2, label=f'Median: {acc.median():.1f} s')
axes[1, 1].set_title('0-100 km/h Acceleration Distribution')
axes[1, 1].set_xlabel('Acceleration (seconds)')
axes[1, 1].set_ylabel('Number of Series')
axes[1, 1].legend(loc='upper right', frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(FIG, 'hardware_features.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print('\n[DONE] Stage 2 outputs:')
print('  data/processed_new/sales_filtered_24m.csv')
print('  data/processed_new/timeseries_summary.csv')
print('  figures_new/sales_trend.png')
print('  figures_new/category_distribution.png')
print('  figures_new/hardware_features.png')
