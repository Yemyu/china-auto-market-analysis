# -*- coding: utf-8 -*-
"""Filter the monthly sales panel and produce exploratory summaries."""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _font_setup
from _sales_repair import apply_verified_sales_corrections
from matplotlib.ticker import FuncFormatter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'data', 'raw')
PROC = os.path.join(BASE, 'data', 'processed')
FIG = os.path.join(BASE, 'assets/analysis')
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

MIN_RUN = 24

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

COLORS = {
    'blue': '#2E86AB',
    'orange': '#F18F01',
    'green': '#C73E1D',
    'purple': '#6A4C93',
    'teal': '#1B998B',
    'gray': '#8D99AE',
}

# Use the latest available configuration row only for descriptive plots.
# Forecast joins remain year-aware.
features = pd.read_csv(os.path.join(RAW, 'feature.csv'), low_memory=False)
features['year'] = pd.to_numeric(features['year'], errors='coerce')
vehicles = (features.sort_values(['series_name', 'year'])
            .drop_duplicates('series_name', keep='last')
            .copy())

sales = pd.read_csv(os.path.join(RAW, 'monthly_sales.csv'))
sales['series_name'] = sales['series_name'].astype(str)
sales, repair_audit = apply_verified_sales_corrections(sales)
print(f'[04] monthly panel: {sales["series_id"].nunique()} series')
print(f'[04] verified overlay: {len(repair_audit)} rows / '
      f'{repair_audit["series_name"].nunique()} series / '
      f'{repair_audit["sales_delta"].sum():+,.0f} net sales')

sales['period'] = sales['year'] * 12 + (sales['month'] - 1)
sales['date'] = pd.to_datetime(dict(year=sales.year, month=sales.month, day=1))
sales['data_source'] = np.where(
    sales['sales_repair_applied'], 'pcauto_verified_overlay', 'pcauto'
)

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
filt = filt[['year', 'month', 'series_id', 'series_name', 'brand', 'category',
             'monthly_sales', 'data_source', 'period', 'date', 'category_en']]
filt.to_csv(os.path.join(PROC, 'sales_filtered_24m.csv'), index=False, encoding='utf-8-sig')
print(f'          Filtered dataset: {len(filt)} rows; written to sales_filtered_24m.csv')

qualified_names = set(filt['series_name'])
veh_filt = vehicles[vehicles['series_name'].isin(qualified_names)].copy()

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

print('\n[DONE] outputs:')
print('  data/processed/sales_filtered_24m.csv')
print('  data/processed/timeseries_summary.csv')
print('  assets/analysis/sales_trend.png')
print('  assets/analysis/category_distribution.png')
print('  assets/analysis/hardware_features.png')
