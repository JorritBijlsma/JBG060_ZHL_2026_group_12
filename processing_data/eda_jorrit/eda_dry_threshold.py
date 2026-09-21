"""
DRY SEASON THRESHOLD SENSITIVITY
Tests whether the NE DRC → SSD flood relationship is robust across different
definitions of "dry season".

Thresholds tested:
  - Absolute: 0.5, 1.0, 2.0, 5.0 mm/day
  - Percentile: 10th, 25th, 50th percentile of each region's rainfall
  - Rolling: 5-day and 10-day rolling mean below 1 mm/day

If NE DRC's signal persists across all definitions, the finding is robust.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r'C:\Users\20241060\OneDrive - TU Eindhoven\JBG060\JBG060_ZHL_2026_group_12\processing_data')

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from loading import load_rainfall_runoff, load_flood_masks

# ============================================================================
# SETUP
# ============================================================================
RESULTS_DIR = Path(r'C:\Users\20241060\OneDrive - TU Eindhoven\JBG060\JBG060_ZHL_2026_group_12\processing_data\results_eda_jorrit')
OUT_DIR = RESULTS_DIR / 'dry_threshold_sensitivity'
FIGS_DIR = OUT_DIR / 'figures'
TABLES_DIR = OUT_DIR / 'tables'
STATS_DIR = OUT_DIR / 'statistics'

for d in [FIGS_DIR, TABLES_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("="*90)
print("🎯 DRY SEASON THRESHOLD SENSITIVITY ANALYSIS")
print("="*90)

# ============================================================================
# REGIONS
# ============================================================================
REGIONS = {
    'South Sudan':   {'lat_min': 3.5,  'lat_max': 12.5, 'lon_min': 24.0, 'lon_max': 36.0},
    'Uganda':        {'lat_min': -1.5, 'lat_max': 3.5,  'lon_min': 29.5, 'lon_max': 35.0},
    'NE DRC':        {'lat_min': 2.0,  'lat_max': 5.0,  'lon_min': 24.0, 'lon_max': 30.0},
    'Ethiopia':      {'lat_min': 5.0,  'lat_max': 12.0, 'lon_min': 33.0, 'lon_max': 37.0},
    'West Kenya':    {'lat_min': -1.0, 'lat_max': 1.0,  'lon_min': 33.5, 'lon_max': 35.5},
    'Sudan central': {'lat_min': 12.5, 'lat_max': 16.0, 'lon_min': 30.0, 'lon_max': 34.0},
}

# ============================================================================
# 1. LOAD
# ============================================================================
print("\n[1] Loading ERA5 + floods...")

years = np.arange(2000, 2025)
era5 = load_rainfall_runoff(years)
era5_tp_mm = era5['tp'] * 1000

regional_rain = {}
for name, box in REGIONS.items():
    lat_idx = (era5.latitude >= box['lat_min']) & (era5.latitude <= box['lat_max'])
    lon_idx = (era5.longitude >= box['lon_min']) & (era5.longitude <= box['lon_max'])
    region = era5_tp_mm.isel(latitude=lat_idx, longitude=lon_idx)
    series = region.mean(dim=['latitude', 'longitude']).to_series()
    series.name = name
    regional_rain[name] = series

rain_df = pd.DataFrame(regional_rain)
rain_df.index.name = 'date'
print(f"  ✓ Regional rainfall: {rain_df.shape}")

# Load floods
flood_bbox = {'lat_min': 3.0, 'lat_max': 13.0, 'lon_min': 24.0, 'lon_max': 36.0}
flood_df = load_flood_masks(years, bbox=flood_bbox)
flood_daily = flood_df.groupby('date').size().reset_index()
flood_daily.columns = ['date', 'flood_pixels']
flood_daily = flood_daily.set_index('date').sort_index()

print(f"  ✓ SSD floods: {len(flood_daily)} days")

# ============================================================================
# 2. DEFINE THRESHOLDS
# ============================================================================
print("\n[2] Defining thresholds...")

# Absolute thresholds
absolute_thresholds = {
    'abs_0.5mm': 0.5,
    'abs_1.0mm': 1.0,
    'abs_2.0mm': 2.0,
    'abs_5.0mm': 5.0,
}

# Rolling-mean thresholds
rolling_thresholds = {
    'roll5_1mm':  (5, 1.0),
    'roll10_1mm': (10, 1.0),
    'roll10_2mm': (10, 2.0),
}

# Percentile thresholds (computed per region)
percentile_thresholds = {
    'pct_10': 10,
    'pct_25': 25,
    'pct_50': 50,
}

# ============================================================================
# 3. FIND DRY SEASONS FOR EACH THRESHOLD
# ============================================================================
print("\n[3] Finding annual dry seasons per region × threshold...")

def find_longest_dry_streak(is_dry_series):
    """Return longest consecutive dry streak within a given boolean series."""
    streaks = []
    current = 0
    start = None
    for idx, dry in is_dry_series.items():
        if dry:
            if current == 0:
                start = idx
            current += 1
        else:
            if current > 0:
                streaks.append({
                    'start': start,
                    'end': idx - timedelta(days=1),
                    'duration': current,
                })
            current = 0
            start = None
    if current > 0:
        streaks.append({
            'start': start,
            'end': is_dry_series.index[-1],
            'duration': current,
        })
    if not streaks:
        return None
    return max(streaks, key=lambda x: x['duration'])


all_dry_seasons = []

for region in REGIONS.keys():
    rain_series = rain_df[region]

    # For each threshold type
    for thresh_name, thresh_val in absolute_thresholds.items():
        for year in range(2000, 2025):
            year_data = rain_series[rain_series.index.year == year]
            if len(year_data) == 0:
                continue
            is_dry = year_data < thresh_val
            if not is_dry.any():
                continue
            result = find_longest_dry_streak(is_dry)
            if result:
                all_dry_seasons.append({
                    'region': region,
                    'threshold': thresh_name,
                    'threshold_value': thresh_val,
                    'year': year,
                    'dry_duration': result['duration'],
                })

    # Rolling thresholds
    for thresh_name, (window, val) in rolling_thresholds.items():
        rolling = rain_series.rolling(window, min_periods=1).mean()
        for year in range(2000, 2025):
            year_data = rolling[rolling.index.year == year]
            if len(year_data) == 0:
                continue
            is_dry = year_data < val
            if not is_dry.any():
                continue
            result = find_longest_dry_streak(is_dry)
            if result:
                all_dry_seasons.append({
                    'region': region,
                    'threshold': thresh_name,
                    'threshold_value': val,
                    'year': year,
                    'dry_duration': result['duration'],
                })

    # Percentile thresholds
    for thresh_name, pct in percentile_thresholds.items():
        pval = np.percentile(rain_series.dropna(), pct)
        for year in range(2000, 2025):
            year_data = rain_series[rain_series.index.year == year]
            if len(year_data) == 0:
                continue
            is_dry = year_data < pval
            if not is_dry.any():
                continue
            result = find_longest_dry_streak(is_dry)
            if result:
                all_dry_seasons.append({
                    'region': region,
                    'threshold': thresh_name,
                    'threshold_value': pval,
                    'year': year,
                    'dry_duration': result['duration'],
                })

dry_df = pd.DataFrame(all_dry_seasons)
dry_df.to_csv(TABLES_DIR / 'dry_seasons_all_thresholds.csv', index=False)

print(f"  ✓ Computed {len(dry_df)} dry-season records across "
      f"{dry_df['threshold'].nunique()} thresholds")

# ============================================================================
# 4. CORRELATE WITH SSD FLOODS AT EACH THRESHOLD
# ============================================================================
print("\n[4] Correlating dry duration with SSD floods at each threshold...")

annual_flood = flood_daily.groupby(flood_daily.index.year).agg(
    flood_mean=('flood_pixels', 'mean'),
    flood_max=('flood_pixels', 'max'),
).reset_index().rename(columns={'date': 'year'})

corr_rows = []
for (region, thresh), group in dry_df.groupby(['region', 'threshold']):
    merged = group.merge(annual_flood, on='year', how='inner')
    if len(merged) < 5:
        continue

    if HAS_SCIPY:
        r, p = pearsonr(merged['dry_duration'].values, merged['flood_mean'].values)
        rs, ps = spearmanr(merged['dry_duration'].values, merged['flood_mean'].values)
    else:
        r = merged['dry_duration'].corr(merged['flood_mean'])
        rs = merged['dry_duration'].corr(merged['flood_mean'], method='spearman')
        p, ps = np.nan, np.nan

    corr_rows.append({
        'region': region,
        'threshold': thresh,
        'pearson_r': r,
        'pearson_p': p,
        'spearman_r': rs,
        'spearman_p': ps,
        'n_years': len(merged),
        'mean_duration': merged['dry_duration'].mean(),
    })

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(TABLES_DIR / 'threshold_sensitivity_correlations.csv', index=False)

# Print summary table
print(f"\n  {'Region':<15} {'Threshold':<15} {'r':>8} {'p':>10} {'mean_dur':>10}")
print("  " + "-"*65)
for _, row in corr_df.sort_values(['region', 'threshold']).iterrows():
    sig = "*" if (not np.isnan(row['pearson_p']) and row['pearson_p'] < 0.05) else ""
    print(f"  {row['region']:<15} {row['threshold']:<15} "
          f"{row['pearson_r']:>8.3f} {row['pearson_p']:>10.4f} "
          f"{row['mean_duration']:>10.1f} {sig}")

# ============================================================================
# 5. ROBUSTNESS SUMMARY
# ============================================================================
print("\n[5] Robustness summary — which regions show consistent signals?")

robustness = []
for region in REGIONS.keys():
    sub = corr_df[corr_df['region'] == region].dropna(subset=['pearson_r'])
    if len(sub) == 0:
        continue

    n_sig = (sub['pearson_p'] < 0.05).sum() if HAS_SCIPY else 0
    mean_r = sub['pearson_r'].mean()
    std_r = sub['pearson_r'].std()
    min_r = sub['pearson_r'].min()
    max_r = sub['pearson_r'].max()

    verdict = '✅ ROBUST' if n_sig >= len(sub) * 0.7 and mean_r > 0.3 else (
        '⚠️ PARTIAL' if n_sig >= 1 and mean_r > 0.2 else '❌ not robust')

    robustness.append({
        'region': region,
        'n_thresholds': len(sub),
        'n_significant': n_sig,
        'mean_r': mean_r,
        'std_r': std_r,
        'min_r': min_r,
        'max_r': max_r,
        'verdict': verdict,
    })

robust_df = pd.DataFrame(robustness)
robust_df.to_csv(TABLES_DIR / 'robustness_summary.csv', index=False)

print(f"\n  {'Region':<15} {'n_thresh':>10} {'n_sig':>8} "
      f"{'mean_r':>10} {'std_r':>10} {'verdict':<20}")
print("  " + "-"*75)
for _, row in robust_df.iterrows():
    print(f"  {row['region']:<15} {row['n_thresholds']:>10} "
          f"{row['n_significant']:>8} {row['mean_r']:>10.3f} "
          f"{row['std_r']:>10.3f} {row['verdict']:<20}")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
if HAS_PLT:
    print("\n[6] Creating visualizations...")

    try:
        # ---- Figure 1: Heatmap of correlations (region × threshold) ----
        fig, ax = plt.subplots(figsize=(14, 7))

        pivot = corr_df.pivot(index='region', columns='threshold',
                              values='pearson_r')

        # Order columns sensibly
        col_order = ['abs_0.5mm', 'abs_1.0mm', 'abs_2.0mm', 'abs_5.0mm',
                     'roll5_1mm', 'roll10_1mm', 'roll10_2mm',
                     'pct_10', 'pct_25', 'pct_50']
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

        im = ax.imshow(pivot.values, cmap='RdBu_r', vmin=-0.8, vmax=0.8, aspect='auto')

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Annotate cells with r-value
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color=color, fontsize=9, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Pearson r (dry duration → SSD floods)')
        ax.set_title('Threshold Sensitivity: Dry Duration → SSD Floods\n'
                     'How consistent is each region\'s signal?',
                     fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'threshold_heatmap.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: threshold_heatmap.png")

        # ---- Figure 2: r-values per region across thresholds ----
        fig, ax = plt.subplots(figsize=(14, 7))
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']
        for i, region in enumerate(REGIONS.keys()):
            sub = corr_df[corr_df['region'] == region].copy()
            if len(sub) == 0:
                continue
            sub = sub.reset_index(drop=True)
            x = np.arange(len(sub))
            ax.plot(x, sub['pearson_r'], marker=markers[i % len(markers)],
                    linewidth=2, markersize=8, label=region, alpha=0.8)

        ax.axhline(0.3, color='green', linestyle='--', alpha=0.4,
                   label='Strong threshold (r=0.3)')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub['threshold'], rotation=45, ha='right')
        ax.set_ylabel('Pearson r (dry duration → SSD floods)')
        ax.set_xlabel('Threshold definition')
        ax.set_title('Threshold Sensitivity: Which Regions Give Consistent Signals?',
                     fontweight='bold', fontsize=13)
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'threshold_robustness_plot.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: threshold_robustness_plot.png")

        # ---- Figure 3: NE DRC dry season duration across thresholds ----
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        # Panel A: dry duration time series for different thresholds
        ax = axes[0]
        ne_drc = dry_df[dry_df['region'] == 'NE DRC']
        for thresh in ne_drc['threshold'].unique():
            sub = ne_drc[ne_drc['threshold'] == thresh].sort_values('year')
            ax.plot(sub['year'], sub['dry_duration'], 'o-',
                    linewidth=1.5, markersize=6, label=thresh, alpha=0.8)
        ax.set_ylabel('NE DRC dry season duration (days)')
        ax.set_title('NE DRC Dry Season Duration Across Threshold Definitions',
                     fontweight='bold', fontsize=12)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(alpha=0.3)

        # Panel B: SSD floods (annual mean)
        ax = axes[1]
        ax.plot(annual_flood['year'], annual_flood['flood_mean'],
                'ro-', linewidth=2, markersize=8)
        ax.set_ylabel('SSD annual flood mean (pixels)')
        ax.set_xlabel('Year')
        ax.set_title('SSD Annual Flood Extent', fontweight='bold', fontsize=12)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'ne_drc_threshold_timeseries.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: ne_drc_threshold_timeseries.png")

    except Exception as e:
        print(f"  ⚠ Plot error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 7. REPORT
# ============================================================================
print("\n[7] Writing report...")

report = f"""
================================================================================
DRY SEASON THRESHOLD SENSITIVITY ANALYSIS
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

1. PURPOSE
--------------------------------------------------------------------------------
The "dry season" definition is arbitrary. This analysis tests whether the
NE DRC → SSD flood relationship is robust across 10 different definitions:

Absolute thresholds:  0.5, 1.0, 2.0, 5.0 mm/day
Rolling-mean:         5-day and 10-day rolling mean at 1-2 mm/day
Percentile:           10th, 25th, 50th percentile per region

A finding is "robust" only if it holds across multiple thresholds.

2. CORRELATIONS PER REGION × THRESHOLD
--------------------------------------------------------------------------------
"""
for region in REGIONS.keys():
    report += f"\n  {region}:\n"
    sub = corr_df[corr_df['region'] == region]
    for _, row in sub.iterrows():
        sig = "*" if (not np.isnan(row['pearson_p']) and row['pearson_p'] < 0.05) else ""
        report += (f"    {row['threshold']:<15}: r={row['pearson_r']:>7.3f} "
                   f"(p={row['pearson_p']:.3f}){sig} "
                   f"mean_dur={row['mean_duration']:.1f}d\n")

report += """
3. ROBUSTNESS VERDICT
--------------------------------------------------------------------------------
"""
for _, row in robust_df.iterrows():
    report += (f"  {row['region']:<15}: {row['n_significant']}/{row['n_thresholds']} "
               f"thresholds significant, mean r={row['mean_r']:.3f} "
               f"(± {row['std_r']:.3f})  {row['verdict']}\n")

report += """
4. INTERPRETATION
--------------------------------------------------------------------------------
Regions marked ✅ ROBUST show the dry-season → flood signal consistently
across all reasonable threshold definitions. These are the reliable
predictors for flood forecasting.

Regions marked ❌ not robust only show the signal at specific thresholds,
meaning the earlier finding may have been an artifact of threshold choice.

5. RECOMMENDATION FOR THE REPORT
--------------------------------------------------------------------------------
Report the ROBUST regions as primary findings, and mention threshold
sensitivity as a robustness check.

================================================================================
"""

with open(STATS_DIR / 'threshold_sensitivity_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  ✓ Report saved")

print("\n" + "="*90)
print("✅ THRESHOLD SENSITIVITY ANALYSIS COMPLETE")
print("="*90)
print(f"\nResults: {OUT_DIR}")
for f in sorted(OUT_DIR.rglob('*')):
    if f.is_file():
        print(f"  - {f.relative_to(OUT_DIR)}")
print("="*90)