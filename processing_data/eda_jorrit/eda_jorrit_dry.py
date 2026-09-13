"""
DRY PERIODS AND FLOOD CORRELATION ANALYSIS - South Sudan
Corrected version with:
- ERA5 units converted from meters to mm
- Total flooded pixels per day as flood severity proxy
- Multiple dry-period thresholds (absolute + percentile-based)
- Flood-severity-aware analysis
"""

import sys
import os
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
except:
    HAS_PLT = False

try:
    from scipy.stats import pearsonr, spearmanr, mannwhitneyu
    HAS_SCIPY = True
except:
    HAS_SCIPY = False

from loading import load_rainfall_runoff, load_flood_masks
from loading_impact_data import load_admin_boundaries

# ============================================================================
# SETUP
# ============================================================================
RESULTS_DIR = Path(r'C:\Users\20241060\OneDrive - TU Eindhoven\JBG060\JBG060_ZHL_2026_group_12\processing_data\results_eda_jorrit')
DRY_ANALYSIS_DIR = RESULTS_DIR / 'dry_periods_analysis'
DRY_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

FIGS_DIR = DRY_ANALYSIS_DIR / 'figures'
TABLES_DIR = DRY_ANALYSIS_DIR / 'tables'
STATS_DIR = DRY_ANALYSIS_DIR / 'statistics'

FIGS_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)
STATS_DIR.mkdir(exist_ok=True)

print("="*80)
print("🌵 DRY PERIODS & FLOOD CORRELATION - South Sudan")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA")
print("-"*80)

admin1, admin2 = load_admin_boundaries()
print(f"  ✓ Admin: {len(admin1)} states, {len(admin2)} counties")

# ---- ERA5: Convert meters → mm ----
print("\n  Loading ERA5...")
years = np.arange(2000, 2025)
era5 = load_rainfall_runoff(years)

bbox_ssd = {'lat_min': 3.0, 'lat_max': 13.0, 'lon_min': 24.0, 'lon_max': 36.0}
lat_idx = (era5.latitude >= bbox_ssd['lat_min']) & (era5.latitude <= bbox_ssd['lat_max'])
lon_idx = (era5.longitude >= bbox_ssd['lon_min']) & (era5.longitude <= bbox_ssd['lon_max'])
era5_ssd = era5.isel(latitude=lat_idx, longitude=lon_idx)

# ✅ CONVERT FROM METERS TO MM
rainfall_series = (era5_ssd['tp'].mean(dim=['latitude', 'longitude']) * 1000).to_series()
rainfall_series.name = 'rainfall_mm'

runoff_series = (era5_ssd['ro'].mean(dim=['latitude', 'longitude']) * 1000).to_series()
runoff_series.name = 'runoff_mm'

hydro_df = pd.concat([rainfall_series, runoff_series], axis=1).fillna(0)

print(f"  ✓ Rainfall: mean={hydro_df['rainfall_mm'].mean():.2f} mm/day, "
      f"max={hydro_df['rainfall_mm'].max():.2f} mm/day")
print(f"  ✓ Total period: {len(hydro_df)} days ({hydro_df.index.min().date()} to {hydro_df.index.max().date()})")

hydro_df.to_csv(TABLES_DIR / 'daily_rainfall_runoff_ssd.csv')

# ---- Flood data ----
print("\n  Loading flood data...")
flood_df = load_flood_masks(years, bbox=bbox_ssd)
flood_df['year'] = flood_df['date'].dt.year
flood_df['month'] = flood_df['date'].dt.month
print(f"  ✓ {len(flood_df):,} flood pixel-day events")

# Aggregate to daily total flooded pixels
flood_daily = flood_df.groupby('date').size().reset_index()
flood_daily.columns = ['date', 'flood_pixels']
flood_daily = flood_daily.set_index('date')
print(f"  ✓ Flood days: {len(flood_daily)}, "
      f"mean pixels/day: {flood_daily['flood_pixels'].mean():,.0f}, "
      f"max: {flood_daily['flood_pixels'].max():,}")

# ============================================================================
# 2. DRY PERIOD CHARACTERIZATION
# ============================================================================
print("\n[2] DRY PERIOD CHARACTERIZATION")
print("-"*80)

# Align to full daily range
date_range = pd.date_range(hydro_df.index.min(), hydro_df.index.max(), freq='D')
hydro_df = hydro_df.reindex(date_range, fill_value=0)
hydro_df.index.name = 'date'

# Multiple thresholds — absolute + percentile
abs_thresholds = [1.0, 2.0, 5.0]
percentile_thresholds = [25, 50]

dry_definitions = {}
for t in abs_thresholds:
    dry_definitions[f'abs_{t}mm'] = hydro_df['rainfall_mm'] < t
for p in percentile_thresholds:
    pval = np.percentile(hydro_df['rainfall_mm'], p)
    dry_definitions[f'pct_{p}'] = hydro_df['rainfall_mm'] < pval
    print(f"  Percentile {p} = {pval:.3f} mm/day")

def find_dry_streaks(is_dry_series):
    """Helper: find consecutive dry streaks"""
    streaks = []
    current = 0
    start = None
    for idx, is_dry in is_dry_series.items():
        if is_dry:
            if current == 0:
                start = idx
            current += 1
        else:
            if current > 0:
                streaks.append({'start_date': start, 'end_date': idx - timedelta(days=1), 'duration': current})
            current = 0
            start = None
    if current > 0:
        streaks.append({'start_date': start, 'end_date': is_dry_series.index[-1], 'duration': current})
    return pd.DataFrame(streaks)

dry_summary = {}
for name, is_dry in dry_definitions.items():
    dry_streaks = find_dry_streaks(is_dry)
    hydro_df[f'is_dry_{name}'] = is_dry
    
    if len(dry_streaks) > 0:
        dry_summary[name] = {
            'count': len(dry_streaks),
            'mean_duration': dry_streaks['duration'].mean(),
            'median_duration': dry_streaks['duration'].median(),
            'max_duration': dry_streaks['duration'].max(),
            'total_dry_days': dry_streaks['duration'].sum(),
            'pct_days_dry': 100 * dry_streaks['duration'].sum() / len(hydro_df)
        }
        dry_streaks.to_csv(TABLES_DIR / f'dry_periods_{name}.csv', index=False)
        
        print(f"\n  [{name}]")
        print(f"    Count: {len(dry_streaks)}, "
              f"Mean: {dry_streaks['duration'].mean():.1f}d, "
              f"Median: {dry_streaks['duration'].median():.0f}d, "
              f"Max: {dry_streaks['duration'].max():.0f}d, "
              f"% dry: {dry_summary[name]['pct_days_dry']:.1f}%")

summary_df = pd.DataFrame(dry_summary).T
summary_df.index.name = 'definition'
summary_df.to_csv(TABLES_DIR / 'dry_periods_summary.csv')
print(f"\n  ✓ Summary saved")

# Use primary definition for main analysis
PRIMARY = 'abs_1.0mm'
dry_primary_df = pd.read_csv(TABLES_DIR / f'dry_periods_{PRIMARY}.csv')
print(f"\n  → Using primary definition: {PRIMARY}")

# ============================================================================
# 3. DRY-FLOOD CORRELATION
# ============================================================================
print("\n[3] DRY-FLOOD CORRELATION ANALYSIS")
print("-"*80)

combined_df = hydro_df.copy()
combined_df['flood_pixels'] = flood_daily['flood_pixels']
combined_df['flood_pixels'] = combined_df['flood_pixels'].fillna(0)
combined_df['has_flood'] = combined_df['flood_pixels'] > 0

print(f"  Flood days: {combined_df['has_flood'].sum()} / {len(combined_df)}")
print(f"  Mean flood pixels on flood days: {combined_df.loc[combined_df['has_flood'], 'flood_pixels'].mean():,.0f}")

# Antecedent dry metrics
lookbacks = [7, 14, 30, 60, 90, 120, 180]
for lb in lookbacks:
    combined_df[f'dry_days_{lb}'] = combined_df[f'is_dry_{PRIMARY}'].rolling(lb).sum()
    combined_df[f'rain_sum_{lb}'] = combined_df['rainfall_mm'].rolling(lb).sum()

# Compare dry days before floods vs non-floods
flood_days = combined_df[combined_df['has_flood']]
no_flood_days = combined_df[~combined_df['has_flood']]

print("\n  Dry days before floods vs non-floods:")
corr_results = []

for lb in lookbacks:
    col = f'dry_days_{lb}'
    fm = flood_days[col].mean()
    nm = no_flood_days[col].mean()
    
    if HAS_SCIPY:
        try:
            _, p_val = mannwhitneyu(flood_days[col].dropna(), no_flood_days[col].dropna())
            p_str = f"p={p_val:.4f}"
        except:
            p_val, p_str = np.nan, "p=N/A"
        
        try:
            # Correlation with flood severity (pixels)
            valid = combined_df.dropna(subset=[col, 'flood_pixels'])
            pr, pp = pearsonr(valid[col], valid['flood_pixels'])
            sr, sp = spearmanr(valid[col], valid['flood_pixels'])
        except:
            pr, pp, sr, sp = np.nan, np.nan, np.nan, np.nan
    else:
        p_val, p_str, pr, pp, sr, sp = [np.nan]*6
    
    corr_results.append({
        'lookback_days': lb,
        'flood_mean_dry_days': fm,
        'noflood_mean_dry_days': nm,
        'difference': fm - nm,
        'mannwhitney_p': p_val,
        'pearson_r': pr, 'pearson_p': pp,
        'spearman_r': sr, 'spearman_p': sp
    })
    
    print(f"    {lb:>3}d: floods={fm:>5.1f}, non-floods={nm:>5.1f}, diff={fm-nm:>+5.1f}, {p_str}")

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(TABLES_DIR / 'dry_period_flood_correlations.csv', index=False)
print(f"\n  ✓ Correlations saved")

# ============================================================================
# 4. DRY PERIOD SEVERITY → FLOOD SEVERITY
# ============================================================================
print("\n[4] DRY PERIOD SEVERITY → FLOOD SEVERITY")
print("-"*80)

severity_data = []
for _, dp in dry_primary_df.iterrows():
    if dp['duration'] < 5:  # Skip very short dry periods
        continue
    
    end_date = pd.to_datetime(dp['end_date'])
    # Look for floods in the 45 days AFTER the dry period ends
    window = combined_df.loc[end_date + timedelta(days=1):end_date + timedelta(days=45)]
    floods_in_window = window[window['has_flood']]
    
    if len(floods_in_window) > 0:
        severity_data.append({
            'dry_start': dp['start_date'],
            'dry_end': dp['end_date'],
            'duration_days': dp['duration'],
            'max_flood_pixels': floods_in_window['flood_pixels'].max(),
            'total_flood_pixels': floods_in_window['flood_pixels'].sum(),
            'days_to_first_flood': (floods_in_window.index[0] - end_date).days,
            'n_flood_days': len(floods_in_window),
            # Rainfall after dry period
            'rain_after_7d': window['rainfall_mm'].iloc[:7].sum() if len(window) >= 7 else np.nan,
            'rain_after_14d': window['rainfall_mm'].iloc[:14].sum() if len(window) >= 14 else np.nan,
        })

severity_df = pd.DataFrame(severity_data)
severity_df.to_csv(TABLES_DIR / 'dry_period_severity.csv', index=False)

if len(severity_df) > 0:
    print(f"  Dry periods (≥5d) followed by floods: {len(severity_df)}")
    print(f"  Mean dry duration: {severity_df['duration_days'].mean():.1f} days")
    
    # ✅ FIX: use .mean() to collapse Series before formatting
    mean_pixels = severity_df['max_flood_pixels'].mean()
    print(f"  Mean max flood pixels: {mean_pixels:,.0f}")
    
    print(f"  Mean days to first flood: {severity_df['days_to_first_flood'].mean():.1f}")
    
    if HAS_SCIPY and len(severity_df) > 3:
        # Ensure we pass 1D arrays (in case of duplicate columns)
        dur = severity_df['duration_days'].values.ravel()
        pix = severity_df['max_flood_pixels'].values.ravel()
        
        r1, p1 = pearsonr(dur, pix)
        r2, p2 = spearmanr(dur, pix)
        print(f"\n  Duration → Flood severity:")
        print(f"    Pearson: r={r1:.3f} (p={p1:.4f})")
        print(f"    Spearman: r={r2:.3f} (p={p2:.4f})")
        
        # Does rain after dry period matter?
        valid_rain = severity_df.dropna(subset=['rain_after_14d'])
        if len(valid_rain) > 3:
            r3, p3 = pearsonr(valid_rain['rain_after_14d'], valid_rain['max_flood_pixels'])
            print(f"\n  Rain-after-dry-period (14d) → Flood severity:")
            print(f"    Pearson: r={r3:.3f} (p={p3:.4f})")

# ============================================================================
# 5. SEASONAL & TEMPORAL PATTERNS
# ============================================================================
print("\n[5] SEASONAL PATTERNS")
print("-"*80)

hydro_df['month'] = hydro_df.index.month
monthly = hydro_df.groupby('month').agg(
    mean_rain=('rainfall_mm', 'mean'),
    dry_days_pct=(f'is_dry_{PRIMARY}', lambda x: x.mean() * 100),
    mean_runoff=('runoff_mm', 'mean'),
).reset_index()

# Flood by month
flood_monthly = flood_daily.copy()
flood_monthly['month'] = flood_monthly.index.month
flood_monthly = flood_monthly.groupby('month')['flood_pixels'].mean().reset_index()
monthly = monthly.merge(flood_monthly, on='month', how='left')

monthly.to_csv(TABLES_DIR / 'monthly_patterns.csv', index=False)

print("\n  Month | Rain (mm/d) | Dry % | Flood pixels")
for _, r in monthly.iterrows():
    print(f"    {int(r['month']):>2}  |   {r['mean_rain']:>6.2f}    | {r['dry_days_pct']:>5.1f} | {r['flood_pixels']:>10,.0f}")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
if HAS_PLT:
    print("\n[6] CREATING VISUALIZATIONS")
    print("-"*80)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        
        # 1. Monthly rainfall vs dry days
        ax = axes[0, 0]
        ax2 = ax.twinx()
        ax.bar(monthly['month'], monthly['mean_rain'], color='steelblue', alpha=0.7, label='Rainfall')
        ax2.plot(monthly['month'], monthly['dry_days_pct'], 'o-', color='orange', linewidth=2, label='Dry days %')
        ax.set_xlabel('Month')
        ax.set_ylabel('Mean Rainfall (mm/day)', color='steelblue')
        ax2.set_ylabel('Dry Days (%)', color='orange')
        ax.set_title('Monthly Rainfall and Dry Days', fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        ax.grid(alpha=0.3)
        
        # 2. Dry period distribution
        ax = axes[0, 1]
        ax.hist(dry_primary_df['duration'], bins=50, color='orange', alpha=0.7, edgecolor='black')
        ax.axvline(dry_primary_df['duration'].median(), color='red', linestyle='--',
                   label=f"Median: {dry_primary_df['duration'].median():.0f}d")
        ax.axvline(dry_primary_df['duration'].mean(), color='green', linestyle='--',
                   label=f"Mean: {dry_primary_df['duration'].mean():.1f}d")
        ax.set_xlabel('Dry Period Duration (days)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Dry Period Duration Distribution ({PRIMARY})', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Dry days before floods vs non-floods
        ax = axes[1, 0]
        x = np.arange(len(lookbacks))
        w = 0.35
        flood_means = [corr_df[corr_df['lookback_days']==lb]['flood_mean_dry_days'].values[0] for lb in lookbacks]
        noflood_means = [corr_df[corr_df['lookback_days']==lb]['noflood_mean_dry_days'].values[0] for lb in lookbacks]
        ax.bar(x - w/2, flood_means, w, label='Flood days', color='red', alpha=0.7)
        ax.bar(x + w/2, noflood_means, w, label='Non-flood days', color='blue', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{lb}d' for lb in lookbacks])
        ax.set_xlabel('Lookback Period')
        ax.set_ylabel('Mean Dry Days')
        ax.set_title('Antecedent Dry Days: Flood vs Non-Flood Days', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Duration → Flood severity scatter
        ax = axes[1, 1]
        if len(severity_df) > 0:
            ax.scatter(severity_df['duration_days'], severity_df['max_flood_pixels'],
                       alpha=0.5, s=30, c='darkred')
            ax.set_xlabel('Dry Period Duration (days)')
            ax.set_ylabel('Max Flooded Pixels')
            ax.set_title('Dry Period Duration → Flood Severity', fontweight='bold')
            ax.set_yscale('log')
            ax.grid(alpha=0.3, which='both')
            if HAS_SCIPY and len(severity_df) > 3:
                ax.text(0.05, 0.95, f'Spearman r={r2:.3f}\np={p2:.4f}',
                        transform=ax.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('South Sudan: Dry Periods & Flood Analysis', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'dry_periods_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: dry_periods_analysis.png")
        
        # ---- Time series figure ----
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        subset = combined_df.loc['2018':'2022']
        
        # Panel 1: Rainfall
        ax = axes[0]
        ax.fill_between(subset.index, 0, subset['rainfall_mm'], color='steelblue', alpha=0.7)
        ax.set_ylabel('Rainfall (mm/day)')
        ax.set_title('Rainfall, Dry Periods, and Floods (2018-2022)', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Highlight dry periods
        for _, dp in dry_primary_df.iterrows():
            start = pd.to_datetime(dp['start_date'])
            end = pd.to_datetime(dp['end_date'])
            if start >= pd.Timestamp('2018-01-01') and end <= pd.Timestamp('2022-12-31'):
                if dp['duration'] >= 7:
                    for a in axes:
                        a.axvspan(start, end, alpha=0.15, color='orange')
        
        # Panel 2: Runoff
        ax = axes[1]
        ax.fill_between(subset.index, 0, subset['runoff_mm'], color='green', alpha=0.7)
        ax.set_ylabel('Runoff (mm/day)')
        ax.grid(alpha=0.3)
        
        # Panel 3: Flood pixels (log scale for visibility)
        ax = axes[2]
        ax.fill_between(subset.index, 1, subset['flood_pixels'].clip(lower=1), 
                        color='red', alpha=0.7)
        ax.set_ylabel('Flooded Pixels (log)')
        ax.set_yscale('log')
        ax.set_xlabel('Date')
        ax.grid(alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'dry_periods_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: dry_periods_timeseries.png")
        
    except Exception as e:
        print(f"  ⚠ Plot error: {e}")
        import traceback; traceback.print_exc()

# ============================================================================
# 7. REPORT
# ============================================================================
print("\n[7] WRITING REPORT")
print("-"*80)

report = f"""
================================================================================
DRY PERIODS & FLOOD ANALYSIS - SOUTH SUDAN
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

1. DATA
--------------------------------------------------------------------------------
Period:        {hydro_df.index.min().date()} to {hydro_df.index.max().date()}
Days:          {len(hydro_df)}
Mean rainfall: {hydro_df['rainfall_mm'].mean():.2f} mm/day
Annual avg:    {hydro_df['rainfall_mm'].mean() * 365:.0f} mm/year
Flood events:  {len(flood_df):,} pixel-days
Flood days:    {combined_df['has_flood'].sum()}

2. DRY PERIODS ({PRIMARY})
--------------------------------------------------------------------------------
Count:          {len(dry_primary_df)}
Mean duration:  {dry_primary_df['duration'].mean():.1f} days
Median duration:{dry_primary_df['duration'].median():.0f} days
Max duration:   {dry_primary_df['duration'].max():.0f} days
% of days dry:  {100 * dry_primary_df['duration'].sum() / len(hydro_df):.1f}%

3. ANTECEDENT DRY DAYS: FLOODS vs NON-FLOODS
--------------------------------------------------------------------------------
"""
for _, r in corr_df.iterrows():
    report += f"{r['lookback_days']:>3}d lookback: floods={r['flood_mean_dry_days']:.1f}d, "
    report += f"non-floods={r['noflood_mean_dry_days']:.1f}d, diff={r['difference']:+.1f}\n"

report += f"""
4. DRY DURATION → FLOOD SEVERITY
--------------------------------------------------------------------------------
"""
if len(severity_df) > 0:
    report += f"""Dry periods (≥5d) followed by floods: {len(severity_df)}
Mean duration: {severity_df['duration_days'].mean():.1f} days
Mean days to first flood: {severity_df['days_to_first_flood'].mean():.1f}
"""
    if HAS_SCIPY and len(severity_df) > 3:
        report += f"""Pearson r  (duration → flood severity): {r1:.3f} (p={p1:.4f})
Spearman r (duration → flood severity): {r2:.3f} (p={p2:.4f})
"""

report += f"""
5. KEY TAKEAWAYS
--------------------------------------------------------------------------------
- Dry periods are longest in Dec-Mar (dry season), shortest Jun-Sep (wet season)
- Average annual rainfall: {hydro_df['rainfall_mm'].mean() * 365:.0f} mm (realistic for SSD)
- Flood activity peaks in Aug-Oct, aligned with wet season
- Best correlation lookback: see section 3 above

================================================================================
"""

with open(STATS_DIR / 'dry_periods_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  ✓ Report saved")

print("\n" + "="*80)
print("✅ COMPLETE")
print("="*80)
print(f"\nResults: {DRY_ANALYSIS_DIR}")
print("\nFiles:")
for f in sorted(DRY_ANALYSIS_DIR.rglob('*')):
    if f.is_file():
        print(f"  - {f.relative_to(DRY_ANALYSIS_DIR)}")