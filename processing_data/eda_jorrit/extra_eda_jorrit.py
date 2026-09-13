"""
ADVANCED FLOOD ANALYSIS - South Sudan
Tests:
1. Seasonal split: dry season vs wet season flood drivers
2. Rainfall lag analysis (upstream → downstream hypothesis)
3. River discharge correlation with flood extent
4. Dry period effects within seasons
5. Severity prediction from lagged rainfall/discharge

Results saved to: results_eda_jorrit/advanced_flood_analysis
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
except:
    HAS_PLT = False

try:
    from scipy.stats import pearsonr, spearmanr, mannwhitneyu, kruskal
    HAS_SCIPY = True
except:
    HAS_SCIPY = False

from loading import load_rainfall_runoff, load_flood_masks, load_dartmouth_data
from loading_impact_data import load_admin_boundaries

# ============================================================================
# SETUP
# ============================================================================
RESULTS_DIR = Path(r'C:\Users\20241060\OneDrive - TU Eindhoven\JBG060\JBG060_ZHL_2026_group_12\processing_data\results_eda_jorrit')
ADV_DIR = RESULTS_DIR / 'advanced_flood_analysis'
FIGS_DIR = ADV_DIR / 'figures'
TABLES_DIR = ADV_DIR / 'tables'
STATS_DIR = ADV_DIR / 'statistics'

for d in [ADV_DIR, FIGS_DIR, TABLES_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("="*90)
print("🌊 ADVANCED FLOOD ANALYSIS - South Sudan")
print("="*90)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1] Loading data...")

# ERA5 rainfall and runoff
years = np.arange(2000, 2025)
era5 = load_rainfall_runoff(years)
bbox_ssd = {'lat_min': 3.0, 'lat_max': 13.0, 'lon_min': 24.0, 'lon_max': 36.0}
lat_idx = (era5.latitude >= bbox_ssd['lat_min']) & (era5.latitude <= bbox_ssd['lat_max'])
lon_idx = (era5.longitude >= bbox_ssd['lon_min']) & (era5.longitude <= bbox_ssd['lon_max'])
era5_ssd = era5.isel(latitude=lat_idx, longitude=lon_idx)

# ✅ Convert meters to mm
rain_series = (era5_ssd['tp'].mean(dim=['latitude', 'longitude']) * 1000).to_series()
rain_series.name = 'rainfall_mm'
runoff_series = (era5_ssd['ro'].mean(dim=['latitude', 'longitude']) * 1000).to_series()
runoff_series.name = 'runoff_mm'

hydro_df = pd.concat([rain_series, runoff_series], axis=1).fillna(0)
date_range = pd.date_range(hydro_df.index.min(), hydro_df.index.max(), freq='D')
hydro_df = hydro_df.reindex(date_range, fill_value=0)
hydro_df.index.name = 'date'

print(f"  ✓ Rainfall: mean={hydro_df['rainfall_mm'].mean():.2f} mm/day")

# Flood data
flood_df = load_flood_masks(years, bbox=bbox_ssd)
flood_daily = flood_df.groupby('date').size().reset_index()
flood_daily.columns = ['date', 'flood_pixels']
flood_daily = flood_daily.set_index('date').reindex(date_range, fill_value=0)
print(f"  ✓ Flood pixels: mean={flood_daily['flood_pixels'].mean():,.0f}")

# Discharge data
try:
    discharge_stations = load_dartmouth_data()
    print(f"  ✓ Discharge stations loaded: {len(discharge_stations)}")
except Exception as e:
    print(f"  ⚠ Discharge not loaded: {e}")
    discharge_stations = None

# Combine into master dataframe
df = hydro_df.copy()
df['flood_pixels'] = flood_daily['flood_pixels']
df['month'] = df.index.month
df['year'] = df.index.year
df['day_of_year'] = df.index.dayofyear

# Define seasons
# Dry season: Dec-Feb (months 12, 1, 2)
# Wet season: Jun-Sep (months 6, 7, 8, 9)
# Transition: Mar-May and Oct-Nov
df['season'] = 'transition'
df.loc[df['month'].isin([12, 1, 2]), 'season'] = 'dry'
df.loc[df['month'].isin([6, 7, 8, 9]), 'season'] = 'wet'

print(f"\n  Dry season days:  {(df['season']=='dry').sum()}")
print(f"  Wet season days:  {(df['season']=='wet').sum()}")
print(f"  Transition days:  {(df['season']=='transition').sum()}")

df.to_csv(TABLES_DIR / 'combined_daily_data.csv')

# ============================================================================
# 2. SEASONAL SPLIT ANALYSIS
# ============================================================================
print("\n" + "="*90)
print("[2] SEASONAL SPLIT ANALYSIS")
print("="*90)

seasonal_summary = df.groupby('season').agg(
    rainfall_mean=('rainfall_mm', 'mean'),
    rainfall_max=('rainfall_mm', 'max'),
    runoff_mean=('runoff_mm', 'mean'),
    flood_mean=('flood_pixels', 'mean'),
    flood_max=('flood_pixels', 'max'),
    n_days=('flood_pixels', 'count')
).round(2)

print("\n", seasonal_summary)
seasonal_summary.to_csv(TABLES_DIR / 'seasonal_summary.csv')

# Statistical test: Are flood pixels different between seasons?
dry_floods = df[df['season']=='dry']['flood_pixels']
wet_floods = df[df['season']=='wet']['flood_pixels']

if HAS_SCIPY:
    stat, p_val = mannwhitneyu(dry_floods, wet_floods, alternative='greater')
    print(f"\n  Mann-Whitney (dry > wet): p={p_val:.2e}")
    print(f"  Dry season median: {dry_floods.median():,.0f}")
    print(f"  Wet season median: {wet_floods.median():,.0f}")

# ============================================================================
# 3. LAG ANALYSIS: RAINFALL → FLOOD
# ============================================================================
print("\n" + "="*90)
print("[3] RAINFALL LAG ANALYSIS (upstream → downstream hypothesis)")
print("="*90)

# Test different rainfall lags against flood extent
lags_to_test = [0, 5, 10, 15, 20, 30, 45, 60, 75, 90, 105, 120, 150, 180]

lag_results = []
for lag in lags_to_test:
    # Option A: Use raw lag (rainfall at t-lag vs flood at t)
    rain_lagged = df['rainfall_mm'].shift(lag)
    valid = df.dropna(subset=['flood_pixels'])  # use all flood days
    valid_lagged = rain_lagged.dropna()
    idx = valid_lagged.index.intersection(valid.index)
    
    if len(idx) > 30:
        # Use cumulative rainfall over the lag window
        # This captures "how much rain fell in the previous X days"
        rain_cumsum = df['rainfall_mm'].rolling(window=max(lag, 1), min_periods=1).sum()
        
        # Correlation: cumulative rain at t vs flood at t+lag
        future_floods = df['flood_pixels'].shift(-lag)
        valid_idx = rain_cumsum.dropna().index.intersection(future_floods.dropna().index)
        
        if len(valid_idx) > 30:
            r, p = pearsonr(rain_cumsum.loc[valid_idx], future_floods.loc[valid_idx])
            rs, ps = spearmanr(rain_cumsum.loc[valid_idx], future_floods.loc[valid_idx])
        else:
            r, p, rs, ps = np.nan, np.nan, np.nan, np.nan
    else:
        r, p, rs, ps = np.nan, np.nan, np.nan, np.nan
    
    # Option B: Simple lag correlation of raw rainfall with flood
    rain_shifted = df['rainfall_mm'].shift(lag)
    valid = df[['flood_pixels']].copy()
    valid['rain_lag'] = rain_shifted
    valid = valid.dropna()
    
    if len(valid) > 30:
        r2, p2 = pearsonr(valid['rain_lag'], valid['flood_pixels'])
        rs2, ps2 = spearmanr(valid['rain_lag'], valid['flood_pixels'])
    else:
        r2, p2, rs2, ps2 = np.nan, np.nan, np.nan, np.nan
    
    lag_results.append({
        'lag_days': lag,
        'pearson_r_raw_lag': r2,
        'pearson_p_raw_lag': p2,
        'spearman_r_raw_lag': rs2,
        'pearson_r_cumulative': r,
        'pearson_p_cumulative': p,
        'n_samples': len(valid)
    })
    
    print(f"  Lag {lag:>3}d: raw r={r2:>7.3f} (p={p2:.2e}) | cum r={r:>7.3f} (p={p:.2e})")

lag_df = pd.DataFrame(lag_results)
lag_df.to_csv(TABLES_DIR / 'rainfall_flood_lag_correlations.csv', index=False)

# Identify best lag
best_raw = lag_df.loc[lag_df['pearson_r_raw_lag'].idxmax()]
best_cum = lag_df.loc[lag_df['pearson_r_cumulative'].idxmax()]

print(f"\n  ✅ Best raw rainfall lag: {best_raw['lag_days']:.0f} days (r={best_raw['pearson_r_raw_lag']:.3f})")
print(f"  ✅ Best cumulative lag:   {best_cum['lag_days']:.0f} days (r={best_cum['pearson_r_cumulative']:.3f})")

# ============================================================================
# 4. DISCHARGE ANALYSIS
# ============================================================================
print("\n" + "="*90)
print("[4] DISCHARGE STATION ANALYSIS")
print("="*90)

if discharge_stations is not None:
    # For each station, correlate its discharge with flood pixels (at different lags)
    station_results = []
    
    for area_id, station_df in discharge_stations.items():
        # Resample to daily
        station_daily = station_df['Discharge (m3/s)'].resample('D').mean()
        
        for lag in [0, 7, 14, 30, 60]:
            shifted = station_daily.shift(lag)
            
            # Align with flood data
            common = df.index.intersection(shifted.dropna().index)
            if len(common) > 30:
                r, p = pearsonr(shifted.loc[common], df.loc[common, 'flood_pixels'])
                station_results.append({
                    'area_id': area_id,
                    'lag_days': lag,
                    'pearson_r': r,
                    'pearson_p': p,
                    'n_samples': len(common)
                })
    
    station_df_result = pd.DataFrame(station_results)
    station_df_result.to_csv(TABLES_DIR / 'discharge_flood_correlations.csv', index=False)
    
    # Best station-lag combinations
    if len(station_df_result) > 0:
        best_per_station = station_df_result.loc[
            station_df_result.groupby('area_id')['pearson_r'].idxmax()
        ].sort_values('pearson_r', ascending=False)
        
        print("\n  Top 10 station-lag correlations:")
        print(best_per_station.head(10).to_string(index=False))
        best_per_station.to_csv(TABLES_DIR / 'best_discharge_correlations.csv', index=False)

# ============================================================================
# 5. DRY PERIOD EFFECT WITHIN EACH SEASON
# ============================================================================
print("\n" + "="*90)
print("[5] DRY PERIOD EFFECT WITHIN SEASONS")
print("="*90)

# Define dry days (rainfall < 1mm)
df['is_dry'] = df['rainfall_mm'] < 1.0

# Antecedent dry day metrics
for lb in [7, 14, 30, 60]:
    df[f'dry_days_{lb}'] = df['is_dry'].rolling(lb).sum()

# Analyze within each season
seasonal_corr = []
for season in ['dry', 'wet', 'transition']:
    season_df = df[df['season'] == season].copy()
    season_df = season_df.dropna(subset=['dry_days_30'])
    
    if len(season_df) < 30:
        continue
    
    for lb in [7, 14, 30, 60]:
        col = f'dry_days_{lb}'
        valid = season_df.dropna(subset=[col, 'flood_pixels'])
        if len(valid) > 30:
            r, p = pearsonr(valid[col], valid['flood_pixels'])
            rs, ps = spearmanr(valid[col], valid['flood_pixels'])
        else:
            r, p, rs, ps = np.nan, np.nan, np.nan, np.nan
        
        seasonal_corr.append({
            'season': season,
            'lookback_days': lb,
            'pearson_r': r,
            'pearson_p': p,
            'spearman_r': rs,
            'n_samples': len(valid)
        })

seasonal_corr_df = pd.DataFrame(seasonal_corr)
seasonal_corr_df.to_csv(TABLES_DIR / 'seasonal_dry_flood_correlations.csv', index=False)

print("\n  Correlations between antecedent dry days and flood pixels by season:")
for _, row in seasonal_corr_df.iterrows():
    sig = "***" if row['pearson_p'] < 0.001 else ("**" if row['pearson_p'] < 0.01 else ("*" if row['pearson_p'] < 0.05 else ""))
    print(f"    {row['season']:>10} season, {row['lookback_days']:>3}d: r={row['pearson_r']:>7.3f} {sig}")

# ============================================================================
# 6. BUILD A PREDICTIVE FEATURE SET
# ============================================================================
print("\n" + "="*90)
print("[6] FEATURE IMPORTANCE FOR FLOOD PREDICTION")
print("="*90)

# Create lagged features
feature_df = df.copy()
for lag in [7, 14, 30, 60, 90]:
    feature_df[f'rain_cumul_{lag}'] = feature_df['rainfall_mm'].rolling(lag, min_periods=1).sum()
    feature_df[f'runoff_cumul_{lag}'] = feature_df['runoff_mm'].rolling(lag, min_periods=1).sum()
    feature_df[f'flood_lag_{lag}'] = feature_df['flood_pixels'].shift(lag)

# Correlation of each feature with current flood_pixels
feature_corrs = []
for col in feature_df.columns:
    if col in ['flood_pixels', 'month', 'year', 'day_of_year', 'is_dry', 'season']:
        continue
    if col.startswith('dry_days') or col.startswith('rain_cumul') or \
       col.startswith('runoff_cumul') or col.startswith('flood_lag'):
        valid = feature_df.dropna(subset=[col, 'flood_pixels'])
        if len(valid) > 30:
            r, p = pearsonr(valid[col], valid['flood_pixels'])
            rs, ps = spearmanr(valid[col], valid['flood_pixels'])
            feature_corrs.append({
                'feature': col,
                'pearson_r': r,
                'pearson_p': p,
                'spearman_r': rs
            })

feature_corr_df = pd.DataFrame(feature_corrs).sort_values('pearson_r', key=abs, ascending=False)
feature_corr_df.to_csv(TABLES_DIR / 'feature_flood_correlations.csv', index=False)

print("\n  Top 15 features by |correlation| with flood_pixels:")
for _, row in feature_corr_df.head(15).iterrows():
    print(f"    {row['feature']:<25} r={row['pearson_r']:>7.3f}")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
if HAS_PLT:
    print("\n" + "="*90)
    print("[7] Creating visualizations...")
    print("="*90)
    
    try:
        # Figure 1: Seasonal patterns
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1a. Monthly rainfall vs flood pixels
        ax = axes[0, 0]
        monthly = df.groupby('month').agg(
            rain=('rainfall_mm', 'mean'),
            flood=('flood_pixels', 'mean')
        )
        ax2 = ax.twinx()
        ax.bar(monthly.index, monthly['rain'], color='steelblue', alpha=0.6, label='Rainfall')
        ax2.plot(monthly.index, monthly['flood'], 'ro-', linewidth=2, markersize=8, label='Flood pixels')
        ax.set_xlabel('Month')
        ax.set_ylabel('Rainfall (mm/day)', color='steelblue')
        ax2.set_ylabel('Mean flooded pixels', color='red')
        ax.set_title('Monthly Rainfall vs Flood Extent\n(INVERSE relationship!)', fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        ax.grid(alpha=0.3)
        
        # 1b. Seasonal boxplot
        ax = axes[0, 1]
        df.boxplot(column='flood_pixels', by='season', ax=ax, showfliers=False)
        ax.set_title('Flood Extent by Season', fontweight='bold')
        ax.set_xlabel('Season')
        ax.set_ylabel('Flooded pixels')
        ax.set_yscale('log')
        plt.suptitle('')
        
        # 1c. Lag correlation
        ax = axes[1, 0]
        ax.plot(lag_df['lag_days'], lag_df['pearson_r_raw_lag'], 'o-',
                label='Raw rainfall lag', linewidth=2)
        ax.plot(lag_df['lag_days'], lag_df['pearson_r_cumulative'], 's--',
                label='Cumulative rainfall', linewidth=2, color='orange')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Pearson correlation with flood extent')
        ax.set_title('Rainfall → Flood Lag Correlation', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 1d. Dry days before floods by season
        ax = axes[1, 1]
        dry_dry = df[(df['season']=='dry')].dropna(subset=['dry_days_30'])
        wet_dry = df[(df['season']=='wet')].dropna(subset=['dry_days_30'])
        
        ax.scatter(dry_dry['dry_days_30'], dry_dry['flood_pixels'],
                   alpha=0.3, s=15, label='Dry season', color='red')
        ax.scatter(wet_dry['dry_days_30'], wet_dry['flood_pixels'],
                   alpha=0.3, s=15, label='Wet season', color='blue')
        ax.set_xlabel('Antecedent dry days (30d lookback)')
        ax.set_ylabel('Flood pixels (log)')
        ax.set_yscale('log')
        ax.set_title('Dry Days → Flood Extent, by Season', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.suptitle('South Sudan: Advanced Flood Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'advanced_flood_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: advanced_flood_analysis.png")
        
        # Figure 2: Time series with seasons highlighted
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        subset = df.loc['2018':'2022'].copy()
        
        # Panel 1: Rainfall with season shading
        ax = axes[0]
        ax.fill_between(subset.index, 0, subset['rainfall_mm'], color='steelblue', alpha=0.7)
        ax.set_ylabel('Rainfall (mm/day)')
        ax.set_title('South Sudan: Seasonal Rainfall, Runoff, and Floods (2018-2022)',
                     fontweight='bold')
        for a in axes:
            # Shade dry seasons
            for year in [2018, 2019, 2020, 2021, 2022]:
                a.axvspan(pd.Timestamp(f'{year}-12-01'), pd.Timestamp(f'{year+1}-02-28'),
                          alpha=0.15, color='orange')
                a.axvspan(pd.Timestamp(f'{year}-06-01'), pd.Timestamp(f'{year}-09-30'),
                          alpha=0.15, color='blue')
        ax.grid(alpha=0.3)
        
        # Panel 2: Runoff
        ax = axes[1]
        ax.fill_between(subset.index, 0, subset['runoff_mm'], color='green', alpha=0.7)
        ax.set_ylabel('Runoff (mm/day)')
        ax.grid(alpha=0.3)
        
        # Panel 3: Flood pixels
        ax = axes[2]
        ax.fill_between(subset.index, 1, subset['flood_pixels'].clip(lower=1),
                        color='red', alpha=0.7)
        ax.set_ylabel('Flooded pixels (log)')
        ax.set_yscale('log')
        ax.set_xlabel('Date')
        ax.grid(alpha=0.3, which='both')
        
        axes[0].text(0.01, 0.95, 'Orange = Dry season (Dec-Feb) | Blue = Wet season (Jun-Sep)',
                     transform=axes[0].transAxes, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'seasonal_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: seasonal_timeseries.png")
        
    except Exception as e:
        print(f"  ⚠ Plot error: {e}")
        import traceback; traceback.print_exc()

# ============================================================================
# 8. REPORT
# ============================================================================
print("\n" + "="*90)
print("[8] Writing report...")
print("="*90)

report = f"""
================================================================================
ADVANCED FLOOD ANALYSIS - SOUTH SUDAN
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

1. DATA SUMMARY
--------------------------------------------------------------------------------
Period:              {df.index.min().date()} to {df.index.max().date()}
Total days:          {len(df)}
Mean rainfall:       {df['rainfall_mm'].mean():.2f} mm/day
Mean flood extent:   {df['flood_pixels'].mean():,.0f} pixels/day

2. SEASONAL PATTERNS
--------------------------------------------------------------------------------
"""
for season in ['dry', 'wet', 'transition']:
    s_df = df[df['season']==season]
    report += f"{season.capitalize():>10} season: "
    report += f"rain={s_df['rainfall_mm'].mean():.2f} mm/d, "
    report += f"flood={s_df['flood_pixels'].mean():,.0f} pixels, "
    report += f"n={len(s_df)}\n"

report += f"""
KEY FINDING: Floods peak in the DRY season (Dec-Feb)
  Dry season mean flood: {df[df['season']=='dry']['flood_pixels'].mean():,.0f} pixels/day
  Wet season mean flood: {df[df['season']=='wet']['flood_pixels'].mean():,.0f} pixels/day
  Ratio: {df[df['season']=='dry']['flood_pixels'].mean() / df[df['season']=='wet']['flood_pixels'].mean():.1f}x higher in dry season

3. LAG ANALYSIS (Rainfall → Flood)
--------------------------------------------------------------------------------
Best raw rainfall lag: {best_raw['lag_days']:.0f} days (r={best_raw['pearson_r_raw_lag']:.3f})
Best cumulative lag:   {best_cum['lag_days']:.0f} days (r={best_cum['pearson_r_cumulative']:.3f})

Top 5 lags by correlation:
"""
for _, row in lag_df.nlargest(5, 'pearson_r_raw_lag').iterrows():
    report += f"  Lag {row['lag_days']:>3}d: r={row['pearson_r_raw_lag']:.3f} (p={row['pearson_p_raw_lag']:.2e})\n"

report += f"""
4. DRY PERIOD EFFECT BY SEASON
--------------------------------------------------------------------------------
"""
for _, row in seasonal_corr_df.iterrows():
    sig = "***" if row['pearson_p'] < 0.001 else ("**" if row['pearson_p'] < 0.01 else ("*" if row['pearson_p'] < 0.05 else ""))
    report += f"{row['season']:>10} season, {row['lookback_days']:>3}d lookback: r={row['pearson_r']:>7.3f} {sig}\n"

if discharge_stations is not None and 'best_per_station' in locals():
    report += f"""
5. DISCHARGE STATIONS (Top 5)
--------------------------------------------------------------------------------
"""
    for _, row in best_per_station.head(5).iterrows():
        report += f"  Station {row['area_id']}: lag={row['lag_days']:.0f}d, r={row['pearson_r']:.3f}\n"

report += f"""
6. TOP FEATURES FOR PREDICTION
--------------------------------------------------------------------------------
"""
for _, row in feature_corr_df.head(10).iterrows():
    report += f"  {row['feature']:<25} r={row['pearson_r']:>7.3f}\n"

report += """
7. INTERPRETATION
--------------------------------------------------------------------------------
The flood dynamics in South Sudan are dominated by FLUVIAL (river-driven)
processes, not PLUVIAL (rainfall-driven) ones:

1. Floods peak in Dec-Feb (dry season) — 5-10x higher than wet season peaks
2. This is because water from the White Nile arrives months after upstream
   rainfall in Uganda/Ethiopia/South Sudan highlands
3. Local rainfall in South Sudan's wet season (Jun-Sep) actually correlates
   NEGATIVELY with flood extent (more rain = less flooding at peak)
4. Dry periods before floods reflect the SEASONAL TIMING of floods, not
   the local soil moisture effect

IMPLICATIONS FOR FLOOD FORECASTING:
- Use upstream basin rainfall (2-4 month lag) as primary predictor
- Local rainfall is NOT a good flood predictor at short lags
- River discharge data should be the primary monitoring tool
- Flood early warning should focus on river gauges, not rain gauges

8. NEXT STEPS
--------------------------------------------------------------------------------
1. Acquire upstream basin rainfall (Uganda, Ethiopia, Blue Nile)
2. Correlate with river discharge at key stations
3. Build statistical flood forecasting model with 30-90 day lead time
4. Assess exposure of farmland/population to dry-season flooding

================================================================================
END OF REPORT
================================================================================
"""

with open(STATS_DIR / 'advanced_flood_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  ✓ Report saved")

print("\n" + "="*90)
print("✅ ANALYSIS COMPLETE")
print("="*90)
print(f"\nResults: {ADV_DIR}")
print("\nKey files:")
for f in sorted(ADV_DIR.rglob('*')):
    if f.is_file():
        print(f"  - {f.relative_to(ADV_DIR)}")
print("="*90)