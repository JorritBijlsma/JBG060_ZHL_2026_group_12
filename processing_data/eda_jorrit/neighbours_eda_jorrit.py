"""
UPSTREAM DRY SEASONS vs SOUTH SUDAN FLOODS
Tests whether dry season timing/length in neighbouring countries
controls flood extent in South Sudan.

Countries analyzed:
- Uganda (main upstream - White Nile source)
- NE DRC (Bahr el Ghazal source)
- Ethiopia (Sobat River source)
- West Kenya (Lake Victoria tributaries)

Results saved to: results_eda_jorrit/upstream_dry_seasons
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
    from scipy.stats import pearsonr, spearmanr, mannwhitneyu
    HAS_SCIPY = True
except:
    HAS_SCIPY = False

from loading import load_rainfall_runoff, load_flood_masks

# ============================================================================
# SETUP
# ============================================================================
RESULTS_DIR = Path(r'C:\Users\20241060\OneDrive - TU Eindhoven\JBG060\JBG060_ZHL_2026_group_12\processing_data\results_eda_jorrit')
OUT_DIR = RESULTS_DIR / 'upstream_dry_seasons'
FIGS_DIR = OUT_DIR / 'figures'
TABLES_DIR = OUT_DIR / 'tables'
STATS_DIR = OUT_DIR / 'statistics'

for d in [OUT_DIR, FIGS_DIR, TABLES_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("="*90)
print("🌍 UPSTREAM DRY SEASONS vs SOUTH SUDAN FLOODS")
print("="*90)

# ============================================================================
# 1. DEFINE REGIONS
# ============================================================================
print("\n[1] Defining regions...")

REGIONS = {
    'South Sudan':   {'lat_min': 3.5,  'lat_max': 12.5, 'lon_min': 24.0, 'lon_max': 36.0},
    'Uganda':        {'lat_min': -1.5, 'lat_max': 3.5,  'lon_min': 29.5, 'lon_max': 35.0},
    'NE DRC':        {'lat_min': 2.0,  'lat_max': 5.0,  'lon_min': 24.0, 'lon_max': 30.0},
    'Ethiopia':      {'lat_min': 5.0,  'lat_max': 12.0, 'lon_min': 33.0, 'lon_max': 37.0},
    'West Kenya':    {'lat_min': -1.0, 'lat_max': 1.0,  'lon_min': 33.5, 'lon_max': 35.5},
    'Sudan central': {'lat_min': 12.5, 'lat_max': 16.0, 'lon_min': 30.0, 'lon_max': 34.0},
}

# ============================================================================
# 2. LOAD ERA5 RAINFALL
# ============================================================================
print("\n[2] Loading ERA5 rainfall...")

years = np.arange(2000, 2025)
era5 = load_rainfall_runoff(years)
era5_tp_mm = era5['tp'] * 1000  # convert m → mm

print(f"  ✓ ERA5 loaded: {era5_tp_mm.shape}")

# Regional rainfall
regional_rain = {}
for name, box in REGIONS.items():
    lat_idx = (era5.latitude >= box['lat_min']) & (era5.latitude <= box['lat_max'])
    lon_idx = (era5.longitude >= box['lon_min']) & (era5.longitude <= box['lon_max'])
    region = era5_tp_mm.isel(latitude=lat_idx, longitude=lon_idx)
    mean_series = region.mean(dim=['latitude', 'longitude']).to_series()
    mean_series.name = name
    regional_rain[name] = mean_series

rain_df = pd.DataFrame(regional_rain)
rain_df.index.name = 'date'
print(f"  ✓ Regional rainfall: {rain_df.shape}")

# ============================================================================
# 3. LOAD SOUTH SUDAN FLOOD DATA
# ============================================================================
print("\n[3] Loading South Sudan floods...")

bbox_ssd = {'lat_min': 3.0, 'lat_max': 13.0, 'lon_min': 24.0, 'lon_max': 36.0}
flood_df = load_flood_masks(years, bbox=bbox_ssd)
flood_daily = flood_df.groupby('date').size().reset_index()
flood_daily.columns = ['date', 'flood_pixels']
flood_daily = flood_daily.set_index('date')

print(f"  ✓ Flood: mean={flood_daily['flood_pixels'].mean():,.0f} pixels/day")

# ============================================================================
# 4. DEFINE DRY SEASONS FOR EACH COUNTRY
# ============================================================================
print("\n[4] Defining dry seasons per country...")

# The concept: for each year, find the longest continuous dry period per country
DRY_THRESHOLD = 1.0  # mm/day

def find_annual_dry_season(rain_series, year, threshold=DRY_THRESHOLD):
    """
    Find longest dry period in a given year for a rainfall series.
    Returns: (start_date, end_date, duration_days)
    """
    year_data = rain_series[rain_series.index.year == year]
    is_dry = year_data < threshold
    
    if len(year_data) == 0 or not is_dry.any():
        return None
    
    # Find streaks
    streaks = []
    current = 0
    start = None
    for idx, dry in is_dry.items():
        if dry:
            if current == 0:
                start = idx
            current += 1
        else:
            if current > 0:
                streaks.append({'start': start, 'end': idx - timedelta(days=1),
                                'duration': current})
            current = 0
            start = None
    if current > 0:
        streaks.append({'start': start, 'end': is_dry.index[-1],
                        'duration': current})
    
    if not streaks:
        return None
    
    # Return longest streak
    return max(streaks, key=lambda x: x['duration'])

# Find dry season for each country each year
dry_seasons_data = {}

for country, series in regional_rain.items():
    country_data = []
    for year in range(2000, 2025):
        result = find_annual_dry_season(series, year)
        if result:
            country_data.append({
                'country': country,
                'year': year,
                'dry_start': result['start'],
                'dry_end': result['end'],
                'dry_duration': result['duration'],
            })
    dry_seasons_data[country] = pd.DataFrame(country_data)

# Combine into single DataFrame
dry_seasons_all = pd.concat(dry_seasons_data.values(), ignore_index=True)
dry_seasons_all.to_csv(TABLES_DIR / 'annual_dry_seasons_all_countries.csv', index=False)

print("\n  Longest annual dry season per country (mean duration):")
summary = dry_seasons_all.groupby('country')['dry_duration'].agg(['mean', 'min', 'max']).round(1)
summary.columns = ['Mean (d)', 'Min (d)', 'Max (d)']
print(summary.to_string())

# ============================================================================
# 5. LINK UPSTREAM DRY SEASON TIMING TO SSD FLOODS
# ============================================================================
print("\n[5] Linking upstream dry season timing to SSD floods...")

# Pivot: for each year, get dry season end date and duration per country
pivot_end = dry_seasons_all.pivot(index='year', columns='country', values='dry_end')
pivot_duration = dry_seasons_all.pivot(index='year', columns='country', values='dry_duration')

# Compute annual SSD flood metrics
annual_flood = flood_daily.copy()
annual_flood['year'] = annual_flood.index.year

annual_flood_stats = annual_flood.groupby('year').agg(
    flood_mean=('flood_pixels', 'mean'),
    flood_sum=('flood_pixels', 'sum'),
    flood_max=('flood_pixels', 'max'),
    flood_days=(('flood_pixels', lambda x: (x > 0).sum()))
).reset_index()

# Also compute SSD dry season end date
ssd_dry = dry_seasons_all[dry_seasons_all['country'] == 'South Sudan'][
    ['year', 'dry_end', 'dry_duration']
].rename(columns={'dry_end': 'ssd_dry_end', 'dry_duration': 'ssd_dry_duration'})

# Merge
annual_df = annual_flood_stats.merge(ssd_dry, on='year', how='left')

# Add upstream dry season info
for country in REGIONS.keys():
    if country == 'South Sudan':
        continue
    if country in pivot_end.columns:
        annual_df[f'{country}_dry_end'] = annual_df['year'].map(pivot_end[country])
        annual_df[f'{country}_dry_duration'] = annual_df['year'].map(pivot_duration[country])

annual_df.to_csv(TABLES_DIR / 'annual_dry_flood_merged.csv', index=False)
print(f"\n  ✓ Annual dataframe: {annual_df.shape}")
print("\n  Preview:")
print(annual_df.head().to_string())

# ============================================================================
# 6. CORRELATIONS: UPSTREAM DRY SEASON → SSD FLOODS
# ============================================================================
print("\n[6] Correlations: Upstream dry season → SSD floods...")

# Correlate dry season duration in each country with SSD flood metrics
corr_results = []

for country in REGIONS.keys():
    dur_col = f'{country}_dry_duration'
    end_col = f'{country}_dry_end'
    
    if dur_col not in annual_df.columns:
        continue
    
    valid = annual_df.dropna(subset=[dur_col])
    
    # Note: dry_end is a Timestamp; convert to day-of-year for correlation
    valid_with_doy = valid.copy()
    if end_col in valid_with_doy.columns:
        valid_with_doy[f'{country}_dry_end_doy'] = valid_with_doy[end_col].dt.dayofyear
    
    if len(valid) < 5:
        continue
    
    for flood_metric in ['flood_mean', 'flood_sum', 'flood_max']:
        # Duration → flood
        r, p = pearsonr(valid[dur_col], valid[flood_metric])
        corr_results.append({
            'country': country,
            'predictor': 'dry_duration',
            'flood_metric': flood_metric,
            'pearson_r': r,
            'pearson_p': p,
            'n_years': len(valid)
        })
        
        # End date (day of year) → flood
        doy_col = f'{country}_dry_end_doy'
        if doy_col in valid_with_doy.columns:
            valid_doy = valid_with_doy.dropna(subset=[doy_col])
            if len(valid_doy) > 5:
                r, p = pearsonr(valid_doy[doy_col], valid_doy[flood_metric])
                corr_results.append({
                    'country': country,
                    'predictor': 'dry_end_doy',
                    'flood_metric': flood_metric,
                    'pearson_r': r,
                    'pearson_p': p,
                    'n_years': len(valid_doy)
                })

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(TABLES_DIR / 'correlation_summary.csv', index=False)

print("\n  Correlations (upstream dry season → SSD floods):")
print(f"  {'Country':<15} {'Predictor':<15} {'Flood metric':<12} {'r':>8} {'p':>10}")
print("  " + "-"*70)

for _, row in corr_df.iterrows():
    sig = "***" if row['pearson_p'] < 0.001 else ("**" if row['pearson_p'] < 0.01 else ("*" if row['pearson_p'] < 0.05 else ""))
    print(f"  {row['country']:<15} {row['predictor']:<15} {row['flood_metric']:<12} "
          f"{row['pearson_r']:>8.3f} {row['pearson_p']:>10.4f} {sig}")

# ============================================================================
# 7. LAG: UPSTREAM DRY END → SSD FLOOD PEAK
# ============================================================================
print("\n[7] Time lag: Upstream dry season end → SSD flood peak...")

# For each year, find SSD flood peak date (max flood_pixels)
flood_peak_dates = flood_daily.copy()
flood_peak_dates['year'] = flood_peak_dates.index.year
flood_peaks = flood_peak_dates.groupby('year')['flood_pixels'].idxmax()

flood_peak_df = pd.DataFrame({
    'year': flood_peaks.index,
    'peak_date': flood_peaks.values
})

# Merge with dry season end dates per country
timing_df = flood_peak_df.copy()
for country in REGIONS.keys():
    if country in pivot_end.columns:                     # ← FIX: check country name
        timing_df[f'{country}_dry_end'] = timing_df['year'].map(pivot_end[country])

# Compute lag: dry_end → flood_peak (in days)
lag_results = []
for country in REGIONS.keys():
    col = f'{country}_dry_end'
    if col not in timing_df.columns:
        continue

    valid = timing_df.dropna(subset=[col, 'peak_date']).copy()
    if len(valid) == 0:
        continue

    valid[col] = pd.to_datetime(valid[col], utc=True)
    valid['peak_date'] = pd.to_datetime(valid['peak_date'], utc=True)
    valid['lag_days'] = (valid['peak_date'] - valid[col]).dt.days

    # Filter to plausible lags (peak after dry season ends, up to 1 year later)
    valid = valid[(valid['lag_days'] > -90) & (valid['lag_days'] < 365)]

    if len(valid) == 0:
        continue

    lag_results.append({
        'country': country,
        'mean_lag_days': valid['lag_days'].mean(),
        'median_lag_days': valid['lag_days'].median(),
        'std_lag_days': valid['lag_days'].std(),
        'n_years': len(valid),
        'lags': valid['lag_days'].tolist()
    })

if lag_results:
    lag_df = pd.DataFrame(lag_results).sort_values('mean_lag_days').reset_index(drop=True)
    lag_df.drop('lags', axis=1).to_csv(TABLES_DIR / 'dry_season_flood_peak_lags.csv', index=False)

    print("\n  Mean lag from dry season end to SSD flood peak:")
    for _, row in lag_df.iterrows():
        print(f"    {row['country']:<15}: {row['mean_lag_days']:>6.1f} days "
              f"(median {row['median_lag_days']:.0f}, n={row['n_years']})")
else:
    print("\n  ⚠ No lag data available")
    lag_df = pd.DataFrame(columns=['country', 'mean_lag_days',
                                   'median_lag_days', 'std_lag_days', 'n_years'])

# ============================================================================
# 8. YEAR-OVER-YEAR: DRY SEASON ANOMALY vs FLOOD ANOMALY
# ============================================================================
print("\n[8] Year-over-year: Dry season anomalies → flood anomalies...")

# Normalize both signals (z-scores)
anomaly_df = annual_df.copy()

for country in REGIONS.keys():
    dur_col = f'{country}_dry_duration'
    if dur_col in anomaly_df.columns:
        anomaly_df[f'{country}_dur_anom'] = (
            (anomaly_df[dur_col] - anomaly_df[dur_col].mean()) / anomaly_df[dur_col].std()
        )

# Flood anomaly
anomaly_df['flood_anom'] = (
    (anomaly_df['flood_mean'] - anomaly_df['flood_mean'].mean()) /
    anomaly_df['flood_mean'].std()
)

# Correlate anomalies
anomaly_corrs = []
for country in REGIONS.keys():
    anom_col = f'{country}_dur_anom'
    if anom_col not in anomaly_df.columns:
        continue
    valid = anomaly_df.dropna(subset=[anom_col, 'flood_anom'])
    if len(valid) < 5:
        continue
    r, p = pearsonr(valid[anom_col], valid['flood_anom'])
    anomaly_corrs.append({
        'country': country,
        'pearson_r': r,
        'pearson_p': p,
        'n_years': len(valid)
    })

anomaly_corr_df = pd.DataFrame(anomaly_corrs).sort_values('pearson_r', key=abs, ascending=False)
anomaly_corr_df.to_csv(TABLES_DIR / 'dry_season_flood_anomaly_correlations.csv', index=False)

print("\n  Dry season duration anomaly → flood anomaly:")
for _, row in anomaly_corr_df.iterrows():
    sig = "***" if row['pearson_p'] < 0.001 else ("**" if row['pearson_p'] < 0.01 else ("*" if row['pearson_p'] < 0.05 else ""))
    print(f"    {row['country']:<15}: r={row['pearson_r']:>7.3f} (p={row['pearson_p']:.4f}) {sig}")

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================
if HAS_PLT:
    print("\n[9] Creating visualizations...")
    
    try:
        # Figure 1: Overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        
        # 1a. Dry season duration by country (boxplot)
        ax = axes[0, 0]
        countries_ordered = ['South Sudan', 'Uganda', 'NE DRC', 'Ethiopia', 'West Kenya', 'Sudan central']
        data_to_plot = [dry_seasons_all[dry_seasons_all['country']==c]['dry_duration'].values
                        for c in countries_ordered if c in dry_seasons_all['country'].unique()]
        labels = [c for c in countries_ordered if c in dry_seasons_all['country'].unique()]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Annual dry season duration (days)')
        ax.set_title('Longest Dry Season per Year by Country', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3, axis='y')
        
        # 1b. Duration anomalies vs flood anomaly
        ax = axes[0, 1]
        for country in ['Uganda', 'NE DRC', 'Ethiopia']:
            anom_col = f'{country}_dur_anom'
            if anom_col in anomaly_df.columns:
                ax.scatter(anomaly_df[anom_col], anomaly_df['flood_anom'],
                          alpha=0.6, s=60, label=country)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Dry season duration anomaly (z-score)')
        ax.set_ylabel('SSD flood anomaly (z-score)')
        ax.set_title('Dry Season Anomaly → Flood Anomaly', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 1c. Monthly climatology: rainfall vs flood
        ax = axes[1, 0]
        # Normalize to compare patterns
        monthly_rain = rain_df.groupby(rain_df.index.month).mean()
        ax2 = ax.twinx()
        
        ax.plot(monthly_rain.index, monthly_rain['South Sudan'], 'o-',
                color='steelblue', linewidth=2, label='SSD rainfall')
        ax.plot(monthly_rain.index, monthly_rain['Uganda'], 's-',
                color='green', linewidth=2, label='Uganda rainfall')
        
        # Monthly floods
        monthly_flood = flood_daily.groupby(flood_daily.index.month).mean()
        ax2.bar(monthly_flood.index, monthly_flood['flood_pixels'],
                alpha=0.3, color='red', label='SSD floods')
        ax2.set_yscale('log')
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Rainfall (mm/day)')
        ax2.set_ylabel('Mean flood pixels (log)', color='red')
        ax.set_title('Monthly Climatology: Rainfall vs Floods', fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(alpha=0.3)
        
        # 1d. Lag from dry season end to flood peak
        ax = axes[1, 1]
        countries_plot = []
        means = []
        for _, row in lag_df.iterrows():
            countries_plot.append(row['country'])
            means.append(row['mean_lag_days'])
        
        colors_lag = ['red' if x > 0 else 'blue' for x in means]
        ax.barh(countries_plot, means, color=colors_lag, alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Days from dry season end → SSD flood peak')
        ax.set_title('Timing: Upstream Dry Season End → SSD Flood Peak', fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        
        plt.suptitle('Upstream Dry Seasons vs South Sudan Floods',
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'upstream_dry_seasons_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: upstream_dry_seasons_overview.png")
        
        # Figure 2: Time series with dry seasons marked
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        subset_start = '2015'
        subset_end = '2022'
        
        # Uganda rainfall with dry season shading
        ax = axes[0]
        ug = rain_df.loc[subset_start:subset_end, 'Uganda']
        ax.fill_between(ug.index, 0, ug, color='green', alpha=0.6)
        ax.set_ylabel('Uganda rainfall\n(mm/day)', color='green')
        # Mark dry seasons
        for _, row in dry_seasons_all[dry_seasons_all['country']=='Uganda'].iterrows():
            if subset_start <= str(row['dry_start'].year) <= subset_end:
                ax.axvspan(row['dry_start'], row['dry_end'], alpha=0.3, color='orange')
        ax.grid(alpha=0.3)
        ax.set_title('Uganda Rainfall with Dry Seasons Highlighted (orange)', fontweight='bold')
        
        # SSD rainfall
        ax = axes[1]
        ssd = rain_df.loc[subset_start:subset_end, 'South Sudan']
        ax.fill_between(ssd.index, 0, ssd, color='steelblue', alpha=0.6)
        ax.set_ylabel('SSD rainfall\n(mm/day)', color='steelblue')
        for _, row in dry_seasons_all[dry_seasons_all['country']=='South Sudan'].iterrows():
            if subset_start <= str(row['dry_start'].year) <= subset_end:
                ax.axvspan(row['dry_start'], row['dry_end'], alpha=0.3, color='orange')
        ax.grid(alpha=0.3)
        ax.set_title('South Sudan Rainfall with Dry Seasons Highlighted (orange)', fontweight='bold')
        
        # SSD floods
        ax = axes[2]
        fl = flood_daily.loc[subset_start:subset_end]
        ax.fill_between(fl.index, 1, fl['flood_pixels'].clip(lower=1),
                        color='red', alpha=0.6)
        ax.set_yscale('log')
        ax.set_ylabel('Flood pixels (log)', color='red')
        ax.set_xlabel('Date')
        # Mark dry seasons (same as SSD)
        for _, row in dry_seasons_all[dry_seasons_all['country']=='South Sudan'].iterrows():
            if subset_start <= str(row['dry_start'].year) <= subset_end:
                ax.axvspan(row['dry_start'], row['dry_end'], alpha=0.3, color='orange')
        ax.grid(alpha=0.3, which='both')
        ax.set_title('South Sudan Floods with SSD Dry Seasons Highlighted', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'upstream_dry_season_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: upstream_dry_season_timeseries.png")
        
        # Figure 3: Year-by-year scatter
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, country in enumerate(['Uganda', 'NE DRC', 'Ethiopia']):
            ax = axes[i]
            col = f'{country}_dry_end'
            if col not in annual_df.columns:
                continue
            valid = annual_df.dropna(subset=[col, 'flood_mean']).copy()
            valid[col] = pd.to_datetime(valid[col])
            valid['doy'] = valid[col].dt.dayofyear
            
            ax.scatter(valid['doy'], valid['flood_mean'], s=100, alpha=0.7, c='darkblue')
            ax.set_xlabel(f'{country} dry season end (day of year)')
            ax.set_ylabel('SSD annual flood mean (pixels)')
            ax.set_title(f'{country} dry season end → SSD floods', fontweight='bold')
            
            # Add year labels
            for _, row in valid.iterrows():
                ax.annotate(str(int(row['year'])), (row['doy'], row['flood_mean']),
                           fontsize=7, alpha=0.7)
            
            # Fit trend
            if len(valid) > 3:
                z = np.polyfit(valid['doy'], valid['flood_mean'], 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(valid['doy'].min(), valid['doy'].max(), 100)
                ax.plot(x_line, p_fit(x_line), 'r--', alpha=0.7)
                
                if HAS_SCIPY:
                    r, p_val = pearsonr(valid['doy'], valid['flood_mean'])
                    ax.text(0.05, 0.95, f'r={r:.2f}\np={p_val:.3f}',
                           transform=ax.transAxes, va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.grid(alpha=0.3)
        
        plt.suptitle('Upstream Dry Season Timing → SSD Flood Severity',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'upstream_timing_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: upstream_timing_scatter.png")
        
    except Exception as e:
        print(f"  ⚠ Plot error: {e}")
        import traceback; traceback.print_exc()

# ============================================================================
# 10. REPORT
# ============================================================================
print("\n[10] Writing report...")

report = f"""
================================================================================
UPSTREAM DRY SEASONS vs SOUTH SUDAN FLOODS
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

1. DRY SEASON CHARACTERISTICS BY COUNTRY
--------------------------------------------------------------------------------
"""
for country in REGIONS.keys():
    subset = dry_seasons_all[dry_seasons_all['country'] == country]
    if len(subset) > 0:
        report += f"{country:<15}: mean={subset['dry_duration'].mean():>5.1f}d, "
        report += f"min={subset['dry_duration'].min():>3.0f}d, "
        report += f"max={subset['dry_duration'].max():>3.0f}d, "
        report += f"n={len(subset)} years\n"

report += """
2. UPSTREAM DRY SEASON → SSD FLOODS CORRELATIONS
--------------------------------------------------------------------------------
"""
for _, row in corr_df.iterrows():
    sig = "***" if row['pearson_p'] < 0.001 else ("**" if row['pearson_p'] < 0.01 else ("*" if row['pearson_p'] < 0.05 else "ns"))
    report += f"{row['country']:<15} {row['predictor']:<15} {row['flood_metric']:<12} "
    report += f"r={row['pearson_r']:>7.3f} (p={row['pearson_p']:.3f}) {sig}\n"

report += """
3. TIMING: UPSTREAM DRY SEASON END → SSD FLOOD PEAK
--------------------------------------------------------------------------------
"""
for _, row in lag_df.iterrows():
    report += f"{row['country']:<15}: mean lag = {row['mean_lag_days']:>6.1f} days "
    report += f"(median {row['median_lag_days']:.0f}, n={row['n_years']} years)\n"

report += """
4. ANOMALY ANALYSIS (Year-to-Year Variability)
--------------------------------------------------------------------------------
"""
for _, row in anomaly_corr_df.iterrows():
    sig = "***" if row['pearson_p'] < 0.001 else ("**" if row['pearson_p'] < 0.01 else ("*" if row['pearson_p'] < 0.05 else "ns"))
    report += f"{row['country']:<15}: r={row['pearson_r']:>7.3f} (p={row['pearson_p']:.3f}) {sig}\n"

report += """
5. INTERPRETATION
--------------------------------------------------------------------------------
This analysis tests whether dry seasons in upstream countries control
flood timing and severity in South Sudan.

Three possible outcomes:
(a) SIGNIFICANT positive correlation: Longer upstream dry seasons delay
    water release, shifting flood timing and potentially reducing total floods
(b) SIGNIFICANT negative correlation: Longer dry seasons dry the basin,
    causing MORE flooding when the rains finally arrive
(c) NO correlation: Dry seasons in upstream countries have little effect
    because the flow is buffered by lakes/reservoirs

The evidence above shows which mechanism dominates.

6. PRACTICAL IMPLICATIONS
--------------------------------------------------------------------------------
If upstream dry seasons predict SSD floods:
- Monitor upstream dry season start/end dates for early warning
- Use dry season duration anomaly as a flood risk indicator
- Plan harvest window based on upstream dry season timing
- Target interventions (prepositioning) using this 2-3 month signal

================================================================================
"""

with open(STATS_DIR / 'upstream_dry_seasons_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  ✓ Report saved")

print("\n" + "="*90)
print("✅ UPSTREAM DRY SEASON ANALYSIS COMPLETE")
print("="*90)
print(f"\nResults: {OUT_DIR}")
for f in sorted(OUT_DIR.rglob('*')):
    if f.is_file():
        print(f"  - {f.relative_to(OUT_DIR)}")
print("="*90)