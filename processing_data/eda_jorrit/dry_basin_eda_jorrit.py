"""
RUNOFF EFFICIENCY ANALYSIS
Tests the "dry basin paradox": do drier years produce MORE runoff per mm of rain?

This directly validates the r=0.76 correlation between NE DRC dry season
duration and SSD floods. If dry years have higher runoff ratios, it proves
parched soils shed water more efficiently when rains finally come.

Regions: NE DRC, South Sudan, Sudan central, Uganda, West Kenya, Ethiopia
Output: results_eda_jorrit/runoff_efficiency/
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r'C:\Users\20241060\OneDrive - TU Eindhoven\JBG060\JBG060_ZHL_2026_group_12\processing_data')

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("⚠ matplotlib not available")

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
OUT_DIR = RESULTS_DIR / 'runoff_efficiency'
FIGS_DIR = OUT_DIR / 'figures'
TABLES_DIR = OUT_DIR / 'tables'
STATS_DIR = OUT_DIR / 'statistics'

for d in [FIGS_DIR, TABLES_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("="*90)
print("💧 RUNOFF EFFICIENCY ANALYSIS — Dry Basin Paradox")
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
# 1. LOAD ERA5
# ============================================================================
print("\n[1] Loading ERA5 rainfall + runoff...")

years = np.arange(2000, 2025)
era5 = load_rainfall_runoff(years)
era5_tp_mm = era5['tp'] * 1000  # m → mm
era5_ro_mm = era5['ro'] * 1000

print(f"  ✓ ERA5: {era5_tp_mm.shape}")

# ============================================================================
# 2. ANNUAL RAINFALL + RUNOFF PER REGION
# ============================================================================
print("\n[2] Computing annual rainfall, runoff, and runoff ratio per region...")

annual_rows = []
daily_series = {}

for name, box in REGIONS.items():
    lat_idx = (era5.latitude >= box['lat_min']) & (era5.latitude <= box['lat_max'])
    lon_idx = (era5.longitude >= box['lon_min']) & (era5.longitude <= box['lon_max'])

    tp = era5_tp_mm.isel(latitude=lat_idx, longitude=lon_idx).mean(
        dim=['latitude', 'longitude'])
    ro = era5_ro_mm.isel(latitude=lat_idx, longitude=lon_idx).mean(
        dim=['latitude', 'longitude'])

    tp_series = tp.to_series()
    ro_series = ro.to_series()

    daily_series[name] = pd.DataFrame({'tp': tp_series, 'ro': ro_series})

    tp_annual = tp_series.groupby(tp_series.index.year).sum()
    ro_annual = ro_series.groupby(ro_series.index.year).sum()

    for year in tp_annual.index:
        tp_v = tp_annual.loc[year]
        ro_v = ro_annual.loc[year]
        ratio = ro_v / tp_v if tp_v > 0 else np.nan

        annual_rows.append({
            'region': name,
            'year': year,
            'tp_mm': tp_v,
            'ro_mm': ro_v,
            'runoff_ratio': ratio,
        })

    print(f"  {name:<15}: mean tp={tp_annual.mean():>7.1f} mm/y, "
          f"mean ro={ro_annual.mean():>6.1f} mm/y, "
          f"mean ratio={np.nanmean([annual_rows[-1]['runoff_ratio'] for _ in range(1)]):.3f}")

runoff_df = pd.DataFrame(annual_rows)
runoff_df.to_csv(TABLES_DIR / 'annual_runoff_ratio.csv', index=False)

print(f"\n  ✓ Annual runoff: {runoff_df.shape}")

# ============================================================================
# 3. DRY-BASIN PARADOX TEST
# ============================================================================
print("\n[3] Dry-basin paradox test (annual rain vs runoff ratio)...")
print("  " + "-"*80)
print(f"  {'Region':<15} {'Pearson r':>10} {'p':>10} {'Spearman r':>12} {'p':>10}  Interpretation")
print("  " + "-"*80)

paradox_results = []
for name in REGIONS.keys():
    sub = runoff_df[runoff_df['region'] == name].dropna(
        subset=['tp_mm', 'runoff_ratio'])
    if len(sub) < 5:
        continue

    if HAS_SCIPY:
        r, p = pearsonr(sub['tp_mm'].values, sub['runoff_ratio'].values)
        rs, ps = spearmanr(sub['tp_mm'].values, sub['runoff_ratio'].values)
    else:
        r = sub['tp_mm'].corr(sub['runoff_ratio'])
        rs = sub['tp_mm'].corr(sub['runoff_ratio'], method='spearman')
        p, ps = np.nan, np.nan

    # Negative r means: dry years → higher runoff ratio → paradox confirmed
    if not np.isnan(r):
        if r < -0.3:
            interp = "✅ PARADOX CONFIRMED"
        elif r < 0:
            interp = "⚠️ weak paradox"
        elif r > 0.3:
            interp = "❌ proportional (buffered)"
        else:
            interp = "— no relationship"
    else:
        interp = "—"

    paradox_results.append({
        'region': name,
        'pearson_r': r,
        'pearson_p': p,
        'spearman_r': rs,
        'spearman_p': ps,
        'n_years': len(sub),
        'interpretation': interp,
    })

    sig = "***" if (not np.isnan(p) and p < 0.001) else \
          ("**" if (not np.isnan(p) and p < 0.01) else
           ("*" if (not np.isnan(p) and p < 0.05) else ""))
    print(f"  {name:<15} {r:>10.3f} {p:>10.4f} {rs:>12.3f} {ps:>10.4f}  {interp} {sig}")

paradox_df = pd.DataFrame(paradox_results)
paradox_df.to_csv(TABLES_DIR / 'dry_basin_paradox_test.csv', index=False)

# ============================================================================
# 4. LINK RUNOFF RATIO TO SSD FLOODS
# ============================================================================
print("\n[4] Link upstream runoff ratio → SSD floods...")

# Load SSD floods
flood_bbox = {'lat_min': 3.0, 'lat_max': 13.0, 'lon_min': 24.0, 'lon_max': 36.0}
flood_df = load_flood_masks(years, bbox=flood_bbox)
flood_daily = flood_df.groupby('date').size().reset_index()
flood_daily.columns = ['date', 'flood_pixels']
flood_daily = flood_daily.set_index('date').sort_index()

annual_flood = flood_daily.groupby(flood_daily.index.year).agg(
    flood_mean=('flood_pixels', 'mean'),
    flood_max=('flood_pixels', 'max'),
).reset_index().rename(columns={'date': 'year'})

# Merge runoff ratios
merge_df = annual_flood.copy()
for name in REGIONS.keys():
    sub = runoff_df[runoff_df['region'] == name][['year', 'runoff_ratio']]
    merge_df = merge_df.merge(
        sub.rename(columns={'runoff_ratio': f'{name}_ratio'}),
        on='year', how='left')

merge_df.to_csv(TABLES_DIR / 'runoff_ratio_vs_floods.csv', index=False)

print("\n  Correlating upstream runoff ratio → SSD floods:")
print(f"  {'Region':<15} {'Ratio vs flood_mean':>25} {'Ratio vs flood_max':>25}")
print("  " + "-"*70)

link_results = []
for name in REGIONS.keys():
    col = f'{name}_ratio'
    if col not in merge_df.columns:
        continue
    sub = merge_df.dropna(subset=[col, 'flood_mean'])
    if len(sub) < 5:
        continue

    if HAS_SCIPY:
        r1, p1 = pearsonr(sub[col].values, sub['flood_mean'].values)
        r2, p2 = pearsonr(sub[col].values, sub['flood_max'].values)
    else:
        r1 = sub[col].corr(sub['flood_mean']); p1 = np.nan
        r2 = sub[col].corr(sub['flood_max']);  p2 = np.nan

    link_results.append({
        'region': name,
        'r_mean': r1, 'p_mean': p1,
        'r_max': r2, 'p_max': p2,
        'n_years': len(sub),
    })

    s1 = "*" if (not np.isnan(p1) and p1 < 0.05) else ""
    s2 = "*" if (not np.isnan(p2) and p2 < 0.05) else ""
    print(f"  {name:<15} r={r1:>7.3f} (p={p1:.3f}){s1:<2} "
          f"r={r2:>7.3f} (p={p2:.3f}){s2}")

link_df = pd.DataFrame(link_results)
link_df.to_csv(TABLES_DIR / 'runoff_ratio_flood_correlations.csv', index=False)

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
if HAS_PLT:
    print("\n[5] Creating visualizations...")

    try:
        # ---- Figure 1: Rain vs runoff ratio scatter per region ----
        n_regions = len(REGIONS)
        ncols = 3
        nrows = (n_regions + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5*nrows))
        axes = axes.flatten()

        for i, name in enumerate(REGIONS.keys()):
            ax = axes[i]
            sub = runoff_df[runoff_df['region'] == name].dropna(
                subset=['tp_mm', 'runoff_ratio'])
            if len(sub) == 0:
                ax.text(0.5, 0.5, f'{name}\nno data', transform=ax.transAxes,
                        ha='center', va='center')
                continue

            row = paradox_df[paradox_df['region'] == name]
            r_val = row['pearson_r'].values[0] if len(row) > 0 else np.nan
            p_val = row['pearson_p'].values[0] if len(row) > 0 else np.nan

            sc = ax.scatter(sub['tp_mm'], sub['runoff_ratio'],
                            c=sub['year'], cmap='viridis',
                            s=100, alpha=0.8, edgecolor='black')
            for _, row_s in sub.iterrows():
                ax.annotate(str(int(row_s['year'])),
                            (row_s['tp_mm'], row_s['runoff_ratio']),
                            fontsize=6, ha='center', va='center')

            # Trend line
            if len(sub) > 3:
                z = np.polyfit(sub['tp_mm'].values, sub['runoff_ratio'].values, 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(sub['tp_mm'].min(), sub['tp_mm'].max(), 50)
                ax.plot(x_line, p_fit(x_line), 'r--', linewidth=1.5, alpha=0.7)

            # Annotation box
            if not np.isnan(r_val):
                box_color = 'lightgreen' if r_val < -0.3 else (
                    'lightyellow' if r_val < 0 else 'lightcoral')
                ax.text(0.05, 0.95,
                        f'r={r_val:.3f}\np={p_val:.3f}\nn={len(sub)}',
                        transform=ax.transAxes, va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor=box_color,
                                  edgecolor='black'))

            ax.set_xlabel('Annual rainfall (mm/year)')
            ax.set_ylabel('Runoff ratio (ro/tp)')
            ax.set_title(name, fontweight='bold')
            ax.grid(alpha=0.3)

        # Hide empty subplots
        for j in range(len(REGIONS), len(axes)):
            axes[j].axis('off')

        plt.suptitle('Dry-Basin Paradox: Do Drier Years Produce More Runoff?',
                     fontsize=15, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'dry_basin_paradox_scatter.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: dry_basin_paradox_scatter.png")

        # ---- Figure 2: Time series of runoff ratio per region ----
        fig, ax = plt.subplots(figsize=(16, 7))
        for name in REGIONS.keys():
            sub = runoff_df[runoff_df['region'] == name].sort_values('year')
            if len(sub) > 0:
                ax.plot(sub['year'], sub['runoff_ratio'],
                        'o-', linewidth=2, markersize=6, label=name)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Runoff ratio (ro / tp)', fontsize=12)
        ax.set_title('Annual Runoff Ratio by Region (2000-2024)',
                     fontweight='bold', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'runoff_ratio_timeseries.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: runoff_ratio_timeseries.png")

        # ---- Figure 3: Runoff ratio anomaly → flood anomaly ----
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Panel A: NE DRC
        ax = axes[0]
        if 'NE DRC_ratio' in merge_df.columns:
            sub = merge_df.dropna(subset=['NE DRC_ratio', 'flood_mean']).copy()
            sub['ratio_anom'] = ((sub['NE DRC_ratio'] - sub['NE DRC_ratio'].mean())
                                 / sub['NE DRC_ratio'].std())
            sub['flood_anom'] = ((sub['flood_mean'] - sub['flood_mean'].mean())
                                 / sub['flood_mean'].std())
            ax.scatter(sub['ratio_anom'], sub['flood_anom'],
                       s=120, alpha=0.7, c=sub['year'], cmap='viridis',
                       edgecolor='black')
            for _, row in sub.iterrows():
                ax.annotate(str(int(row['year'])),
                            (row['ratio_anom'], row['flood_anom']),
                            fontsize=7, ha='center', va='center')
            if len(sub) > 3:
                z = np.polyfit(sub['ratio_anom'].values, sub['flood_anom'].values, 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(sub['ratio_anom'].min(),
                                     sub['ratio_anom'].max(), 50)
                ax.plot(x_line, p_fit(x_line), 'r--', linewidth=2)
                if HAS_SCIPY:
                    r, p = pearsonr(sub['ratio_anom'].values, sub['flood_anom'].values)
                    ax.text(0.05, 0.95, f'r={r:.3f}\np={p:.3f}',
                            transform=ax.transAxes, va='top', fontsize=11,
                            bbox=dict(boxstyle='round', facecolor='lightyellow'))
            ax.set_xlabel('NE DRC runoff ratio anomaly (z)')
            ax.set_ylabel('SSD flood anomaly (z)')
            ax.set_title('NE DRC Runoff Ratio → SSD Floods',
                         fontweight='bold', fontsize=12)
            ax.grid(alpha=0.3)

        # Panel B: Comparison bar chart of correlations
        ax = axes[1]
        if len(link_df) > 0:
            link_sorted = link_df.sort_values('r_mean')
            colors = ['red' if r > 0 else 'blue' for r in link_sorted['r_mean']]
            y_pos = np.arange(len(link_sorted))
            ax.barh(y_pos, link_sorted['r_mean'], color=colors, alpha=0.7,
                    edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(link_sorted['region'])
            ax.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Pearson r (runoff ratio → flood_mean)')
            ax.set_title('Runoff Ratio → SSD Floods\nBy region',
                         fontweight='bold', fontsize=12)
            ax.grid(alpha=0.3, axis='x')

            for i, (_, row) in enumerate(link_sorted.iterrows()):
                sig = "*" if (not np.isnan(row['p_mean']) and row['p_mean'] < 0.05) else ""
                offset = 0.03 * np.sign(row['r_mean']) if row['r_mean'] != 0 else 0.03
                ax.text(row['r_mean'] + offset, i,
                        f"{row['r_mean']:.2f}{sig}", va='center', fontsize=10)

        plt.suptitle('Runoff Efficiency as an Upstream Flood Driver',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'runoff_ratio_flood_link.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: runoff_ratio_flood_link.png")

    except Exception as e:
        print(f"  ⚠ Plot error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 6. REPORT
# ============================================================================
print("\n[6] Writing report...")

report = f"""
================================================================================
RUNOFF EFFICIENCY ANALYSIS — DRY BASIN PARADOX
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

1. HYPOTHESIS
--------------------------------------------------------------------------------
If dry years produce MORE runoff per mm of rain (higher runoff ratio), then
parched soils shed water efficiently when rains return — this is the "dry
basin paradox". This would explain why NE DRC dry season duration correlates
with SSD floods at r=0.76.

2. ANNUAL RUNOFF RATIO BY REGION
--------------------------------------------------------------------------------
Mean runoff ratios (ro/tp) across 2000-2024:
"""
for name in REGIONS.keys():
    sub = runoff_df[runoff_df['region'] == name].dropna(subset=['runoff_ratio'])
    if len(sub) > 0:
        report += f"  {name:<15}: {sub['runoff_ratio'].mean():.4f} " \
                  f"(std {sub['runoff_ratio'].std():.4f}, " \
                  f"n={len(sub)})\n"

report += """
3. DRY-BASIN PARADOX TEST
--------------------------------------------------------------------------------
Correlation between annual rainfall and runoff ratio per region.
NEGATIVE r = dry years have higher runoff ratio = paradox confirmed.
"""
for _, row in paradox_df.iterrows():
    sig = "***" if (not np.isnan(row['pearson_p']) and row['pearson_p'] < 0.001) \
        else ("**" if (not np.isnan(row['pearson_p']) and row['pearson_p'] < 0.01) \
        else ("*" if (not np.isnan(row['pearson_p']) and row['pearson_p'] < 0.05) else "ns"))
    report += f"  {row['region']:<15}: r={row['pearson_r']:>7.3f} " \
              f"(p={row['pearson_p']:.3f}) {sig}  {row['interpretation']}\n"

report += """
4. RUNOFF RATIO → SSD FLOODS
--------------------------------------------------------------------------------
Does upstream runoff efficiency predict SSD flood extent?
"""
for _, row in link_df.iterrows():
    s1 = "*" if (not np.isnan(row['p_mean']) and row['p_mean'] < 0.05) else ""
    s2 = "*" if (not np.isnan(row['p_max']) and row['p_max'] < 0.05) else ""
    report += (f"  {row['region']:<15}: "
               f"flood_mean r={row['r_mean']:>7.3f} (p={row['p_mean']:.3f}){s1}, "
               f"flood_max r={row['r_max']:>7.3f} (p={row['p_max']:.3f}){s2}\n")

report += """
5. INTERPRETATION
--------------------------------------------------------------------------------
If NE DRC or Sudan central show:
  - NEGATIVE correlation between rainfall and runoff ratio (paradox confirmed)
  - POSITIVE correlation between runoff ratio and SSD floods

then the full mechanism chain is:

  Dry year in NE DRC
    → parched soils, low vegetation cover
    → higher runoff ratio when rains arrive
    → more water enters Bahr el Ghazal
    → Sudd wetland expands
    → SSD floods increase

This is a stronger, more mechanistic version of the earlier finding that
just dry season DURATION correlates with floods.

================================================================================
"""

with open(STATS_DIR / 'runoff_efficiency_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  ✓ Report saved")

print("\n" + "="*90)
print("✅ RUNOFF EFFICIENCY ANALYSIS COMPLETE")
print("="*90)
print(f"\nResults: {OUT_DIR}")
for f in sorted(OUT_DIR.rglob('*')):
    if f.is_file():
        print(f"  - {f.relative_to(OUT_DIR)}")
print("="*90)