"""
LAKES -> SUDD -> UNITY / WARRAP  and  BAHR EL ARAB RAIN -> WARRAP
=================================================================
Tests (and visualises) the hypothesis that floods in Unity and Warrap are driven by
  (a) Lake Victoria / Kyoga / Albert levels (via the Sudd), and
  (b) rain over the Bahr el Arab catchment.

Steps
  1. Daily flood extent per state from MODIS (recurring + unusual = ALL flood pixels)
  2. Remove seasonality (+ trend) -> anomalies; partial correlation given day-of-year
  3. Lake -> flood lag correlations, 1-6 months, for Unity and Warrap
  4. Annual analysis: peak flood extent vs prior-season lake level (2019/2020 highlighted)
  5. Bahr el Arab catchment rain -> Warrap flood, lags 2-8 weeks
  6. Ablation with leave-one-year-out CV (history+season | +lakes | +rain | +both)

!! Look at the CONFIG block: a few names/paths are assumptions you must check.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import geopandas as gpd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG  (edit the items marked TODO)
# ============================================================================
PROC_DIR = r'C:\Users\20241060\OneDrive - TU Eindhoven\JBG060\JBG060_ZHL_2026_group_12\processing_data'
sys.path.append(PROC_DIR)
from loading import load_rainfall_runoff, load_flood_masks   # same as your EDA script

import importlib
from loading import load_lake_stations

# TODO 1: the loaders use relative paths ('./raw_data/...'), so the working directory must be the
# folder that CONTAINS raw_data. Set it here if you don't launch the script from there.
DATA_ROOT = None            # e.g. r'C:\...\JBG060_ZHL_2026_group_12'
if DATA_ROOT:
    import os
    os.chdir(DATA_ROOT)

# TODO 2: name of the module (file without .py) holding load_admin_boundaries() (your impact loader)
IMPACT_MODULE = None        # optional override: file name without .py
if IMPACT_MODULE is None:
    for _f in sorted(Path(PROC_DIR).glob('*.py')):
        if 'def load_admin_boundaries' in _f.read_text(encoding='utf-8', errors='ignore'):
            IMPACT_MODULE = _f.stem
            break
    else:
        raise RuntimeError(f"No .py file with load_admin_boundaries found in {PROC_DIR}. "
                           "Set IMPACT_MODULE manually.")
print(f"  admin loader found in module: {IMPACT_MODULE}")
load_admin_boundaries = importlib.import_module(IMPACT_MODULE).load_admin_boundaries
NAMECOL = 'adm1_name'       # column name used in your loader (see locate_coordinate)


def load_lakes():
    """Wrap load_lake_stations() into one DataFrame: date index, columns Victoria / Kyoga / Albert."""
    raw = load_lake_stations()
    out = {}
    for key, df in raw.items():
        name = key.capitalize()                       # 'victoria' -> 'Victoria'
        if 'height_egm2008' in df.columns:            # Victoria / Kyoga text files
            col = 'height_egm2008'
        else:                                         # Albert netCDF (DAHITI): find the level column
            exact = [c for c in ['water_level', 'water_surface_height', 'height'] if c in df.columns]
            cands = [c for c in df.columns if any(k in str(c).lower() for k in ('level', 'height'))
                     and not any(k in str(c).lower() for k in ('err', 'uncert', 'sigma'))]
            col = (exact or cands or list(df.select_dtypes('number').columns))[0]
        print(f"  lake {name}: using column '{col}'  <-- check this is the water level in metres")
        s = df[col].astype(float)
        s.index = pd.to_datetime(s.index)
        out[name] = s.groupby(level=0).mean()         # several missions can give >1 value per day
    return pd.DataFrame(out).sort_index()

STATES = ['Unity', 'Warrap']
YEARS = np.arange(2000, 2025)

# Approximate Bahr el Arab (Kiir) catchment: S. Darfur / S. Kordofan -> N. Bahr el Ghazal/Warrap.
# TODO 3 (better): replace with a HydroBASINS / HydroSHEDS polygon average.
BAHR_EL_ARAB_BOX = {'lat_min': 8.5, 'lat_max': 12.0, 'lon_min': 24.0, 'lon_max': 30.0}

HIGHLIGHT_YEARS = [2019, 2020]
ANNUAL_LAKE_WINDOW = ((-1, 10), (0, 6))   # previous Oct -> current Jun = "prior season"
LEADS_WEEKS = 4                            # forecast horizon for the ablation
LAKE_LAGS_MONTHS = range(0, 7)             # 0..6, focus on 1-6
RAIN_LAGS_WEEKS = range(0, 11)             # 0..10, focus on 2-8

OUT_DIR = Path(PROC_DIR) / 'results_eda_jorrit' / 'lakes_sudd_analysis'
FIGS, TABLES = OUT_DIR / 'figures', OUT_DIR / 'tables'
for d in (FIGS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

LAKE_COLORS = {'Victoria': '#1f77b4', 'Kyoga': '#2ca02c', 'Albert': '#d62728'}

print("=" * 90)
print("LAKES -> SUDD -> UNITY/WARRAP + BAHR EL ARAB RAIN")
print("=" * 90)


# ============================================================================
# HELPERS
# ============================================================================
def harmonics(index, k=2):
    doy = index.dayofyear.values
    out = {}
    for i in range(1, k + 1):
        out[f'sin{i}'] = np.sin(2 * np.pi * i * doy / 365.25)
        out[f'cos{i}'] = np.cos(2 * np.pi * i * doy / 365.25)
    return pd.DataFrame(out, index=index)


def deseasonalise(s, k=3):
    """Remove smooth climatology (harmonic regression on day of year).
    Correlating two deseasonalised series == partial correlation controlling for day of year."""
    s = s.astype(float)
    X = np.column_stack([np.ones(len(s)), harmonics(s.index, k).values])
    m = s.notna().values
    beta = np.linalg.lstsq(X[m], s.values[m], rcond=None)[0]
    return pd.Series(s.values - X @ beta, index=s.index)


def detrend(s):
    s = s.astype(float)
    t = np.arange(len(s))
    m = s.notna().values
    p = np.polyfit(t[m], s.values[m], 1)
    return s - np.polyval(p, t)


def anomalies(s):
    return detrend(deseasonalise(s))


VARIANTS = {
    'Raw': lambda s: s,
    'Partial | day-of-year': deseasonalise,
    'Anomaly (deseason + detrend)': anomalies,
}


def lag_corr(x, y, lag):
    """Correlation of x(t-lag) with y(t) + autocorrelation-adjusted p-value (Bretherton n_eff)."""
    d = pd.concat([x.shift(lag), y], axis=1).dropna()
    n = len(d)
    if n < 10:
        return np.nan, np.nan, n
    r = d.iloc[:, 0].corr(d.iloc[:, 1])
    r1x, r1y = d.iloc[:, 0].autocorr(1), d.iloc[:, 1].autocorr(1)
    neff = n * (1 - r1x * r1y) / (1 + r1x * r1y)
    neff = float(np.clip(neff, 3, n))
    tval = r * np.sqrt((neff - 2) / max(1 - r ** 2, 1e-9))
    p = 2 * stats.t.sf(abs(tval), neff - 2)
    return r, p, neff


def lag_table(x, y, lags, transform):
    xs, ys = transform(x), transform(y)
    rows = [dict(lag=l, **dict(zip(['r', 'p', 'n_eff'], lag_corr(xs, ys, l)))) for l in lags]
    return pd.DataFrame(rows).set_index('lag')


# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading ERA5, MODIS floods, lakes, admin boundaries...")

era5 = load_rainfall_runoff(YEARS)
tp_mm = era5['tp'] * 1000


def box_rain(box):
    li = (era5.latitude >= box['lat_min']) & (era5.latitude <= box['lat_max'])
    lo = (era5.longitude >= box['lon_min']) & (era5.longitude <= box['lon_max'])
    s = tp_mm.isel(latitude=li, longitude=lo).mean(dim=['latitude', 'longitude']).to_series()
    s.index = pd.to_datetime(s.index)
    return s


admin, _admin2 = load_admin_boundaries()
admin = admin.to_crs('EPSG:4326')
admin[NAMECOL] = admin[NAMECOL].astype(str).str.strip()
print("  admin1 names:", sorted(admin[NAMECOL].unique()))

rain_bea = box_rain(BAHR_EL_ARAB_BOX)
mnx, mny, mxx, mxy = admin.loc[admin[NAMECOL] == 'Warrap'].total_bounds
rain_warrap_local = box_rain({'lat_min': mny, 'lat_max': mxy, 'lon_min': mnx, 'lon_max': mxx})

flood_df = load_flood_masks(YEARS, bbox={'lat_min': 3.0, 'lat_max': 13.0, 'lon_min': 24.0, 'lon_max': 36.0})
flood_df['date'] = pd.to_datetime(flood_df['date'])
latc = next(c for c in ['lat', 'latitude', 'y'] if c in flood_df.columns)
lonc = next(c for c in ['lon', 'longitude', 'x'] if c in flood_df.columns)

# Assign each unique pixel to a state ONCE (fast), then count pixels per day per state.
pix = flood_df[[latc, lonc]].drop_duplicates().reset_index(drop=True)
pts = gpd.GeoDataFrame(pix, geometry=gpd.points_from_xy(pix[lonc], pix[latc]), crs='EPSG:4326')
joined = gpd.sjoin(pts, admin[[NAMECOL, 'geometry']], how='left', predicate='within')
joined = joined[~joined.index.duplicated()]
pix['state'] = joined[NAMECOL].values
flood_df = flood_df.merge(pix[[latc, lonc, 'state']], on=[latc, lonc], how='left')

# ALL flood pixels = recurring + unusual. Days with no detection -> 0 (also = cloud/no pass; see notes).
counts = flood_df[flood_df['state'].isin(STATES)].groupby(['date', 'state']).size().unstack(fill_value=0)
full_idx = pd.date_range(counts.index.min(), counts.index.max(), freq='D')
flood_daily = counts.reindex(full_idx, fill_value=0)[STATES]

lakes = load_lakes()
if 'date' in lakes.columns:
    lakes = lakes.set_index('date')
lakes.index = pd.to_datetime(lakes.index)
lakes = lakes.rename(columns=lambda c: next((k for k in LAKE_COLORS if k.lower() in str(c).lower()), c))
lakes = lakes[[c for c in LAKE_COLORS if c in lakes.columns]].sort_index()
# Causal daily series: forward-fill the ~10-day readings (no future values leaked)
lakes_daily = lakes.resample('D').last().ffill(limit=15)
lakes_daily = lakes_daily.loc[lakes_daily.index >= flood_daily.index.min()]

print(f"  floods: {flood_daily.index.min().date()} -> {flood_daily.index.max().date()}")
print(f"  lakes : {list(lakes_daily.columns)}  {lakes_daily.index.min().date()} -> {lakes_daily.index.max().date()}")
print("  total flood pixel-days:", flood_daily.sum().to_dict())

# Monthly series (log1p flood extent = less skewed, more robust to MODIS noise)
flood_m = np.log1p(flood_daily.resample('MS').mean())
lakes_m = lakes_daily.resample('MS').mean()
common_m = flood_m.index.intersection(lakes_m.dropna(how='all').index)
flood_m, lakes_m = flood_m.loc[common_m], lakes_m.loc[common_m]


# ============================================================================
# 2. FIGURE 1 - what "remove seasonality and trend" does
# ============================================================================
print("\n[2] Figure 1: raw -> deseasonalised -> anomaly")

fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
for j, (name, series, color) in enumerate([('Lake Albert level (m)', lakes_m['Albert'], LAKE_COLORS['Albert']),
                                           ('Unity flood extent (log1p pixels)', flood_m['Unity'], 'navy')]):
    for i, (vname, fn) in enumerate(VARIANTS.items()):
        ax = axes[i, j]
        ax.plot(fn(series), color=color, lw=1.4)
        if i > 0:
            ax.axhline(0, color='k', lw=0.6)
        ax.axvspan(pd.Timestamp('2019-01-01'), pd.Timestamp('2020-12-31'), color='orange', alpha=0.15)
        ax.set_title(f'{name}: {vname}', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
fig.suptitle('Removing seasonality and trend (orange = 2019-2020)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGS / 'fig1_anomaly_decomposition.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. LAKE -> FLOOD LAGS, 1-6 MONTHS
# ============================================================================
print("\n[3] Lake -> flood lag correlations (0-6 months)")

lag_rows = []
fig, axes = plt.subplots(len(STATES), 3, figsize=(17, 8), sharey=True)
for i, st in enumerate(STATES):
    for j, (vname, fn) in enumerate(VARIANTS.items()):
        ax = axes[i, j]
        for lake in lakes_m.columns:
            t = lag_table(lakes_m[lake], flood_m[st], LAKE_LAGS_MONTHS, fn)
            ax.plot(t.index, t['r'], '-o', color=LAKE_COLORS[lake], label=lake, lw=1.8, ms=5)
            sig = t[t['p'] < 0.05]
            ax.scatter(sig.index, sig['r'], s=110, facecolors='none', edgecolors='k', zorder=5)
            if t['r'].notna().any():
                pk = t['r'].abs().idxmax()
                ax.axvline(pk, color=LAKE_COLORS[lake], ls=':', alpha=0.5)
                lag_rows.append(dict(state=st, variant=vname, lake=lake, peak_lag_months=pk,
                                     peak_r=t.loc[pk, 'r'], p_adj=t.loc[pk, 'p'], n_eff=t.loc[pk, 'n_eff']))
        ax.axhline(0, color='k', lw=0.6)
        ax.axvspan(1, 6, color='grey', alpha=0.08)
        ax.set_title(f'{st} - {vname}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Lag (months, lake leads flood)')
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel('Pearson r')
        if i == 0 and j == 0:
            ax.legend(title='circled = p<0.05 (n_eff adj.)', fontsize=8)
fig.suptitle('Lake level -> flood extent: does the correlation survive removing season & trend?\n'
             '(dotted line = peak lag; compare with Sudd travel time of ~weeks-months)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGS / 'fig2_lake_lag_correlations.png', dpi=200, bbox_inches='tight')
plt.close()
pd.DataFrame(lag_rows).to_csv(TABLES / 'lake_lag_peaks.csv', index=False)

# ============================================================================
# 4. ANNUAL ANALYSIS
# ============================================================================
print("\n[4] Annual peak flood vs prior-season lake level")

peak = np.log1p(flood_daily.rolling(7, min_periods=4).mean()).groupby(flood_daily.index.year).max()
(m0, d0), (m1, d1) = ANNUAL_LAKE_WINDOW[0], ANNUAL_LAKE_WINDOW[1]


def prior_season_level(lake, year):
    a = pd.Timestamp(year - 1, ANNUAL_LAKE_WINDOW[0][1], 1)
    b = pd.Timestamp(year, ANNUAL_LAKE_WINDOW[1][1], 28)
    return lakes_daily[lake].loc[a:b].mean()


annual_lakes = ['Victoria', 'Albert']
annual_rows = []
fig, axes = plt.subplots(len(STATES), len(annual_lakes), figsize=(13, 10))
for i, st in enumerate(STATES):
    for j, lake in enumerate(annual_lakes):
        ax = axes[i, j]
        yrs = [y for y in peak.index if y in range(2003, 2025)]
        x = np.array([prior_season_level(lake, y) for y in yrs])
        y_ = peak.loc[yrs, st].values
        ok = ~np.isnan(x)
        x, y_, yy = x[ok], y_[ok], np.array(yrs)[ok]
        lr = stats.linregress(x, y_)
        # leave-one-year-out R2
        preds = [np.polyval(np.polyfit(np.delete(x, k), np.delete(y_, k), 1), x[k]) for k in range(len(x))]
        loyo_r2 = 1 - np.sum((y_ - preds) ** 2) / np.sum((y_ - y_.mean()) ** 2)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, lr.intercept + lr.slope * xs, 'k--', lw=1)
        hl = np.isin(yy, HIGHLIGHT_YEARS)
        ax.scatter(x[~hl], y_[~hl], color=LAKE_COLORS[lake], s=50)
        ax.scatter(x[hl], y_[hl], color='red', s=90, edgecolor='k', zorder=5)
        for xi, yi, yr in zip(x, y_, yy):
            ax.annotate(str(yr), (xi, yi), fontsize=7, xytext=(3, 3), textcoords='offset points')
        ax.set_title(f'{st}: peak flood vs {lake} (prior Oct-Jun)\n'
                     f'r={lr.rvalue:.2f}, p={lr.pvalue:.3f}, LOYO R2={loyo_r2:.2f}, n={len(x)}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel(f'{lake} mean level (m)')
        ax.set_ylabel('Peak flood extent (log1p pixels)')
        ax.grid(alpha=0.3)
        annual_rows.append(dict(state=st, lake=lake, r=lr.rvalue, p=lr.pvalue, slope=lr.slope,
                                loyo_r2=loyo_r2, n=len(x)))
fig.suptitle('Annual analysis (red = 2019 / 2020)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGS / 'fig3_annual_peak_vs_lake.png', dpi=200, bbox_inches='tight')
plt.close()
pd.DataFrame(annual_rows).to_csv(TABLES / 'annual_regression.csv', index=False)

# ============================================================================
# 5. BAHR EL ARAB RAIN -> WARRAP, LAGS 2-8 WEEKS
# ============================================================================
print("\n[5] Bahr el Arab catchment rain -> Warrap flood (2-8 weeks)")

flood_w = np.log1p(flood_daily['Warrap'].resample('W').mean())
rain_w = {'Bahr el Arab catchment': rain_bea.resample('W').mean(),
          'Warrap local (ERA5 box)': rain_warrap_local.resample('W').mean()}
rcol = {'Bahr el Arab catchment': 'teal', 'Warrap local (ERA5 box)': 'grey'}

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
rain_rows = []
for j, (vname, fn) in enumerate(VARIANTS.items()):
    ax = axes[j]
    for nm, rs in rain_w.items():
        t = lag_table(rs, flood_w, RAIN_LAGS_WEEKS, fn)
        ax.plot(t.index, t['r'], '-o', color=rcol[nm], label=nm, lw=1.8)
        sig = t[t['p'] < 0.05]
        ax.scatter(sig.index, sig['r'], s=110, facecolors='none', edgecolors='k', zorder=5)
        for l, row in t.iterrows():
            rain_rows.append(dict(variant=vname, predictor=nm, lag_weeks=l, r=row['r'], p_adj=row['p']))
    ax.axvspan(2, 8, color='grey', alpha=0.12)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_title(f'Warrap flood - {vname}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Lag (weeks, rain leads flood)')
    ax.grid(alpha=0.3)
    if j == 0:
        ax.set_ylabel('Pearson r')
        ax.legend(title='circled = p<0.05 (n_eff adj.)', fontsize=8)
fig.suptitle('Rain over the Bahr el Arab catchment vs local rain (shaded: 2-8 weeks)', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGS / 'fig4_bahr_el_arab_rain_lags.png', dpi=200, bbox_inches='tight')
plt.close()
pd.DataFrame(rain_rows).to_csv(TABLES / 'bahr_el_arab_rain_lags.csv', index=False)

# ============================================================================
# 6. ABLATION WITH LEAVE-ONE-YEAR-OUT CV  (weekly, horizon = LEADS_WEEKS)
# ============================================================================
print(f"\n[6] Ablation, leave-one-year-out, horizon {LEADS_WEEKS} weeks")


def make_dataset(state):
    f = np.log1p(flood_daily[state].rolling(7, min_periods=4).mean()).resample('W').last()
    lk = lakes_daily.resample('W').last()
    ra = rain_bea.resample('W').mean()
    X = pd.DataFrame(index=f.index)
    X['flood_t'], X['flood_t-4w'] = f, f.shift(4)
    h = harmonics(f.index, 2)
    season = list(h.columns)
    X[season] = h
    hist = ['flood_t', 'flood_t-4w'] + season
    lake_cols = []
    for c in lk.columns:
        X[f'{c}_lvl'] = lk[c].reindex(f.index)
        X[f'{c}_d13w'] = X[f'{c}_lvl'] - X[f'{c}_lvl'].shift(13)
        lake_cols += [f'{c}_lvl', f'{c}_d13w']
    rain_cols = []
    for w in (4, 8, 12):
        X[f'bea_r{w}w'] = ra.reindex(f.index).rolling(w).mean()
        rain_cols.append(f'bea_r{w}w')
    y = f.shift(-LEADS_WEEKS)
    data = pd.concat([X, y.rename('y')], axis=1).dropna()
    sets = {'Climatology (season only)': season,
            'A: history + season': hist,
            'B: A + lakes': hist + lake_cols,
            'C: A + Bahr el Arab rain': hist + rain_cols,
            'D: A + lakes + rain': hist + lake_cols + rain_cols}
    return data, sets


MODELS = {
    'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    'GradBoost': lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                       max_iter=200, random_state=0),
}

abl_rows, preds_store = [], {}
logo = LeaveOneGroupOut()
for st in STATES:
    data, sets = make_dataset(st)
    y = data['y'].values
    groups = data.index.year.values
    # persistence baseline
    persist = data['flood_t'].values
    abl_rows.append(dict(state=st, model='-', features='Persistence',
                         rmse=np.sqrt(np.mean((y - persist) ** 2)),
                         r2=1 - np.sum((y - persist) ** 2) / np.sum((y - y.mean()) ** 2)))
    for mname, mk in MODELS.items():
        fold_err = {}
        for fname, cols in sets.items():
            pred = np.zeros(len(y))
            for tr, te in logo.split(data, y, groups):
                mdl = mk().fit(data.iloc[tr][cols], y[tr])
                pred[te] = mdl.predict(data.iloc[te][cols])
            preds_store[(st, mname, fname)] = pd.Series(pred, index=data.index)
            err = (y - pred) ** 2
            fold_err[fname] = pd.Series(err).groupby(groups).mean()
            abl_rows.append(dict(state=st, model=mname, features=fname,
                                 rmse=np.sqrt(err.mean()),
                                 r2=1 - err.sum() / np.sum((y - y.mean()) ** 2)))
        base = fold_err['A: history + season']
        for fname in sets:
            if fname.startswith(('B', 'C', 'D')):
                wins = int((fold_err[fname].values < base.values).sum())
                abl_rows[[i for i, r in enumerate(abl_rows) if r['state'] == st and r['model'] == mname
                          and r['features'] == fname][0]]['years_beating_A'] = f"{wins}/{len(base)}"

abl = pd.DataFrame(abl_rows)
for (st, mn), g in abl[abl.model != '-'].groupby(['state', 'model']):
    rA = g.loc[g.features == 'A: history + season', 'rmse'].iloc[0]
    abl.loc[g.index, 'skill_vs_A_%'] = 100 * (1 - (g['rmse'] / rA) ** 2)
abl.to_csv(TABLES / 'ablation_loyo.csv', index=False)
print(abl.round(3).to_string(index=False))

# Fig 5: RMSE bars
fig, axes = plt.subplots(len(STATES), len(MODELS), figsize=(15, 9), sharex=False)
for i, st in enumerate(STATES):
    for j, mn in enumerate(MODELS):
        ax = axes[i, j]
        g = abl[(abl.state == st) & (abl.model == mn)]
        pers = abl[(abl.state == st) & (abl.features == 'Persistence')]['rmse'].iloc[0]
        cols = ['#bbbbbb', '#888888', '#d62728', '#1f77b4', '#6a3d9a']
        ax.bar(range(len(g)), g['rmse'], color=cols)
        ax.axhline(pers, color='k', ls='--', lw=1, label='Persistence')
        for k, (_, r) in enumerate(g.iterrows()):
            ax.text(k, r['rmse'], f"{r['r2']:.2f}", ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([f.split(':')[0] if ':' in f else 'Clim.' for f in g['features']])
        ax.set_title(f'{st} - {mn}  (bar labels = LOYO R2)', fontsize=10, fontweight='bold')
        ax.set_ylabel('LOYO RMSE (log1p pixels)')
        ax.grid(alpha=0.3, axis='y')
        if i == 0 and j == 0:
            ax.legend()
fig.suptitle(f'Ablation: does adding lakes / Bahr el Arab rain improve skill? '
             f'(leave-one-year-out, {LEADS_WEEKS}-week lead)\n'
             'A=history+season, B=+lakes, C=+Bahr el Arab rain, D=+both', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGS / 'fig5_ablation_rmse.png', dpi=200, bbox_inches='tight')
plt.close()

# Fig 6: observed vs predicted around 2019-2021
fig, axes = plt.subplots(len(STATES), 1, figsize=(15, 8))
for i, st in enumerate(STATES):
    data, _ = make_dataset(st)
    ax = axes[i]
    rng = slice('2017-01-01', '2022-12-31')
    ax.plot(data['y'].loc[rng], 'k', lw=2, label=f'Observed (t+{LEADS_WEEKS}w)')
    ax.plot(preds_store[(st, 'Ridge', 'A: history + season')].loc[rng], color='#d62728', label='Ridge A')
    ax.plot(preds_store[(st, 'Ridge', 'D: A + lakes + rain')].loc[rng], color='#6a3d9a', label='Ridge D')
    ax.axvspan(pd.Timestamp('2019-01-01'), pd.Timestamp('2020-12-31'), color='orange', alpha=0.12)
    ax.set_title(f'{st}: out-of-sample predictions (each year held out)', fontweight='bold')
    ax.set_ylabel('log1p flood pixels')
    ax.grid(alpha=0.3)
    ax.legend(ncol=3)
plt.tight_layout()
plt.savefig(FIGS / 'fig6_predictions_2019_2020.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. REPORT + LITERATURE
# ============================================================================
print("\n[7] Writing notes...")
notes = f"""
LAKES -> SUDD -> UNITY/WARRAP, BAHR EL ARAB RAIN  -  notes ({pd.Timestamp.now():%Y-%m-%d %H:%M})

HOW TO READ THE FIGURES
- Fig 2: if the lake correlation exists only in 'Raw' and vanishes in the anomaly panel, it is shared
  season/trend, not a driver. Peak lag should be consistent with Sudd travel time (weeks-months).
- Fig 3: with ~22 points, use as illustration; check whether 2019/2020 sit on the line.
- Fig 4: does the Bahr el Arab catchment beat the local Warrap rain at 2-8 weeks?
- Fig 5/6: strongest evidence = B/C/D beat A out-of-sample (see 'years_beating_A' in ablation_loyo.csv).

CAVEATS
- Detrending removes the post-2019 lake rise, which may itself be the real signal (2019-20 event).
  Report both the 'partial | day-of-year' and the fully detrended result.
- Days without a MODIS flood detection are filled with 0 (cloud / no pass are not distinguishable).
- Bahr el Arab catchment box is approximate; use a proper basin polygon for the final report.
- Leave-one-year-out is not purged: rows in the first weeks after a held-out year can share information.
- Association only. These tests cannot prove causation; lakes and Sudd flooding share a wet-season forcing.

LITERATURE TO CITE (verify details and page numbers before using)
- Sutcliffe, J.V. & Parks, Y.P. (1999). The Hydrology of the Nile. IAHS Special Publication 5.
  (Sudd water balance, Bahr el Jebel inflow, Bahr el Ghazal basin losses)
- Sutcliffe, J.V. & Parks, Y.P. (1987). Comparative water balances of selected African wetlands.
  Hydrological Sciences Journal 32(2).
- Sutcliffe, J.V. & Petersen, G. (2007). Lake Victoria: derivation of a corrected natural water level
  series. Hydrological Sciences Journal 52(6).
- Mohamed, Y.A., Bastiaanssen, W.G.M., Savenije, H.H.G. (2004). Spatial variability of evaporation and
  moisture storage in the swamps of the upper Nile studied by remote sensing techniques.
  Journal of Hydrology 289.
- 2019-2020 floods: OCHA / ReliefWeb South Sudan flood situation reports (2019-2020), and recent
  papers on the record Lake Victoria level and Sudd expansion (search: 'Lake Victoria 2020 record level
  Sudd flooding', 'Indian Ocean Dipole 2019 East Africa rainfall').
"""
(OUT_DIR / 'notes_and_literature.txt').write_text(notes, encoding='utf-8')

print("\nDONE. Outputs:", OUT_DIR)
for f in sorted(OUT_DIR.rglob('*')):
    if f.is_file():
        print("  -", f.relative_to(OUT_DIR))