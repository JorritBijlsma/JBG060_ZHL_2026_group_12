"""Diagnostic: Check ERA5 units and values"""
import sys
import numpy as np
import pandas as pd

sys.path.append(r'C:\Users\20241060\OneDrive - TU Eindhoven\JBG060\JBG060_ZHL_2026_group_12\processing_data')
from loading import load_rainfall_runoff

print("Loading ERA5 data...")
years = np.arange(2000, 2025)
era5 = load_rainfall_runoff(years)

# Filter to South Sudan
bbox_ssd = {'lat_min': 3.0, 'lat_max': 13.0, 'lon_min': 24.0, 'lon_max': 36.0}
lat_idx = (era5.latitude >= bbox_ssd['lat_min']) & (era5.latitude <= bbox_ssd['lat_max'])
lon_idx = (era5.longitude >= bbox_ssd['lon_min']) & (era5.longitude <= bbox_ssd['lon_max'])
era5_ssd = era5.isel(latitude=lat_idx, longitude=lon_idx)

print(f"\n=== ERA5 Data Diagnostics ===")
print(f"Shape: {era5_ssd['tp'].shape}")
print(f"Lat range: {float(era5_ssd.latitude.min()):.2f} to {float(era5_ssd.latitude.max()):.2f}")
print(f"Lon range: {float(era5_ssd.longitude.min()):.2f} to {float(era5_ssd.longitude.max()):.2f}")

print(f"\n--- tp (total precipitation) raw stats ---")
tp_vals = era5_ssd['tp'].values.flatten()
tp_vals = tp_vals[~np.isnan(tp_vals)]
print(f"  Min: {tp_vals.min():.8f}")
print(f"  Max: {tp_vals.max():.8f}")
print(f"  Mean: {tp_vals.mean():.8f}")
print(f"  Median: {np.median(tp_vals):.8f}")
print(f"  Percentiles (50, 90, 99, 99.9): {np.percentile(tp_vals, [50, 90, 99, 99.9])}")
print(f"  % nonzero: {(tp_vals > 0).mean()*100:.2f}%")

print(f"\n--- ro (runoff) raw stats ---")
ro_vals = era5_ssd['ro'].values.flatten()
ro_vals = ro_vals[~np.isnan(ro_vals)]
print(f"  Min: {ro_vals.min():.8f}")
print(f"  Max: {ro_vals.max():.8f}")
print(f"  Mean: {ro_vals.mean():.8f}")
print(f"  % nonzero: {(ro_vals > 0).mean()*100:.2f}%")

# Check units from attributes
print(f"\n=== Units ===")
print(f"tp units: {era5_ssd['tp'].attrs.get('units', 'not specified')}")
print(f"ro units: {era5_ssd['ro'].attrs.get('units', 'not specified')}")

# Spatial average
tp_spatial = era5_ssd['tp'].mean(dim=['latitude', 'longitude']).to_series()
print(f"\n--- Spatial mean (as stored) ---")
print(f"  Daily mean: {tp_spatial.mean():.8f}")
print(f"  Daily max: {tp_spatial.max():.8f}")

# Monthly analysis
tp_monthly = tp_spatial.groupby(tp_spatial.index.month).mean()
print(f"\n--- Monthly mean (as stored) ---")
for m, v in tp_monthly.items():
    print(f"  Month {m}: {v:.8f}")

# If units are meters, convert
print(f"\n=== If units are meters (× 1000 to get mm) ===")
print(f"  Daily mean rainfall: {tp_spatial.mean()*1000:.4f} mm/day")
print(f"  Daily max rainfall: {tp_spatial.max()*1000:.4f} mm/day")
print(f"  Annual total (avg year): {tp_spatial.mean()*365*1000:.1f} mm/year")