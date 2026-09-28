"""Two slide-ready figures on flooding and cropland.

    python processing_data/crop_flood_visuals.py

Outputs into ``EDA/crop_impact/``:

    flood_cropland_map_<year>.png      flood x cropland, clipped to South Sudan
    flooded_cropland_by_payam.csv      flooded cropland in hectares, per payam
    stage_damage_rice.png              yield loss by growth stage
    flooded_area_by_year.csv           the scan used to choose the year

HOW THE TWO GRIDS ARE COMBINED
------------------------------
The two inputs do not share a grid, so one has to be resampled onto the other:

    ASAP crop mask   ~0.00446 deg (~500 m), values 0-100 = *percent* of the
                     cell under crops.  Not a binary mask.
    Flood masks      ~0.00208 deg (~232 m), one row per flooded pixel-day.

Everything is accumulated onto the ASAP grid, which is the coarser of the two,
because upsampling the crop mask would invent detail it does not have.  Each
ASAP cell gets a flooded *fraction* - the area of distinct flood pixels that
fell inside it, over the cell's own area, capped at 1 - rather than a yes/no
flag, so a cell clipped by the edge of a flood is not counted as fully
inundated.

Flooded cropland is then, per cell:

    crop_fraction  x  flooded_fraction  x  cell_area_ha

**This assumes crops are spread evenly within the cell.** Inside a 500 m cell
that is half cropland and half flooded, the estimator says a quarter of it is
flooded cropland; in reality crops sit on the drier ground and the true figure
is lower.  The hectare numbers are therefore an upper bound, and are reported
as such.  Cell area is computed per row of latitude rather than assumed
constant, since a degree of longitude shortens by about 1.5% between 3 N and
12 N.

WHICH FLOODS COUNT
------------------
``load_flood_masks`` returns recurring floods (flood_type 0) and unusual ones
(flood_type 1).  Recurring flooding is the normal seasonal inundation of the
Sudd and its margins - farmers plant around it, and counting it as crop damage
would overstate the problem enormously.  The default here is therefore
``unusual`` only.  ``--flood-type both`` is available and the year-selection
table reports both, so the choice is visible rather than buried.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import geometry_mask, rasterize

# loading.py builds its paths as './raw_data/...', so it only works with the
# repository root as the working directory.  Putting the root on sys.path as
# well lets this run either as a script or as `python -m processing_data.…`.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from processing_data.loading import flood_mask_bbox, load_flood_masks  # noqa: E402

RAW = REPO_ROOT / "raw_data"
OUT = REPO_ROOT / "EDA" / "crop_impact"
BOUNDARIES = RAW / "Administrative boundaries"
CROP_MASK = RAW / "farmland" / "asap_mask_crop_v04.tif"
RANGELAND_MASK = RAW / "farmland" / "asap_mask_rangeland_v04.tif"

# The two MODIS/VIIRS tiles together span South Sudan; this box trims them to
# the country's own extent before anything expensive happens.
SSD_BBOX = {"lat_min": 3.4, "lat_max": 12.3, "lon_min": 24.0, "lon_max": 36.0}

FLOOD_PIXEL_DEG = 0.0020833          # ~232 m at the equator, per the user guide
FLOOD_YEARS = range(2000, 2026)      # what raw_data/flood_masks actually holds

# Display threshold for calling a cell "cropland" on the map.  Hectare figures
# never use it - they weight by the actual percentage - but a map needs a cut.
CROP_DISPLAY_PCT = 5.0

# Share of crop-flagged cells nationally where ASAP puts more rangeland cover
# than crop cover.  Used as the reference point for the per-payam column, so a
# payam is only called out when it is well above what is normal everywhere.
RANGELAND_DOMINANT_BASELINE = 0.73
RANGELAND_DOMINANT_FLAG = 0.90

DEG_KM = 111.32                      # length of a degree of latitude, km

# Shrestha, B.B., Sawano, H., Ohara, M. & Mishra, B.K. (2021).  Flood damage
# assessment in the Pampanga river basin of the Philippines.  *Journal of
# Hydrology: Regional Studies* 34, 100810.  Paddy-rice yield loss at 1 m
# inundation depth sustained for 8 days, by growth stage at the time of the
# flood.
RICE_STAGE_DAMAGE = {
    "Vegetative": 50.0,
    "Reproductive": 40.0,
    "Maturity": 36.0,
}
RICE_SOURCE = ("Shrestha et al. (2021), J. Hydrology: Regional Studies 34, 100810 - "
               "paddy rice, 1 m inundation depth sustained for 8 days")


# --------------------------------------------------------------- geometry

def cell_area_ha(lats: np.ndarray, res_deg: float) -> np.ndarray:
    """Area of one grid cell, in hectares, for each row of latitude."""
    height_km = res_deg * DEG_KM
    width_km = res_deg * DEG_KM * np.cos(np.radians(lats))
    return height_km * width_km * 100.0            # km^2 -> ha


def load_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    country = gpd.read_file(BOUNDARIES / "ssd_admin0.geojson")
    payams = gpd.read_file(BOUNDARIES / "ssd_admin3.geojson")
    return country, payams


def read_asap_window(path: Path) -> tuple[np.ndarray, rasterio.Affine, dict]:
    """One ASAP mask, cut to the South Sudan box.

    The full raster is 29,346 x 80,640 cells of global coverage; reading a
    window keeps this to a few tens of megabytes.
    """
    with rasterio.open(path) as src:
        window = src.window(SSD_BBOX["lon_min"], SSD_BBOX["lat_min"],
                            SSD_BBOX["lon_max"], SSD_BBOX["lat_max"])
        pct = src.read(1, window=window).astype("float32")
        transform = src.window_transform(window)
    meta = {"res": abs(transform.a), "height": pct.shape[0], "width": pct.shape[1]}
    return pct, transform, meta


def read_crop_window() -> tuple[np.ndarray, rasterio.Affine, dict]:
    return read_asap_window(CROP_MASK)


# ----------------------------------------------------------- flood inputs

def flooded_extent_by_year(flood_type: str, cache: Path) -> pd.DataFrame:
    """Distinct flooded pixels per year, inside the South Sudan box.

    Extent, not pixel-days: a pixel flooded for three months counts once.  That
    is the right basis for "which year flooded the largest area", which is the
    question being asked of it.
    """
    if cache.exists():
        return pd.read_csv(cache)

    pixel_km2 = (FLOOD_PIXEL_DEG * DEG_KM) ** 2
    rows = []
    for year in FLOOD_YEARS:
        counts = {}
        for kind, folder in (("recurring", "compact_recurring"),
                             ("unusual", "compact_unusual")):
            parts = []
            for tile in ("h20v08", "h21v08"):
                path = RAW / "flood_masks" / folder / f"flood_events_{tile}_{year}.parquet"
                if path.exists():
                    parts.append(pd.read_parquet(path, columns=["lat", "lon"]))
            if not parts:
                counts[kind] = np.nan
                continue
            df = flood_mask_bbox(pd.concat(parts, ignore_index=True), SSD_BBOX)
            counts[kind] = len(df.drop_duplicates(subset=["lat", "lon"]))
        rows.append({
            "year": year,
            "unusual_km2": counts["unusual"] * pixel_km2,
            "recurring_km2": counts["recurring"] * pixel_km2,
        })
        print(f"  {year}: unusual {rows[-1]['unusual_km2']:>9,.0f} km2   "
              f"recurring {rows[-1]['recurring_km2']:>9,.0f} km2")

    out = pd.DataFrame(rows)
    out["both_km2"] = out.unusual_km2 + out.recurring_km2
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def flooded_fraction(year: int, flood_type: str, transform, meta) -> np.ndarray:
    """Fraction of each ASAP cell covered by distinct flood pixels.

    Distinct pixels, again: the same pixel flooding on forty separate dates is
    one flooded pixel, not forty.
    """
    floods = load_flood_masks(np.array([year]), bbox=SSD_BBOX)
    if flood_type != "both":
        keep = 1 if flood_type == "unusual" else 0
        floods = floods[floods.flood_type == keep]
    floods = floods.drop_duplicates(subset=["lat", "lon"])

    res = meta["res"]
    col = np.floor((floods.lon.to_numpy() - transform.c) / res).astype(int)
    row = np.floor((transform.f - floods.lat.to_numpy()) / res).astype(int)
    inside = ((row >= 0) & (row < meta["height"]) & (col >= 0) & (col < meta["width"]))

    counts = np.zeros((meta["height"], meta["width"]), dtype="float32")
    np.add.at(counts, (row[inside], col[inside]), 1.0)

    # Both grids shrink with latitude by the same cosine factor, so the ratio
    # of their areas is constant and the cell-area term cancels.
    pixels_per_cell = (res / FLOOD_PIXEL_DEG) ** 2
    return np.clip(counts / pixels_per_cell, 0.0, 1.0)


# ------------------------------------------------------------ aggregation

def payam_totals(crop_frac, flood_frac, inside, payams, transform, meta,
                 range_pct=None) -> pd.DataFrame:
    """Flooded cropland in hectares, per payam.

    ``rangeland_dominant`` is a data-quality column, not a result.  ASAP's crop
    and rangeland masks are both percent-cover layers and are *not* exclusive:
    a cell can be 20% crop and 65% rangeland, and nationally 89% of
    crop-flagged cells carry some rangeland, so "has any rangeland" flags
    almost everything and says nothing.  What discriminates is which land use
    dominates the cell.  This column is the share of the payam's crop-flagged
    cells where rangeland cover exceeds crop cover; the national figure is
    ``RANGELAND_DOMINANT_BASELINE``.  Well above that, the payam's "cropland"
    is mostly grazing land with some cultivation in it, and its hectare figure
    should not be quoted without saying so.
    """
    shapes = ((geom, idx) for idx, geom in enumerate(payams.geometry, start=1))
    labels = rasterize(shapes, out_shape=(meta["height"], meta["width"]),
                       transform=transform, fill=0, dtype="int32")

    lats = transform.f - (np.arange(meta["height"]) + 0.5) * meta["res"]
    area = cell_area_ha(lats, meta["res"])[:, None]

    flooded_ha = crop_frac * flood_frac * area * inside
    cropland_ha = crop_frac * area * inside

    flat = labels.ravel()
    n = len(payams) + 1
    totals = np.bincount(flat, weights=flooded_ha.ravel(), minlength=n)
    crop_tot = np.bincount(flat, weights=cropland_ha.ravel(), minlength=n)

    out = payams[["adm3_name", "adm2_name", "adm1_name", "adm3_pcode"]].copy()
    out.columns = ["payam", "county", "state", "pcode"]
    out["cropland_ha"] = crop_tot[1:]
    out["flooded_cropland_ha"] = totals[1:]
    out["flooded_share_of_cropland"] = np.where(
        out.cropland_ha > 0, out.flooded_cropland_ha / out.cropland_ha, np.nan)

    if range_pct is not None:
        is_crop = (crop_frac * 100.0 >= CROP_DISPLAY_PCT) & inside
        dominant = is_crop & (range_pct > crop_frac * 100.0)
        n_crop = np.bincount(flat, weights=is_crop.ravel().astype(float), minlength=n)
        n_dom = np.bincount(flat, weights=dominant.ravel().astype(float), minlength=n)
        out["rangeland_dominant"] = np.where(n_crop[1:] > 0, n_dom[1:] / n_crop[1:], np.nan)

    return out.sort_values("flooded_cropland_ha", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- figures

def _separate(values: np.ndarray, min_gap: float) -> np.ndarray:
    """Nudge values apart until no two are closer than ``min_gap``.

    One upward pass in sorted order, which is enough for five labels and keeps
    each one near where it started.
    """
    out = values.astype(float).copy()
    order = np.argsort(out)
    for prev, cur in zip(order, order[1:]):
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out


def plot_map(crop_frac, flood_frac, inside, payams, transform, meta,
             totals, year, flood_type, path) -> None:
    """Categorical map: cropland, flooded cropland, other flooded area."""
    is_crop = (crop_frac * 100.0 >= CROP_DISPLAY_PCT) & inside
    is_flood = (flood_frac > 0) & inside

    # 0 background, 1 cropland, 2 other flooded, 3 flooded cropland.  Flooded
    # cropland is painted last so it is never hidden by the water layer.
    cat = np.zeros(crop_frac.shape, dtype="uint8")
    cat[is_crop] = 1
    cat[is_flood & ~is_crop] = 2
    cat[is_flood & is_crop] = 3

    cmap = ListedColormap([(1, 1, 1, 0), "#B7DFA8", "#AED6F1", "#CC2B2B"])
    west, east = SSD_BBOX["lon_min"], SSD_BBOX["lon_max"]
    south, north = SSD_BBOX["lat_min"], SSD_BBOX["lat_max"]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(cat, cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
              extent=[west, east, south, north], zorder=2)
    payams.boundary.plot(ax=ax, color="#8C979F", linewidth=0.22, zorder=3)

    top = totals.head(5)
    centres = payams.set_index("adm3_pcode").loc[top.pcode]
    anchors = list(zip(centres.center_lon.to_numpy(), centres.center_lat.to_numpy()))
    # Several of the top payams are neighbours (Kuac North and Kuac South share
    # a border), so labels placed at a fixed offset overlap and hide each
    # other's figures.  Push them apart vertically before drawing.
    label_y = _separate(np.array([a[1] for a in anchors]), min_gap=0.62)
    anchor_x = np.array([a[0] for a in anchors])
    # Labels sit to the right of their payam, except near the eastern edge
    # where the box would leave the map; Narus is the case that needs it.
    to_left = anchor_x > east - 2.2
    label_x = np.where(to_left, anchor_x - 0.55, anchor_x + 0.55)

    for (_, row), (ax_lon, ax_lat), lx, ly, flip in zip(
            top.iterrows(), anchors, label_x, label_y, to_left):
        flag = "*" if row.get("rangeland_dominant", 0) >= RANGELAND_DOMINANT_FLAG else ""
        ax.annotate(
            f"{row.payam}{flag}\n{row.flooded_cropland_ha:,.0f} ha",
            xy=(ax_lon, ax_lat), xytext=(lx, ly), textcoords="data",
            fontsize=8.5, weight="bold", color="#1A1A1A", zorder=6,
            ha="right" if flip else "left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CC2B2B", lw=0.9,
                      alpha=0.94),
            arrowprops=dict(arrowstyle="-", color="#CC2B2B", lw=0.9,
                            shrinkA=0, shrinkB=2),
        )
    ax.scatter(centres.center_lon, centres.center_lat, s=14, color="#CC2B2B",
               edgecolor="white", linewidth=0.6, zorder=5)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1.0 / np.cos(np.radians(8.0)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    label = "unusual" if flood_type == "unusual" else flood_type
    total_ha = totals.flooded_cropland_ha.sum()
    share = total_ha / totals.cropland_ha.sum() if totals.cropland_ha.sum() else np.nan
    ax.set_title(
        f"Flooding on cropland, South Sudan, {year}\n"
        f"{total_ha:,.0f} ha of cropland flooded - {share:.1%} of the national total",
        fontsize=13, loc="left", weight="bold")

    ax.legend(handles=[
        Patch(facecolor="#B7DFA8", label=f"cropland (>= {CROP_DISPLAY_PCT:.0f}% cover, not flooded)"),
        Patch(facecolor="#CC2B2B", label="flooded cropland"),
        Patch(facecolor="#AED6F1", label="other flooded area"),
        Line2D([], [], color="#8C979F", lw=0.6, label="payam boundaries"),
    ], loc="lower left", fontsize=8.5, framealpha=0.94)

    lines = [
        f"Cropland: ASAP crop mask v04 (percent cover, ~500 m).  Flooding: MODIS/VIIRS "
        f"MCDWD_L3_NRT 3-day composite, {label} floods only, distinct pixels flooded at "
        f"any point in {year}.",
        "Hectares weight crop cover by the flooded fraction of each cell and assume crops are "
        "spread evenly within it, so they are an upper bound.",
    ]
    if "rangeland_dominant" in totals and (top.rangeland_dominant >= RANGELAND_DOMINANT_FLAG).any():
        lines.append(
            f"* ASAP puts more rangeland cover than crop cover on over "
            f"{RANGELAND_DOMINANT_FLAG:.0%} of this payam's crop-flagged land "
            f"(national figure {RANGELAND_DOMINANT_BASELINE:.0%}): its land is mostly grazed, "
            f"so the hectare figure is uncertain.")
    fig.text(0.01, 0.012, "\n".join(lines), fontsize=7.2, color="#5A6570", va="bottom")

    fig.tight_layout(rect=(0, 0.02 + 0.018 * len(lines), 1, 0.97))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_stage_damage(path) -> None:
    """Rice yield loss by growth stage, at 1 m depth for 8 days."""
    stages = list(RICE_STAGE_DAMAGE)
    losses = [RICE_STAGE_DAMAGE[s] for s in stages]

    fig, ax = plt.subplots(figsize=(8, 5.6))
    bars = ax.bar(stages, losses, width=0.58,
                  color=["#8C3B3B", "#B5603F", "#D89A57"], edgecolor="white")
    ax.bar_label(bars, fmt="%.0f%%", padding=5, fontsize=13, weight="bold")

    ax.set_ylim(0, 60)
    ax.set_ylabel("yield loss (%)", fontsize=11)
    ax.set_title("Rice yield loss by growth stage when flooded\n"
                 "1 m inundation depth, sustained for 8 days",
                 fontsize=13, loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDE2E6", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)

    fig.text(0.01, 0.015, f"Source: {RICE_SOURCE}.", fontsize=8, color="#5A6570")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flood-type", default="unusual",
                    choices=["unusual", "recurring", "both"],
                    help="which flood class to map (default: unusual)")
    ap.add_argument("--year", type=int, default=None,
                    help="override the automatically chosen year")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.size": 10, "savefig.facecolor": "white",
                         "figure.facecolor": "white"})

    print("Scanning flood masks for the largest flooded area ...")
    by_year = flooded_extent_by_year(args.flood_type, OUT / "flooded_area_by_year.csv")
    column = {"unusual": "unusual_km2", "recurring": "recurring_km2",
              "both": "both_km2"}[args.flood_type]
    year = args.year or int(by_year.loc[by_year[column].idxmax(), "year"])
    print(f"\nLargest {args.flood_type} flood extent: {year} "
          f"({by_year.loc[by_year.year == year, column].iloc[0]:,.0f} km2)")
    print(by_year.sort_values(column, ascending=False).head(5).round(0).to_string(index=False))

    print("\nReading the ASAP crop mask ...")
    crop_pct, transform, meta = read_crop_window()
    crop_frac = crop_pct / 100.0
    range_pct, _, _ = read_asap_window(RANGELAND_MASK)

    country, payams = load_boundaries()
    inside = ~geometry_mask(country.geometry, out_shape=(meta["height"], meta["width"]),
                            transform=transform, invert=False)

    print(f"Rasterising {year} flood pixels onto the crop grid ...")
    flood_frac = flooded_fraction(year, args.flood_type, transform, meta)

    totals = payam_totals(crop_frac, flood_frac, inside, payams, transform, meta,
                          range_pct=range_pct)
    csv_path = OUT / "flooded_cropland_by_payam.csv"
    totals.to_csv(csv_path, index=False)

    print(f"\nNational cropland          : {totals.cropland_ha.sum():>12,.0f} ha")
    print(f"Flooded cropland in {year}   : {totals.flooded_cropland_ha.sum():>12,.0f} ha "
          f"({totals.flooded_cropland_ha.sum() / totals.cropland_ha.sum():.1%})")
    print(f"Payams with any flooded crop: {int((totals.flooded_cropland_ha > 0).sum())} of {len(totals)}")
    print("\nTop 10 payams by flooded cropland:")
    print(totals.head(10)[["payam", "county", "state", "cropland_ha",
                           "flooded_cropland_ha", "flooded_share_of_cropland",
                           "rangeland_dominant"]]
          .round(3).to_string(index=False))

    suspect = totals.head(10)[totals.head(10).rangeland_dominant >= RANGELAND_DOMINANT_FLAG]
    if len(suspect):
        print(f"\n  CAVEAT  On most of the crop-flagged land in these payams ASAP puts more")
        print(f"          rangeland cover than crop cover (national figure "
              f"{RANGELAND_DOMINANT_BASELINE:.0%}), so their 'cropland'")
        print(f"          is mostly grazing land with cultivation in it:")
        for _, r in suspect.iterrows():
            print(f"            {r.payam} ({r.county}): {r.rangeland_dominant:.0%} rangeland-dominant, "
                  f"{r.flooded_cropland_ha:,.0f} ha")

    map_path = OUT / f"flood_cropland_map_{year}.png"
    plot_map(crop_frac, flood_frac, inside, payams, transform, meta,
             totals, year, args.flood_type, map_path)

    stage_path = OUT / "stage_damage_rice.png"
    plot_stage_damage(stage_path)

    print(f"\nWritten:\n  {map_path}\n  {stage_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
