"""
02_build_stacks.py
==================
Step 1  Load aligned monthly rasters; compute growing-season (Apr-Oct) mean
        per pixel per year -> ndvi_annual, lst_annual, precip_annual, srad_annual
Step 2  Load HFP annual rasters -> hfp_annual
Step 3  Apply veg_mask  (classes 10/20/30/40 = keep; else NaN)
Step 4  Drop pixels with any NaN in their 10-year series (build valid_mask)
Step 5  Save analysis_stacks.npz
Step 6  Print sanity statistics and memory usage
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(os.environ.get("DATA_ROOT", "."))  # folder holding the raw data (see README)

DIRS = {
    "NDVI":   BASE / "MOD13Q1_MONTHLY_NDVI_aligned",
    "LST":    BASE / "TEMPERATURE_MOD11A2_MONTHLY_aligned",
    "CHIRPS": BASE / "CHIRPS_MONTHLY_PRECIPITATION_aligned",
    "SRAD":   BASE / "ERA5_SSRD_1km_Monthly_solar_rad_aligned",
    "HFP":    BASE / "hfp_2015-2024_aligned",
}

VEG_MASK_PATH = BASE / "veg_mask.tif"
OUT_NPZ       = BASE / "analysis_stacks.npz"

YEARS          = list(range(2015, 2025))   # 10 years
GROWING_MONTHS = list(range(4, 11))        # April – October  (7 months)


# ---------------------------------------------------------------------------
# Filename resolvers
# ---------------------------------------------------------------------------
def _monthly_path(var, year, month):
    m = f"{month:02d}"
    y = str(year)
    names = {
        "NDVI":   f"MOD13Q1_NDVI_{y}_{m}_samarkand_4326.tif",
        "LST":    f"MOD11A2_LST_{y}_{m}_samarkand_4326.tif",
        "CHIRPS": f"CHIRPS_MONTHLY_{y}_{m}_samarkand_4326.tif",
        "SRAD":   f"ERA5_SSRD_1km_{y}_{m}.tif",
    }
    return DIRS[var] / names[var]


def _hfp_path(year):
    return DIRS["HFP"] / f"hfp{year}_samarkand.tif"


# ---------------------------------------------------------------------------
# Helper: read a single raster band as float64
# ---------------------------------------------------------------------------
def _read(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        # nodata=-inf or nodata=NaN both end up here; convert anything non-finite
        # that came from a tagged nodata to NaN
        nd = src.nodata
        if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
            arr[arr == nd] = np.nan
        arr[~np.isfinite(arr)] = np.nan
    return arr


# ---------------------------------------------------------------------------
# Step 1 – Growing-season annual means for monthly variables
# ---------------------------------------------------------------------------
print("=" * 68)
print("STEP 1  Growing-season annual means  (Apr-Oct, months 4-10)")
print("=" * 68)

# Read master grid shape from one reference file
with rasterio.open(_monthly_path("NDVI", 2015, 4)) as ref:
    H, W      = ref.height, ref.width
    TRANSFORM = ref.transform
    CRS_WKT   = ref.crs.to_wkt()

print(f"  Grid : {W} x {H} px  |  years: {YEARS[0]}-{YEARS[-1]}")
print(f"  Growing season: months {GROWING_MONTHS[0]}-{GROWING_MONTHS[-1]}  ({len(GROWING_MONTHS)} months/year)\n")

monthly_vars = ["NDVI", "LST", "CHIRPS", "SRAD"]
stacks = {}   # var -> (10, H, W) float64

for var in monthly_vars:
    arr = np.full((len(YEARS), H, W), np.nan, dtype=np.float64)
    n_missing = 0
    for yi, year in enumerate(YEARS):
        month_slices = []
        for month in GROWING_MONTHS:
            fp = _monthly_path(var, year, month)
            if not fp.exists():
                n_missing += 1
                print(f"  WARNING: missing {fp.name}")
                continue
            month_slices.append(_read(fp))
        if month_slices:
            stack_m = np.stack(month_slices, axis=0)          # (7, H, W)
            arr[yi] = np.nanmean(stack_m, axis=0)              # (H, W)
            # Pixels where ALL 7 months were NaN -> nanmean gives NaN -> correct
            all_nan = np.all(np.isnan(stack_m), axis=0)
            arr[yi, all_nan] = np.nan
    stacks[var] = arr
    valid_count = int(np.sum(~np.isnan(arr)))
    total_count = arr.size
    print(f"  [{var}]  shape={arr.shape}  valid={valid_count:,}/{total_count:,}"
          f"  ({100*valid_count/total_count:.1f}%)  "
          f"missing_files={n_missing}")


# ---------------------------------------------------------------------------
# Step 2 – Load HFP (annual)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 68}")
print("STEP 2  Load HFP annual rasters")
print("=" * 68)

hfp = np.full((len(YEARS), H, W), np.nan, dtype=np.float64)
for yi, year in enumerate(YEARS):
    fp = _hfp_path(year)
    if not fp.exists():
        print(f"  WARNING: missing {fp.name}")
        continue
    hfp[yi] = _read(fp)

valid_count = int(np.sum(~np.isnan(hfp)))
total_count = hfp.size
print(f"  [HFP]  shape={hfp.shape}  valid={valid_count:,}/{total_count:,}"
      f"  ({100*valid_count/total_count:.1f}%)")
stacks["HFP"] = hfp


# ---------------------------------------------------------------------------
# Step 3 – Apply vegetation mask
# ---------------------------------------------------------------------------
print(f"\n{'=' * 68}")
print("STEP 3  Apply vegetation mask  (veg_mask == 1 -> keep)")
print("=" * 68)

with rasterio.open(VEG_MASK_PATH) as src:
    veg_mask_arr = src.read(1)          # uint8: 1=keep, 0=exclude, 255=nodata

keep = veg_mask_arr == 1               # (H, W) bool

for var, arr in stacks.items():
    before = int(np.sum(~np.isnan(arr)))
    arr[:, ~keep] = np.nan             # broadcast: all 10 years at non-veg pixels
    after  = int(np.sum(~np.isnan(arr)))
    print(f"  [{var}]  zeroed {before-after:,} pixel-years outside vegetation")

stacks_labels = ["NDVI", "LST", "CHIRPS", "SRAD", "HFP"]


# ---------------------------------------------------------------------------
# Step 4 – Drop pixels with any NaN across 10-year series in any variable
# ---------------------------------------------------------------------------
print(f"\n{'=' * 68}")
print("STEP 4  Valid-pixel mask  (NaN-free across all variables & all years)")
print("=" * 68)

valid_mask = np.ones((H, W), dtype=bool)
for var in stacks_labels:
    has_nan = np.any(np.isnan(stacks[var]), axis=0)    # (H, W)
    dropped  = int(np.sum(has_nan & valid_mask))
    valid_mask &= ~has_nan
    print(f"  [{var}]  pixels dropped by this variable: {dropped:,}")

n_valid = int(valid_mask.sum())
n_total_px = H * W
print(f"\n  Final valid pixels : {n_valid:,} / {n_total_px:,}  ({100*n_valid/n_total_px:.1f}%)")

# Enforce mask: set invalid pixels to NaN in all arrays
for var in stacks_labels:
    stacks[var][:, ~valid_mask] = np.nan


# ---------------------------------------------------------------------------
# Step 5 – Save analysis_stacks.npz
# ---------------------------------------------------------------------------
print(f"\n{'=' * 68}")
print("STEP 5  Saving analysis_stacks.npz")
print("=" * 68)

# Serialize Affine transform as 6-element array [a, b, c, d, e, f]
tf = TRANSFORM
transform_arr = np.array([tf.a, tf.b, tf.c, tf.d, tf.e, tf.f], dtype=np.float64)

np.savez_compressed(
    OUT_NPZ,
    ndvi_annual   = stacks["NDVI"].astype(np.float32),   # downcast to float32 to halve size
    lst_annual    = stacks["LST"].astype(np.float32),
    precip_annual = stacks["CHIRPS"].astype(np.float32),
    srad_annual   = stacks["SRAD"].astype(np.float32),
    hfp_annual    = stacks["HFP"].astype(np.float32),
    years         = np.array(YEARS, dtype=np.int32),
    valid_mask    = valid_mask,
    transform     = transform_arr,
    crs_wkt       = np.array(CRS_WKT),
)

size_mb = OUT_NPZ.stat().st_size / 1024**2
print(f"  Saved  -> {OUT_NPZ.name}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Step 6 – Sanity statistics
# ---------------------------------------------------------------------------
print(f"\n{'=' * 68}")
print("STEP 6  Sanity statistics")
print("=" * 68)

# Memory (in-memory float64 arrays, pre-downcast)
bytes_in_mem = sum(s.nbytes for s in stacks.values())
print(f"\n  In-memory (float64 before downcast) : {bytes_in_mem/1024**2:.2f} MB")
print(f"  Compressed .npz on disk             : {size_mb:.2f} MB\n")

# --- Per-variable stats over valid pixels only (valid_mask == True) ---
units = {
    "NDVI":   "dimensionless",
    "LST":    "deg C",
    "CHIRPS": "mm/month",
    "SRAD":   "W/m2",
    "HFP":    "index",
}

print(f"  {'Variable':<8}  {'Min':>8}  {'Max':>8}  {'Mean':>8}  {'Std':>8}  "
      f"{'Valid px':>10}  {'% valid':>8}  Units")
print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*12}")

for var in stacks_labels:
    arr  = stacks[var]                 # (10, H, W) float64 with NaN for invalid
    vals = arr[~np.isnan(arr)]         # all valid pixel-years
    n_v  = len(vals)
    n_t  = arr.size
    print(f"  {var:<8}  {vals.min():>8.4f}  {vals.max():>8.4f}  "
          f"{vals.mean():>8.4f}  {vals.std():>8.4f}  "
          f"{n_v:>10,}  {100*n_v/n_t:>7.1f}%  {units[var]}")

# --- Per-year NDVI mean (sanity: should show interannual variability) ---
print("\n  NDVI growing-season mean per year (valid pixels only):")
print(f"  {'Year':>4}  {'Mean NDVI':>10}  {'Std':>8}  {'Valid px':>10}")
print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*10}")
for yi, year in enumerate(YEARS):
    row = stacks["NDVI"][yi]
    v   = row[~np.isnan(row)]
    if len(v):
        print(f"  {year:>4}  {v.mean():>10.4f}  {v.std():>8.4f}  {len(v):>10,}")
    else:
        print(f"  {year:>4}  {'no data':>10}")

print(f"\n  Valid pixel count : {n_valid:,}  ({100*n_valid/n_total_px:.1f}% of {H}x{W} grid)")
print(f"  Array shape       : (10, {H}, {W}) per variable")
print(f"  Growing season    : Apr-Oct  ({len(GROWING_MONTHS)} months averaged per year)")
print("\nDone.")
