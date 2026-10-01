"""
01_align_rasters.py
====================
Step 1  Build WorldCover_2021_Samarkand_1km.tif  (master grid, mode resample)
Step 2  Align NDVI / LST / CHIRPS / SRAD / HFP to master  (bilinear; skip if done)
Step 3  Verify every aligned file: transform / size / CRS  -> PASS / FAIL
Step 4  Build veg_mask.tif  (ESA classes 10,20,30,40 -> 1; else -> 0)
Step 5  Print markdown summary table
"""

import os
import sys
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(os.environ.get("DATA_ROOT", "."))  # folder holding the raw data (see README)

ESA_SRC  = BASE / "ESA_world_cover" / "ESA_WorldCover_2021_Clipped.tif"
NDVI_REF = BASE / "MOD13Q1_MONTHLY_NDVI" / "MOD13Q1_NDVI_2015_04_samarkand_4326.tif"
ESA_1KM  = BASE / "ESA_world_cover" / "WorldCover_2021_Samarkand_1km.tif"
VEG_MASK = BASE / "veg_mask.tif"

# (label, source_folder, aligned_folder, resampling)
DATASETS = [
    ("NDVI",   "MOD13Q1_MONTHLY_NDVI",            "MOD13Q1_MONTHLY_NDVI_aligned",            Resampling.bilinear),
    ("LST",    "TEMPERATURE_MOD11A2_MONTHLY",      "TEMPERATURE_MOD11A2_MONTHLY_aligned",      Resampling.bilinear),
    ("CHIRPS", "CHIRPS_MONTHLY_PRECIPITATION",     "CHIRPS_MONTHLY_PRECIPITATION_aligned",     Resampling.bilinear),
    ("SRAD",   "ERA5_SSRD_1km_Monthly_solar_rad",  "ERA5_SSRD_1km_Monthly_solar_rad_aligned",  Resampling.bilinear),
    ("HFP",    "hfp_2015-2024",                    "hfp_2015-2024_aligned",                    Resampling.bilinear),
]

VEG_CLASSES = {10, 20, 30, 40}   # Tree, Shrub, Grassland, Cropland

LZW_OPTS = dict(compress="lzw", tiled=True, blockxsize=256, blockysize=256)

WC_LABELS = {
    0:   "No data / outside AOI",
    10:  "Tree cover",
    20:  "Shrubland",
    30:  "Grassland",
    40:  "Cropland",
    50:  "Built-up",
    60:  "Bare / sparse vegetation",
    70:  "Snow and ice",
    80:  "Permanent water",
    90:  "Herbaceous wetland",
    95:  "Mangroves",
    100: "Moss and lichen",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transforms_match(t1, t2, tol=1e-7):
    """Compare two Affine transforms coefficient by coefficient."""
    return all(abs(a - b) < tol for a, b in zip(t1, t2))


def _crs_match(c1, c2):
    """Compare CRS via WKT to avoid EPSG database lookups."""
    return c1.to_wkt() == c2.to_wkt()


def _is_aligned(path, ref_tf, ref_w, ref_h, ref_crs):
    try:
        with rasterio.open(path) as s:
            return (
                s.width  == ref_w
                and s.height == ref_h
                and _transforms_match(s.transform, ref_tf)
                and _crs_match(s.crs, ref_crs)
            )
    except Exception:
        return False


def _src_nodata(src):
    """Nodata to mask source pixels; NaN for NaN-encoded float files."""
    if src.nodata is not None:
        return src.nodata          # -inf for HFP, None for others
    if src.dtypes[0] in ("float32", "float64"):
        return float("nan")        # NDVI / LST / CHIRPS / SRAD
    return None


def _reproject_file(src_path, dst_path, ref_tf, ref_w, ref_h, ref_crs, resampling):
    with rasterio.open(src_path) as src:
        src_nd = _src_nodata(src)

        # Normalize output nodata: -inf -> NaN for floats (GDAL handles NaN cleanly)
        if src.dtypes[0] in ("float32", "float64"):
            dst_nd = float("nan")
        else:
            dst_nd = src_nd

        profile = src.profile.copy()
        profile.update(
            crs=ref_crs,
            transform=ref_tf,
            width=ref_w,
            height=ref_h,
            nodata=dst_nd,
            **LZW_OPTS,
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for bi in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, bi),
                    destination=rasterio.band(dst, bi),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_tf,
                    dst_crs=ref_crs,
                    resampling=resampling,
                    src_nodata=src_nd,
                    dst_nodata=dst_nd,
                )


# ---------------------------------------------------------------------------
# Step 0 — Master grid (from NDVI reference)
# ---------------------------------------------------------------------------
print("=" * 68)
print("STEP 0  Master grid from NDVI reference")
print("=" * 68)
with rasterio.open(NDVI_REF) as ref:
    MASTER_TF  = ref.transform
    MASTER_W   = ref.width
    MASTER_H   = ref.height
    MASTER_CRS = ref.crs

print(f"  Grid     : {MASTER_W} x {MASTER_H} px")
print(f"  Pixel    : {MASTER_TF.a:.10f} deg  (~{MASTER_TF.a * 111320:.0f} m at equator)")
print(f"  Origin   : ({MASTER_TF.c:.8f}, {MASTER_TF.f:.8f})  [left, top]")
print(f"  CRS      : {MASTER_CRS.to_wkt().split(chr(34))[1]}")


# ---------------------------------------------------------------------------
# Step 1 — WorldCover 10 m → 1 km  (mode / majority resampling)
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("STEP 1  WorldCover 10 m -> 1 km  (mode resampling -> master grid)")
print("=" * 68)

if ESA_1KM.exists() and _is_aligned(ESA_1KM, MASTER_TF, MASTER_W, MASTER_H, MASTER_CRS):
    print("  Already exists and aligned — skipped.")
else:
    print(f"  Resampling {ESA_SRC.name}  ...  ", end="", flush=True)
    with rasterio.open(ESA_SRC) as src:
        profile = src.profile.copy()
        profile.update(
            crs=MASTER_CRS,
            transform=MASTER_TF,
            width=MASTER_W,
            height=MASTER_H,
            dtype="uint8",
            nodata=0,
            count=1,
            **LZW_OPTS,
        )
        with rasterio.open(ESA_1KM, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=MASTER_TF,
                dst_crs=MASTER_CRS,
                resampling=Resampling.mode,
                src_nodata=0,
                dst_nodata=0,
            )
    print("done.")
    print(f"  Saved  → {ESA_1KM.relative_to(BASE)}")


# ---------------------------------------------------------------------------
# Step 2 — Align all datasets
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("STEP 2  Aligning datasets to master grid")
print("=" * 68)

align_log = []   # (label, n_total, n_reprojected, n_skipped, dst_folder)

for label, src_folder, dst_folder, method in DATASETS:
    src_dir = BASE / src_folder
    dst_dir = BASE / dst_folder
    dst_dir.mkdir(exist_ok=True)

    tif_files = sorted(src_dir.glob("*.tif"))
    n_total = len(tif_files)
    n_repr = n_skip = 0

    print(f"\n  [{label}]  {n_total} files  |  {dst_folder}")

    for i, src_path in enumerate(tif_files, 1):
        dst_path = dst_dir / src_path.name
        if dst_path.exists() and _is_aligned(dst_path, MASTER_TF, MASTER_W, MASTER_H, MASTER_CRS):
            n_skip += 1
            print(f"\r    {i}/{n_total}  skip: {n_skip}  reproject: {n_repr}", end="", flush=True)
        else:
            _reproject_file(src_path, dst_path, MASTER_TF, MASTER_W, MASTER_H, MASTER_CRS, method)
            n_repr += 1
            print(f"\r    {i}/{n_total}  skip: {n_skip}  reproject: {n_repr}", end="", flush=True)

    print()  # newline after progress
    align_log.append((label, n_total, n_repr, n_skip, dst_folder))


# ---------------------------------------------------------------------------
# Step 3 — Verify all aligned outputs
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("STEP 3  Verification — transform / size / CRS")
print("=" * 68)

verify_log = []   # (label, n_files, n_pass, n_fail, fail_names)

# Also verify WorldCover 1km
wc_ok = _is_aligned(ESA_1KM, MASTER_TF, MASTER_W, MASTER_H, MASTER_CRS)
verify_log.append(("WorldCover 1km", 1, int(wc_ok), int(not wc_ok), [] if wc_ok else [ESA_1KM.name]))
print(f"  [WorldCover 1km]   {'PASS' if wc_ok else 'FAIL'}")

for label, src_folder, dst_folder, _ in DATASETS:
    dst_dir = BASE / dst_folder
    files   = sorted(dst_dir.glob("*.tif"))
    n_pass  = n_fail = 0
    fails   = []
    for f in files:
        if _is_aligned(f, MASTER_TF, MASTER_W, MASTER_H, MASTER_CRS):
            n_pass += 1
        else:
            n_fail += 1
            fails.append(f.name)
    status = "PASS" if n_fail == 0 else "FAIL"
    print(f"  [{label:<7}]  {len(files):3d} files  →  {status}  (pass={n_pass}, fail={n_fail})")
    for fn in fails[:3]:
        print(f"    ✗ {fn}")
    verify_log.append((label, len(files), n_pass, n_fail, fails))


# ---------------------------------------------------------------------------
# Step 4 — Vegetation mask
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("STEP 4  Building vegetation mask from WorldCover 1 km")
print("=" * 68)

with rasterio.open(ESA_1KM) as src:
    wc = src.read(1)
    veg_mask = np.isin(wc, sorted(VEG_CLASSES)).astype(np.uint8)
    profile = src.profile.copy()
    profile.update(dtype="uint8", nodata=255, count=1, **LZW_OPTS)
    with rasterio.open(VEG_MASK, "w", **profile) as dst:
        dst.write(veg_mask, 1)

n_veg    = int(veg_mask.sum())
n_px_tot = int(wc.size)
pct_veg  = 100 * n_veg / n_px_tot

print(f"  Vegetation pixels : {n_veg:,} / {n_px_tot:,}  ({pct_veg:.1f}%)")
print(f"  Saved  → {VEG_MASK.relative_to(BASE)}")

print("\n  WorldCover class distribution (1 km grid):")
print(f"  {'Class':>5}  {'Label':<28}  {'Pixels':>8}  {'%':>6}  {'In mask':>8}")
print(f"  {'-'*5}  {'-'*28}  {'-'*8}  {'-'*6}  {'-'*8}")
for val, cnt in zip(*np.unique(wc, return_counts=True)):
    lbl  = WC_LABELS.get(int(val), f"class {val}")
    keep = "YES" if int(val) in VEG_CLASSES else ""
    print(f"  {val:>5}  {lbl:<28}  {cnt:>8,}  {100*cnt/n_px_tot:>5.1f}%  {keep:>8}")


# ---------------------------------------------------------------------------
# Step 5 — Summary tables (markdown)
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("SUMMARY REPORT (markdown)")
print("=" * 68)

print(f"""
### Master Grid

| Property | Value |
|----------|-------|
| CRS | EPSG:4326 (WGS 84) |
| Pixel size | {MASTER_TF.a:.10f} deg (~{MASTER_TF.a*111320:.0f} m) |
| Grid size | {MASTER_W} × {MASTER_H} pixels |
| Origin (left, top) | ({MASTER_TF.c:.8f}, {MASTER_TF.f:.8f}) |
| Extent W | {MASTER_TF.c:.4f} – {MASTER_TF.c + MASTER_W*MASTER_TF.a:.4f} |
| Extent N–S | {MASTER_TF.f + MASTER_H*MASTER_TF.e:.4f} – {MASTER_TF.f:.4f} |
| Reference file | {NDVI_REF.name} |
""")

print("### Alignment Results\n")
print(f"| {'Dataset':<14} | {'Source files':>12} | {'Reprojected':>11} | {'Skipped':>7} | Output folder |")
print(f"|{'-'*16}|{'-'*14}|{'-'*13}|{'-'*9}|{'-'*40}|")
for lbl, n_tot, n_repr, n_skip, dst_fld in align_log:
    print(f"| {lbl:<14} | {n_tot:>12} | {n_repr:>11} | {n_skip:>7} | `{dst_fld}` |")

print("\n### Verification Results\n")
print(f"| {'Dataset':<14} | {'Files':>5} | {'PASS':>4} | {'FAIL':>4} | Status |")
print(f"|{'-'*16}|{'-'*7}|{'-'*6}|{'-'*6}|{'-'*8}|")
for lbl, n_files, n_pass, n_fail, _ in verify_log:
    status = "✓ PASS" if n_fail == 0 else "✗ FAIL"
    print(f"| {lbl:<14} | {n_files:>5} | {n_pass:>4} | {n_fail:>4} | {status} |")

print(f"""
### Vegetation Mask

| Property | Value |
|----------|-------|
| Output file | `veg_mask.tif` |
| Keep classes | 10 Tree, 20 Shrub, 30 Grassland, 40 Cropland |
| Vegetation pixels | {n_veg:,} / {n_px_tot:,} ({pct_veg:.1f}%) |
| NoData value | 255 |
| Dtype | uint8 |
| Compression | LZW (tiled 256×256) |
""")

print("Done.")
