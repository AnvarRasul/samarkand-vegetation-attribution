"""
Per-pixel Mann-Kendall trend analysis + Sen's slope on annual stacks.
Outputs 10 GeoTIFFs (Trends/{var}_sen_slope.tif, Trends/{var}_mk_pvalue.tif).
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
import pymannkendall as mk
from joblib import Parallel, delayed
from tqdm import tqdm

# Variable name → key inside analysis_stacks.npz
VARIABLES = [
    ("ndvi",   "ndvi_annual"),
    ("lst",    "lst_annual"),
    ("precip", "precip_annual"),
    ("srad",   "srad_annual"),
    ("hfi",    "hfp_annual"),
]


def _process_row(row_data):
    """Sen's slope + MK p-value for every pixel in one raster row (n_years × n_cols)."""
    n_years, n_cols = row_data.shape
    slopes = np.full(n_cols, np.nan, dtype=np.float32)
    pvalues = np.full(n_cols, np.nan, dtype=np.float32)
    for j in range(n_cols):
        ts = row_data[:, j]
        if not np.any(np.isnan(ts)):
            res = mk.original_test(ts)
            slopes[j] = res.slope
            pvalues[j] = res.p
    return slopes, pvalues


def _write_tif(path, arr2d, transform, crs):
    profile = dict(
        driver="GTiff", dtype="float32",
        width=arr2d.shape[1], height=arr2d.shape[0],
        count=1, crs=crs, transform=transform,
        nodata=float("nan"), compress="lzw",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2d.astype(np.float32)[np.newaxis])


def main():
    npz_path = "analysis_stacks.npz"
    if not os.path.exists(npz_path):
        sys.exit(f"ERROR: {npz_path} not found. Run the stacking script first.")

    data = np.load(npz_path)

    # Reconstruct transform (saved as 3×3 or 6-element array)
    t = data["transform"].flatten()
    transform = Affine(float(t[0]), float(t[1]), float(t[2]),
                       float(t[3]), float(t[4]), float(t[5]))
    crs = CRS.from_wkt(str(data["crs_wkt"]))

    os.makedirs("Trends", exist_ok=True)

    # ── MK power caveat ----------------------------------------------------──
    print("=" * 70)
    print("NOTE  Mann-Kendall power with n = 10")
    print("      For a two-sided test at alpha = 0.05, the minimum detectable |tau|")
    print("      is ~0.51 (Kendall S = +/-16 out of max 45).  That means only")
    print("      strong, near-monotonic trends will be flagged as significant.")
    print("      Low % significant pixels is expected behavior, not an error.")
    print("=" * 70 + "\n")

    summary = []

    for var_name, arr_key in VARIABLES:
        arr = data[arr_key].astype(np.float64)   # shape (10, H, W)
        _, height, width = arr.shape

        print(f"[{var_name}]  Sen's slope + original Mann-Kendall  ({height} rows x {width} cols) ...")

        row_results = Parallel(n_jobs=-1)(
            delayed(_process_row)(arr[:, i, :])
            for i in tqdm(range(height), desc=f"  {var_name}", unit="row", ncols=72)
        )

        slope_map = np.full((height, width), np.nan, dtype=np.float32)
        pval_map  = np.full((height, width), np.nan, dtype=np.float32)
        for i, (s_row, p_row) in enumerate(row_results):
            slope_map[i] = s_row
            pval_map[i]  = p_row

        out_slope = f"Trends/{var_name}_sen_slope.tif"
        out_pval  = f"Trends/{var_name}_mk_pvalue.tif"
        _write_tif(out_slope, slope_map, transform, crs)
        _write_tif(out_pval,  pval_map,  transform, crs)
        print(f"  saved {out_slope}")
        print(f"  saved {out_pval}")

        valid_mask = ~np.isnan(pval_map)
        n_valid  = int(valid_mask.sum())
        n_sig    = int((pval_map[valid_mask] < 0.05).sum())
        pct_sig  = 100.0 * n_sig / n_valid if n_valid else 0.0
        mean_s   = float(np.nanmean(slope_map))
        median_s = float(np.nanmedian(slope_map))

        summary.append((var_name, n_valid, n_sig, pct_sig, mean_s, median_s))
        print(f"  -> {pct_sig:.1f}% significant  |  mean slope {mean_s:+.5f}  |  median slope {median_s:+.5f}\n")

    # ── Final report ----------------------------------------------------─────
    sep = "=" * 78
    print(sep)
    print("TREND ANALYSIS RESULTS   2015-2024  n=10  growing-season annual means")
    print(sep)
    hdr = f"{'Variable':<10} {'Valid px':>9} {'Sig px':>8} {'% Sig':>7}  {'Mean slope':>13}  {'Median slope':>13}"
    print(hdr)
    print("-" * 78)
    for var_name, n_valid, n_sig, pct_sig, mean_s, median_s in summary:
        print(f"{var_name:<10} {n_valid:>9,} {n_sig:>8,} {pct_sig:>6.1f}%  {mean_s:>+13.5f}  {median_s:>+13.5f}")
    print(sep)

    print("""
WHAT WAS DONE
-------------
1. Loaded analysis_stacks.npz  (arrays shape 10 x 161 x 269, valid px ~20 132).
2. For every pixel with a complete 10-year time series (no NaN):
     - pymannkendall.original_test()  ->  Kendall tau, p-value (two-sided)
     - Sen's slope extracted from result.slope (Theil-Sen estimator)
   Pixels with any NaN year are written as NaN in both output rasters.
3. Parallelized over raster rows with joblib.Parallel(n_jobs=-1); progress
   tracked with tqdm.
4. Saved 10 LZW-compressed GeoTIFFs (EPSG:4326, same grid as input):
     Trends/ndvi_sen_slope.tif    Trends/ndvi_mk_pvalue.tif
     Trends/lst_sen_slope.tif     Trends/lst_mk_pvalue.tif
     Trends/precip_sen_slope.tif  Trends/precip_mk_pvalue.tif
     Trends/srad_sen_slope.tif    Trends/srad_mk_pvalue.tif
     Trends/hfi_sen_slope.tif     Trends/hfi_mk_pvalue.tif
5. Reported % significant pixels (p<0.05), mean and median Sen's slope
   for each variable (see table above).

STATISTICAL CAVEAT
------------------
Mann-Kendall with n=10 requires |tau| >= 0.51 to reach p<0.05 (two-sided).
This corresponds to a near-monotonic signal (e.g. 8 of 9 consecutive pairs
must increase/decrease). Variables such as NDVI, LST and precipitation in
a semi-arid region have high interannual variability; few pixels will show
trends strong enough to be detectable with only a decade of data.
""")


if __name__ == "__main__":
    main()
