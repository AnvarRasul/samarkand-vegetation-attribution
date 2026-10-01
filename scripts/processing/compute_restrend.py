"""
RESTREND: per-pixel OLS of NDVI on climate drivers (LST, Precip, SRAD),
then Mann-Kendall trend test on residuals.

Model: NDVI(t) = a*LST(t) + b*Precip(t) + c*SRAD(t) + delta + epsilon(t)
OLS is fully vectorized via batched normal equations (einsum + pinv).
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

N_YEARS = 10
N_PRED  = 3        # LST, Precip, SRAD (intercept is the 4th param)
DOF     = N_YEARS - N_PRED - 1   # = 6


def _write_tif(path, arr2d, transform, crs, nodata=float("nan")):
    profile = dict(
        driver="GTiff", dtype="float32",
        width=arr2d.shape[1], height=arr2d.shape[0],
        count=1, crs=crs, transform=transform,
        nodata=nodata, compress="lzw",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2d.astype(np.float32)[np.newaxis])


def _mk_row(row_data):
    """Sen's slope + MK p-value for each pixel in one row (n_years x n_cols)."""
    n_years, n_cols = row_data.shape
    slopes = np.full(n_cols, np.nan, dtype=np.float32)
    pvals  = np.full(n_cols, np.nan, dtype=np.float32)
    for j in range(n_cols):
        ts = row_data[:, j]
        if not np.any(np.isnan(ts)):
            res = mk.original_test(ts)
            slopes[j] = res.slope
            pvals[j]  = res.p
    return slopes, pvals


def main():
    npz_path = "analysis_stacks.npz"
    if not os.path.exists(npz_path):
        sys.exit(f"ERROR: {npz_path} not found.")

    data   = np.load(npz_path)
    ndvi   = data["ndvi_annual"].astype(np.float64)    # (10, H, W)
    lst    = data["lst_annual"].astype(np.float64)
    precip = data["precip_annual"].astype(np.float64)
    srad   = data["srad_annual"].astype(np.float64)
    valid  = data["valid_mask"]                        # (H, W) bool, 20132 px

    _, height, width = ndvi.shape

    t = data["transform"].flatten()
    transform = Affine(float(t[0]), float(t[1]), float(t[2]),
                       float(t[3]), float(t[4]), float(t[5]))
    crs = CRS.from_wkt(str(data["crs_wkt"]))

    os.makedirs("RESTREND", exist_ok=True)

    # ── DOF / adj-R2 caveat ───────────────────────────────────────────────
    print("=" * 70)
    print("DOF NOTE: n=10, 4 fitted parameters (LST, Precip, SRAD, intercept).")
    print(f"          Residual dof = n - k - 1 = {DOF}.")
    print("          Raw R2 is biased upward with only 6 dof; it rises")
    print("          toward 1 even for noise when k/n is large.")
    print(f"          Adjusted-R2 = 1 - (1-R2)*{N_YEARS-1}/{DOF}  corrects for this")
    print("          and is the meaningful model quality metric.")
    print("          It can be negative when the model fits worse than the mean.")
    print("=" * 70 + "\n")

    n_valid = int(valid.sum())
    print(f"Valid pixels (pre-masked): {n_valid:,} / {height*width:,}\n")

    # ── Step 1: Flatten to (n_valid, n_years) ────────────────────────────
    def flat(arr):      # (10, H, W) -> (n_valid, 10)
        return arr[:, valid].T

    Y = flat(ndvi)      # (n_valid, 10)
    T = flat(lst)
    P = flat(precip)
    S = flat(srad)

    # ── Step 2: Vectorized OLS ────────────────────────────────────────────
    # Design matrix X: (n_valid, 10, 4)  columns: [LST, Precip, SRAD, 1]
    ones = np.ones((n_valid, N_YEARS), dtype=np.float64)
    X = np.stack([T, P, S, ones], axis=2)   # (n_valid, 10, 4)

    print("Building normal equations (vectorized einsum) ...")
    XtX = np.einsum("pti,ptj->pij", X, X)   # (n_valid, 4, 4)
    Xty = np.einsum("pti,pt->pi",  X, Y)    # (n_valid, 4)

    # Solve via batched pseudoinverse (handles near-singular pixels safely)
    print("Solving via batched pseudoinverse (np.linalg.pinv) ...")
    XtX_inv = np.linalg.pinv(XtX)           # (n_valid, 4, 4)
    coef = np.einsum("pij,pj->pi", XtX_inv, Xty)  # (n_valid, 4)

    a_T   = coef[:, 0]   # LST coefficient
    b_P   = coef[:, 1]   # Precip coefficient
    c_S   = coef[:, 2]   # SRAD coefficient
    delta = coef[:, 3]   # intercept

    # ── Step 3: Predictions, residuals, R2 ───────────────────────────────
    Y_pred = np.einsum("pti,pi->pt", X, coef)      # (n_valid, 10)
    resid  = Y - Y_pred                             # (n_valid, 10)

    SS_res = np.sum(resid ** 2, axis=1)
    SS_tot = np.sum((Y - Y.mean(axis=1, keepdims=True)) ** 2, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        R2    = np.where(SS_tot > 0, 1.0 - SS_res / SS_tot, np.nan)
        adjR2 = np.where(np.isfinite(R2),
                         1.0 - (1.0 - R2) * (N_YEARS - 1) / DOF,
                         np.nan)

    # ── Step 4: Save coefficient rasters ─────────────────────────────────
    def unflat(vec):
        out = np.full((height, width), np.nan, dtype=np.float32)
        out[valid] = vec.astype(np.float32)
        return out

    rasters = {
        "RESTREND/coef_T.tif":     a_T,
        "RESTREND/coef_P.tif":     b_P,
        "RESTREND/coef_S.tif":     c_S,
        "RESTREND/intercept.tif":  delta,
        "RESTREND/R2.tif":         R2,
        "RESTREND/adjR2.tif":      adjR2,
    }
    for path, vec in rasters.items():
        _write_tif(path, unflat(vec), transform, crs)
        print(f"  saved {path}")

    # ── Step 5: Save residuals.npz ────────────────────────────────────────
    resid_3d = np.full((N_YEARS, height, width), np.nan, dtype=np.float32)
    resid_3d[:, valid] = resid.T.astype(np.float32)

    np.savez_compressed(
        "RESTREND/residuals.npz",
        residuals=resid_3d,
        valid_mask=valid,
        transform=data["transform"],
        crs_wkt=data["crs_wkt"],
        years=data["years"],
    )
    print("  saved RESTREND/residuals.npz")

    # ── Step 6: Sen+MK on residuals ───────────────────────────────────────
    print("\nRunning Sen's slope + Mann-Kendall on residuals ...")
    row_results = Parallel(n_jobs=-1)(
        delayed(_mk_row)(resid_3d[:, i, :])
        for i in tqdm(range(height), desc="  residual MK", unit="row", ncols=72)
    )

    slope_map = np.full((height, width), np.nan, dtype=np.float32)
    pval_map  = np.full((height, width), np.nan, dtype=np.float32)
    for i, (s_row, p_row) in enumerate(row_results):
        slope_map[i] = s_row
        pval_map[i]  = p_row

    _write_tif("RESTREND/residual_sen_slope.tif", slope_map, transform, crs)
    _write_tif("RESTREND/residual_mk_pvalue.tif", pval_map,  transform, crs)
    print("  saved RESTREND/residual_sen_slope.tif")
    print("  saved RESTREND/residual_mk_pvalue.tif")

    # ── Step 7: Report ────────────────────────────────────────────────────
    n_fin = int(np.isfinite(R2).sum())
    pct_R2_gt50    = 100.0 * (R2[np.isfinite(R2)]    > 0.5).sum() / n_fin
    pct_adjR2_gt50 = 100.0 * (adjR2[np.isfinite(adjR2)] > 0.5).sum() / n_fin
    n_adjR2_neg    = int((adjR2[np.isfinite(adjR2)] < 0).sum())

    pv = pval_map[~np.isnan(pval_map)]
    n_sig    = int((pv < 0.05).sum())
    pct_sig  = 100.0 * n_sig / len(pv) if len(pv) else 0.0
    mean_sl  = float(np.nanmean(slope_map))
    med_sl   = float(np.nanmedian(slope_map))

    sep = "=" * 72
    print("\n" + sep)
    print("RESTREND RESULTS   2015-2024   n=10   dof=6")
    print(sep)

    print(f"\n  Model: NDVI = a*LST + b*Precip + c*SRAD + delta")
    print(f"\n  {'Metric':<14} {'Mean':>8} {'Median':>8} {'% > 0.5':>9}")
    print(f"  {'-'*44}")
    print(f"  {'R2':<14} {np.nanmean(R2):>8.3f} {np.nanmedian(R2):>8.3f} {pct_R2_gt50:>8.1f}%")
    print(f"  {'adj-R2':<14} {np.nanmean(adjR2):>8.3f} {np.nanmedian(adjR2):>8.3f} {pct_adjR2_gt50:>8.1f}%")
    print(f"  adj-R2 < 0 in {n_adjR2_neg:,} pixels ({100.0*n_adjR2_neg/n_fin:.1f}%) -- model worse than mean")

    print(f"\n  Coefficient medians over valid pixels:")
    print(f"  {'a_T  (LST)':<22}: {float(np.nanmedian(a_T)):>+10.5f}  NDVI per degC")
    print(f"  {'b_P  (Precip)':<22}: {float(np.nanmedian(b_P)):>+10.5f}  NDVI per mm/month")
    print(f"  {'c_S  (SRAD)':<22}: {float(np.nanmedian(c_S)):>+10.5f}  NDVI per W/m2")
    print(f"  {'delta (intercept)':<22}: {float(np.nanmedian(delta)):>+10.5f}")

    print(f"\n  Residual trend (Sen+MK):")
    print(f"  % significant (p<0.05) : {pct_sig:.1f}%  ({n_sig:,} / {len(pv):,} pixels)")
    print(f"  Mean residual slope    : {mean_sl:>+.5f}  NDVI/year")
    print(f"  Median residual slope  : {med_sl:>+.5f}  NDVI/year")
    print("\n" + sep)

    print("""
WHAT WAS DONE
-------------
1. Loaded analysis_stacks.npz. Used pre-computed valid_mask (20,132 pixels
   with valid data in all 5 variables across all 10 years).

2. Built design matrix X (n_valid x 10 x 4) with columns [LST, Precip, SRAD, 1].
   Solved OLS for all pixels simultaneously using batched normal equations:
     XtX = einsum("pti,ptj->pij", X, X)   shape (n_valid, 4, 4)
     Xty = einsum("pti,pt->pi",  X, Y)    shape (n_valid, 4)
     coef = pinv(XtX) @ Xty               shape (n_valid, 4)
   np.linalg.pinv handles near-singular pixels via SVD (no pixel skipped).

3. Computed R2 and adjusted-R2 (n=10, k=3, dof=6) for every valid pixel.
   adj-R2 = 1 - (1-R2)*(n-1)/(n-k-1) = 1 - (1-R2)*9/6.
   Pixels with SS_tot=0 (constant NDVI) receive NaN.

4. Saved 6 LZW-compressed GeoTIFFs to RESTREND/ (float32, EPSG:4326):
     coef_T.tif    coef_P.tif    coef_S.tif
     intercept.tif  R2.tif        adjR2.tif

5. Reshaped residuals (n_valid, 10) back to (10, 161, 269) with NaN fill
   for invalid pixels. Saved RESTREND/residuals.npz with valid_mask,
   transform, crs_wkt, and years arrays for downstream use.

6. Ran pymannkendall.original_test + Theil-Sen slope on residuals using
   joblib.Parallel(n_jobs=-1) over rows with tqdm progress bar.
   Saved RESTREND/residual_sen_slope.tif, RESTREND/residual_mk_pvalue.tif.

7. Reported R2 / adj-R2 distribution, coefficient medians, and residual
   trend significance (see table above).

INTERPRETATION NOTE
-------------------
- Significant residual trends (after removing climate signal) indicate
  non-climatic NDVI change (land-use change, degradation, greening).
- With n=10 and dof=6, MK still requires near-monotonic residuals for
  p<0.05, so low % significant is expected.
- Negative adj-R2 pixels: the climate variables explain less variance
  than the unconditional mean; NDVI in those pixels is not well driven
  by these three predictors on annual timescales.
""")


if __name__ == "__main__":
    main()
