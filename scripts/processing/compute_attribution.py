"""
Final attribution and headline 6-class map.

Climate Contribution (CC) = coef_T*lst_slope + coef_P*precip_slope + coef_S*srad_slope
Human Attribution (HA)    = coef_HFI * hfi_slope
  where coef_HFI is from per-pixel OLS: RESTREND-residual ~ HFI_annual

6-class map (priority order: co-driven checked first):
  1 Climate-dom restoration   NDVI_slope>0, CC>0, HA>0, CC>HA
  2 Human-dom restoration     NDVI_slope>0, CC>0, HA>0, HA>=CC
  3 Co-driven restoration     NDVI_slope>0, |CC-HA|/max(|CC|,|HA|) < 0.2
  4 Climate-dom degradation   NDVI_slope<0, CC<0, |CC|>|HA|
  5 Human-dom degradation     NDVI_slope<0, HA<0, |HA|>|CC|
  6 Co-driven degradation     NDVI_slope<0, CC<0, HA<0, |CC-HA|/max(|CC|,|HA|) < 0.2
  0 Unclassified              all other valid pixels
"""

import os
import sys
import math
import numpy as np
import rasterio
from rasterio.crs import CRS


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read1(path):
    """Read band 1 as float32; return (array, profile)."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.profile


def save_f32(path, arr, transform, crs):
    profile = dict(
        driver="GTiff", dtype="float32", count=1,
        width=arr.shape[1], height=arr.shape[0],
        crs=crs, transform=transform,
        nodata=float("nan"), compress="lzw",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32)[np.newaxis])


def save_u8(path, arr, transform, crs):
    profile = dict(
        driver="GTiff", dtype="uint8", count=1,
        width=arr.shape[1], height=arr.shape[0],
        crs=crs, transform=transform,
        nodata=0, compress="lzw",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint8)[np.newaxis])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load regression coefficients ──────────────────────────────────
    coef_T, prof = read1("RESTREND/coef_T.tif")
    coef_P, _    = read1("RESTREND/coef_P.tif")
    coef_S, _    = read1("RESTREND/coef_S.tif")

    transform = prof["transform"]
    crs       = prof["crs"]
    height, width = coef_T.shape

    # ── 2. Load Sen slope rasters ─────────────────────────────────────────
    lst_slope,    _ = read1("Trends/lst_sen_slope.tif")
    precip_slope, _ = read1("Trends/precip_sen_slope.tif")
    srad_slope,   _ = read1("Trends/srad_sen_slope.tif")
    hfi_slope,    _ = read1("Trends/hfi_sen_slope.tif")
    ndvi_slope,   _ = read1("Trends/ndvi_sen_slope.tif")

    # Valid pixel mask
    stacks = np.load("analysis_stacks.npz")
    valid  = stacks["valid_mask"]        # (H, W) bool, 20 132 px
    n_valid = int(valid.sum())

    # ── 3. Climate Contribution ───────────────────────────────────────────
    CC = coef_T * lst_slope + coef_P * precip_slope + coef_S * srad_slope
    CC[~valid] = np.nan

    # ── 4. Human Attribution: residual ~ HFI per pixel ───────────────────
    print("Fitting RESTREND-residual ~ HFI_annual (vectorized OLS) ...")

    resid_3d = np.load("RESTREND/residuals.npz")["residuals"].astype(np.float64)
    hfp_3d   = stacks["hfp_annual"].astype(np.float64)

    # Flatten valid pixels to (n_valid, 10)
    resid_flat = resid_3d[:, valid].T    # (n_valid, 10)
    hfp_flat   = hfp_3d[:, valid].T     # (n_valid, 10)

    ones   = np.ones((n_valid, 10), dtype=np.float64)
    X_hfi  = np.stack([hfp_flat, ones], axis=2)          # (n_valid, 10, 2)
    XtX    = np.einsum("pti,ptj->pij", X_hfi, X_hfi)     # (n_valid, 2, 2)
    Xty    = np.einsum("pti,pt->pi",   X_hfi, resid_flat) # (n_valid, 2)
    coef_h = np.einsum("pij,pj->pi",   np.linalg.pinv(XtX), Xty)  # (n_valid, 2)

    e_flat = coef_h[:, 0].astype(np.float32)             # HFI sensitivity

    e_map = np.full((height, width), np.nan, dtype=np.float32)
    e_map[valid] = e_flat

    HA = e_map * hfi_slope
    HA[~valid] = np.nan

    # ── 5. Save CC, HA, coef_HFI ─────────────────────────────────────────
    save_f32("RESTREND/CC.tif",        CC,    transform, crs)
    save_f32("RESTREND/HA.tif",        HA,    transform, crs)
    save_f32("RESTREND/coef_HFI.tif",  e_map, transform, crs)
    print("  saved RESTREND/CC.tif")
    print("  saved RESTREND/HA.tif")
    print("  saved RESTREND/coef_HFI.tif")

    # ── 6. Relative contributions ─────────────────────────────────────────
    abs_CC = np.abs(CC)
    abs_HA = np.abs(HA)
    total  = abs_CC + abs_HA
    with np.errstate(invalid="ignore", divide="ignore"):
        CC_pct = np.where(total > 0, abs_CC / total * 100.0, np.nan).astype(np.float32)
        HA_pct = np.where(total > 0, abs_HA / total * 100.0, np.nan).astype(np.float32)
    save_f32("RESTREND/CC_pct.tif", CC_pct, transform, crs)
    save_f32("RESTREND/HA_pct.tif", HA_pct, transform, crs)
    print("  saved RESTREND/CC_pct.tif, RESTREND/HA_pct.tif")

    # ── 7. 6-class attribution ────────────────────────────────────────────
    # Similarity ratio: |CC - HA| / max(|CC|, |HA|)
    max_abs = np.maximum(abs_CC, abs_HA)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(max_abs > 0,
                         np.abs(CC - HA) / max_abs,
                         np.inf).astype(np.float32)

    cls  = np.zeros((height, width), dtype=np.uint8)
    rest = valid & (ndvi_slope > 0)
    degr = valid & (ndvi_slope < 0)

    # --- Co-driven checks first (priority) ---
    cls[rest & (ratio < 0.2)] = 3                         # co-driven restoration
    cls[degr & (CC < 0) & (HA < 0) & (ratio < 0.2)] = 6  # co-driven degradation

    # --- Dominant driver ---
    # Restoration (must not already be class 3)
    m_rest_free = rest & (cls == 0)
    cls[m_rest_free & (CC > 0) & (HA > 0) & (CC > HA)]  = 1  # climate-dom restoration
    cls[m_rest_free & (CC > 0) & (HA > 0) & (HA >= CC)] = 2  # human-dom restoration

    # Degradation (must not already be class 6)
    m_degr_free = degr & (cls == 0)
    cls[m_degr_free & (CC < 0) & (abs_CC > abs_HA)] = 4  # climate-dom degradation
    cls[m_degr_free & (HA < 0) & (abs_HA > abs_CC)] = 5  # human-dom degradation

    save_u8("RESTREND/attribution_6class.tif", cls, transform, crs)
    print("  saved RESTREND/attribution_6class.tif")

    # ── 8. Area / percentage table ────────────────────────────────────────
    # Pixel area at centre latitude
    lat_ctr  = transform.f + transform.e * (height / 2.0)
    px_deg   = abs(transform.a)
    px_km_ns = px_deg * 111.0
    px_km_ew = px_deg * 111.0 * math.cos(math.radians(lat_ctr))
    px_km2   = px_km_ns * px_km_ew

    class_labels = {
        1: "Climate-dom restoration",
        2: "Human-dom restoration  ",
        3: "Co-driven restoration  ",
        4: "Climate-dom degradation",
        5: "Human-dom degradation  ",
        6: "Co-driven degradation  ",
        0: "Unclassified           ",
    }

    sep = "=" * 72
    print("\n" + sep)
    print(f"ATTRIBUTION MAP RESULTS  (pixel area ~{px_km2:.3f} km2 at lat {lat_ctr:.2f} deg)")
    print(sep)
    print(f"  {'Cl':<3} {'Class':<28} {'Pixels':>8} {'Area km2':>10} {'% valid':>9}")
    print("  " + "-" * 64)

    total_area = 0.0
    restoration_px = 0
    degradation_px = 0
    for cid in [1, 2, 3, 4, 5, 6, 0]:
        n_px = int((cls[valid] == cid).sum())
        area = n_px * px_km2
        pct  = 100.0 * n_px / n_valid
        print(f"  {cid:<3} {class_labels[cid]:<28} {n_px:>8,} {area:>10.1f} {pct:>8.1f}%")
        total_area += area
        if cid in (1, 2, 3):
            restoration_px += n_px
        elif cid in (4, 5, 6):
            degradation_px += n_px

    print("  " + "-" * 64)
    print(f"  {'':3} {'Total valid':<28} {n_valid:>8,} {total_area:>10.1f} {'100.0%':>9}")
    print(sep)

    print(f"\n  Net summary over classified pixels:")
    n_cls = restoration_px + degradation_px
    print(f"    Restoration classes (1+2+3): {restoration_px:>6,}  "
          f"({100.0*restoration_px/n_cls:.1f}% of classified)" if n_cls else "")
    print(f"    Degradation classes (4+5+6): {degradation_px:>6,}  "
          f"({100.0*degradation_px/n_cls:.1f}% of classified)" if n_cls else "")

    # CC and HA stats on valid pixels
    cc_v  = CC[valid]
    ha_v  = HA[valid]
    e_v   = e_map[valid]
    print(f"\n  Climate Contribution (CC) -- valid pixels:")
    print(f"    mean={float(np.nanmean(cc_v)):+.5f}  median={float(np.nanmedian(cc_v)):+.5f}  "
          f"range [{float(np.nanmin(cc_v)):+.5f}, {float(np.nanmax(cc_v)):+.5f}]")
    print(f"  Human Attribution (HA):")
    print(f"    mean={float(np.nanmean(ha_v)):+.5f}  median={float(np.nanmedian(ha_v)):+.5f}  "
          f"range [{float(np.nanmin(ha_v)):+.5f}, {float(np.nanmax(ha_v)):+.5f}]")
    print(f"  coef_HFI (e, NDVI per HFI unit):")
    print(f"    mean={float(np.nanmean(e_v)):+.5f}  median={float(np.nanmedian(e_v)):+.5f}")

    print("""
WHAT WAS DONE
-------------
1. Loaded coef_T, coef_P, coef_S from RESTREND/ (OLS coefficients of the
   NDVI ~ climate model computed in compute_restrend.py).

2. Loaded lst_sen_slope, precip_sen_slope, srad_sen_slope, hfi_sen_slope,
   ndvi_sen_slope from Trends/ (Theil-Sen slopes from compute_trends.py).

3. Computed Climate Contribution (CC) per pixel:
     CC = coef_T*lst_slope + coef_P*precip_slope + coef_S*srad_slope
   CC represents the component of the NDVI trend explained by climate trends.

4. Computed Human Attribution (HA) per pixel:
   a. Fitted per-pixel OLS: RESTREND-residual(t) = e*HFI_annual(t) + intercept
      Fully vectorized via batched normal equations (einsum + pinv), same
      approach as compute_restrend.py.  Saved coef_HFI.tif (e).
   b. HA = e * hfi_sen_slope
      HA is the component of the residual trend driven by human footprint change.

5. Saved to RESTREND/:
     CC.tif          -- climate contribution to NDVI trend (NDVI/year)
     HA.tif          -- human attribution to NDVI trend (NDVI/year)
     coef_HFI.tif    -- residual sensitivity to HFI (NDVI per HFI unit)
     CC_pct.tif      -- |CC|/(|CC|+|HA|)*100
     HA_pct.tif      -- |HA|/(|CC|+|HA|)*100

6. Built 6-class attribution raster (RESTREND/attribution_6class.tif, uint8):
   Priority rule: co-driven classes (3, 6) checked FIRST to avoid overlap.
     Class 1 -- Climate-dom restoration   NDVI_slope>0, CC>0, HA>0, CC>HA
     Class 2 -- Human-dom restoration     NDVI_slope>0, CC>0, HA>0, HA>=CC
     Class 3 -- Co-driven restoration     NDVI_slope>0, ratio<0.2
     Class 4 -- Climate-dom degradation   NDVI_slope<0, CC<0, |CC|>|HA|
     Class 5 -- Human-dom degradation     NDVI_slope<0, HA<0, |HA|>|CC|
     Class 6 -- Co-driven degradation     NDVI_slope<0, CC<0, HA<0, ratio<0.2
     Class 0 -- Unclassified              pixels with mixed/unclear drivers
   where ratio = |CC - HA| / max(|CC|, |HA|)

7. Reported pixel count, area (km2) and % of valid pixels per class.
""")


if __name__ == "__main__":
    main()
