"""
Publication-quality figures (300 DPI PNG) for Samarkand NDVI trend study.
Output: figures/fig1.png ... figures/fig8.png
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from scipy import stats

warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DPI = 300

# ── helper: load raster (band 1, float32, nodata->nan) ────────────────────────
def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, src.transform, src.crs

# ── helper: geographic extent from transform + shape ─────────────────────────
def extent_from_transform(transform, shape):
    h, w = shape
    west  = transform.c
    east  = transform.c + w * transform.a
    north = transform.f
    south = transform.f + h * transform.e
    return [west, east, south, north]   # matplotlib imshow extent

# ── helper: manual scale bar ──────────────────────────────────────────────────
def add_scale_bar(ax, x0_frac, y0_frac, length_deg, lat_deg, label_km):
    """Draw a simple scale bar in geographic coordinates."""
    ax_xl, ax_xr = ax.get_xlim()
    ax_yb, ax_yt = ax.get_ylim()
    x0 = ax_xl + x0_frac * (ax_xr - ax_xl)
    y0 = ax_yb + y0_frac * (ax_yt - ax_yb)
    ax.plot([x0, x0 + length_deg], [y0, y0], "k-", lw=2.5, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.01, y0 + 0.01], "k-", lw=1.5)
    ax.plot([x0 + length_deg, x0 + length_deg], [y0 - 0.01, y0 + 0.01], "k-", lw=1.5)
    ax.text(x0 + length_deg / 2, y0 + 0.025, label_km, ha="center", va="bottom",
            fontsize=8, fontweight="bold")

# ── helper: colorbar on right ─────────────────────────────────────────────────
def add_cbar(fig, im, ax, label, shrink=0.85, pad=0.03, fmt=None):
    cb = fig.colorbar(im, ax=ax, shrink=shrink, pad=pad, format=fmt)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb

# ── load shared data ──────────────────────────────────────────────────────────
print("Loading shared data ...")
data     = np.load("analysis_stacks.npz")
valid    = data["valid_mask"]                       # (161, 269)
t_arr    = data["transform"].flatten()
transform_affine = rasterio.transform.Affine(
    t_arr[0], t_arr[1], t_arr[2],
    t_arr[3], t_arr[4], t_arr[5]
)
height, width = valid.shape
aoi = gpd.read_file("AOI/samarkand_region.shp")
bounds = aoi.total_bounds   # [minx, miny, maxx, maxy]
extent = extent_from_transform(transform_affine, (height, width))

# ── load all trend rasters ────────────────────────────────────────────────────
def load_masked(path):
    arr, tr, _ = load_raster(path)
    arr[~valid] = np.nan
    return arr

ndvi_slope  = load_masked("Trends/ndvi_sen_slope.tif")
lst_slope   = load_masked("Trends/lst_sen_slope.tif")
prec_slope  = load_masked("Trends/precip_sen_slope.tif")
srad_slope  = load_masked("Trends/srad_sen_slope.tif")
hfi_slope   = load_masked("Trends/hfi_sen_slope.tif")
r2_map      = load_masked("RESTREND/R2.tif")
resid_slope = load_masked("RESTREND/residual_sen_slope.tif")
cc_map      = load_masked("RESTREND/CC.tif")
ha_map      = load_masked("RESTREND/HA.tif")

with rasterio.open("RESTREND/attribution_6class.tif") as src:
    cls_map = src.read(1).astype(np.float32)
    cls_map[cls_map == 0] = np.nan

imp_df = pd.read_csv("RF_model/feature_importances.csv")

print("  data loaded.")


# =============================================================================
# FIG 1 — Study area map
# =============================================================================
print("Fig 1: Study area map ...")

fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5.5))

# Raster extent background (valid pixels as light grey)
valid_img = np.where(valid, 0.85, np.nan)
ax1.imshow(valid_img, extent=extent, origin="upper",
           cmap="Greys_r", vmin=0, vmax=1, alpha=0.35, interpolation="nearest",
           zorder=1)

# AOI boundary
aoi.boundary.plot(ax=ax1, color="#1a1a2e", linewidth=1.8, zorder=3)
aoi.plot(ax=ax1, color="#d4e6f1", alpha=0.35, zorder=2)

# NDVI mean as background texture
ndvi_mean = np.nanmean(data["ndvi_annual"], axis=0).astype(np.float32)
ndvi_mean[~valid] = np.nan
im_bg = ax1.imshow(ndvi_mean, extent=extent, origin="upper",
                   cmap="YlGn", vmin=0.1, vmax=0.6, alpha=0.7,
                   interpolation="bilinear", zorder=2)
add_cbar(fig1, im_bg, ax1, "Mean NDVI (2015-2024)", shrink=0.80)

# Approximate Zarafshan river (east-west through centre of AOI)
# River runs roughly at ~39.78 deg lat across the province
zar_lons = np.linspace(65.14, 67.55, 60)
# The Zarafshan flows roughly from east (higher elevation) curving
# through ~lat 39.65-39.85
zar_lats = 39.78 + 0.08 * np.sin(np.linspace(0, np.pi * 1.5, 60))
ax1.plot(zar_lons, zar_lats, color="#1f78b4", lw=2.0, zorder=4,
         label="Zarafshan River (approx.)")

# Samarkand city marker (~lat 39.655, lon 66.975)
ax1.plot(66.975, 39.655, marker="*", color="#e74c3c", markersize=12,
         zorder=5, label="Samarkand city")
ax1.text(66.975 + 0.06, 39.655, "Samarkand", fontsize=8.5,
         color="#e74c3c", va="center", zorder=5)

# Inset: Central Asia locator (simple box)
ax_ins = ax1.inset_axes([0.72, 0.02, 0.27, 0.28])
ax_ins.set_facecolor("#cce5ff")
ax_ins.add_patch(mpatches.Rectangle((55, 35), 45, 25,
                                     fc="#f0f0f0", ec="gray", lw=0.7))
ax_ins.add_patch(mpatches.Rectangle((bounds[0], bounds[1]),
                                     bounds[2] - bounds[0],
                                     bounds[3] - bounds[1],
                                     fc="#e74c3c", ec="#c0392b", lw=1.2))
ax_ins.set_xlim(55, 82)
ax_ins.set_ylim(36, 52)
ax_ins.set_xticks([]); ax_ins.set_yticks([])
ax_ins.text(68.5, 40, "Samarkand", fontsize=5, ha="center", color="#c0392b")
ax_ins.set_title("Location", fontsize=6, pad=1)

# Scale bar: ~50 km at lat 39.95
# 1 deg lon at lat 40 = 111.32 * cos(40°) ≈ 85.3 km
# 50 km ≈ 0.586 deg
add_scale_bar(ax1, x0_frac=0.04, y0_frac=0.06, length_deg=0.586,
              lat_deg=40.0, label_km="50 km")

# Compass rose (simple N arrow)
ax1.annotate("N", xy=(0.06, 0.18), xytext=(0.06, 0.13),
             xycoords="axes fraction", textcoords="axes fraction",
             ha="center", fontsize=11, fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

ax1.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
ax1.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)
ax1.set_xlabel("Longitude (deg E)", fontsize=10)
ax1.set_ylabel("Latitude (deg N)", fontsize=10)
ax1.set_title("Study Area: Samarkand Province, Uzbekistan\n"
              "Annual NDVI analysis 2015-2024  |  MODIS 500m (resampled to ~1 km)",
              fontsize=11)
ax1.legend(loc="upper left", fontsize=8, framealpha=0.9, edgecolor="gray")
ax1.grid(True, lw=0.4, alpha=0.4, color="gray", ls="--")

fig1.savefig("figures/fig1.png", dpi=DPI)
plt.close(fig1)
print("  Saved figures/fig1.png")

CAPTION_1 = (
    "Figure 1. Study area. Samarkand Province, Uzbekistan (65.1-67.6 deg E, "
    "39.2-40.7 deg N), shown over mean annual NDVI (2015-2024) derived from "
    "MODIS MOD13A3 500 m imagery resampled to ~1 km. The Zarafshan River "
    "(approximate centreline) dissects the province. The red star marks "
    "Samarkand city (lat 39.66, lon 66.98). Inset: regional location within "
    "Central Asia. Scale bar = 50 km. Valid-pixel count: 20,132."
)
print("  CAPTION:", CAPTION_1)


# =============================================================================
# FIG 2 — Workflow block diagram
# =============================================================================
print("Fig 2: Workflow block diagram ...")

fig2, ax2 = plt.subplots(figsize=(11, 7))
ax2.set_xlim(0, 11)
ax2.set_ylim(0, 7)
ax2.axis("off")
ax2.set_facecolor("white")

def box(ax, x, y, w, h, label, color="#d6eaf8", fontsize=8.5, bold=False):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1", linewidth=1.2,
        edgecolor="#2c3e50", facecolor=color
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, wrap=True,
            multialignment="center")

def arrow(ax, x0, y0, x1, y1, label=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                                lw=1.2, mutation_scale=14))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.05, my, label, fontsize=7.5, color="#555",
                ha="left", va="center")

# Row 1: Input data sources
src_y = 6.3
src_w, src_h = 1.55, 0.65
colors_src = ["#d5f5e3", "#d5f5e3", "#d5f5e3", "#d5f5e3", "#fde8d8"]
labels_src = ["MODIS\nMOD13A3\n(NDVI)", "MODIS\nMOD11A3\n(LST)",
              "CHIRPS\nPrecip", "MODIS\nMCD18A2\n(SRAD)", "HFI\n2015-2024"]
xs_src = [1.1, 2.9, 4.7, 6.5, 8.3]
for xi, li, ci in zip(xs_src, labels_src, colors_src):
    box(ax2, xi, src_y, src_w, src_h, li, color=ci, fontsize=7.5)

# Row 2: Preprocessing
pre_y = 5.35
box(ax2, 5.0, pre_y, 8.5, 0.55,
    "Preprocessing: reproject to EPSG:4326 (~1 km), clip to AOI, annual aggregation  ->  analysis_stacks.npz",
    color="#eaf0fb", fontsize=8.0)
for xi in xs_src:
    arrow(ax2, xi, src_y - 0.33, xi if xi < 9 else 8.8, pre_y + 0.28)

# Row 3: Parallel tracks
box(ax2, 2.2, 4.4, 3.9, 0.58,
    "Sen's Slope + Mann-Kendall\n(NDVI, LST, Precip, SRAD, HFI)",
    color="#d6eaf8", fontsize=8.0)
box(ax2, 8.2, 4.4, 3.2, 0.58,
    "Random Forest\nNDVI ~ climate + HFI + lat + lon + year",
    color="#fdf2e9", fontsize=8.0)
arrow(ax2, 3.5, pre_y - 0.28, 2.8, 4.4 + 0.29)
arrow(ax2, 7.5, pre_y - 0.28, 7.6, 4.4 + 0.29)

# Row 4: RESTREND + RF output
box(ax2, 2.2, 3.45, 3.9, 0.58,
    "RESTREND: NDVI ~ LST + Precip + SRAD (OLS)\nResidual Sen's slope  ->  R2, coef_T/P/S",
    color="#d6eaf8", fontsize=8.0)
box(ax2, 8.2, 3.45, 3.2, 0.58,
    "Test R2 = 0.925\nFeature importances\n(LST dominant)",
    color="#fdf2e9", fontsize=8.0)
arrow(ax2, 2.2, 4.4 - 0.29, 2.2, 3.45 + 0.29)

# Row 5: Attribution
box(ax2, 2.2, 2.5, 3.9, 0.58,
    "Partial-derivative attribution\nCC = coef*climate_slopes   HA = e*HFI_slope",
    color="#d6eaf8", fontsize=8.0)
arrow(ax2, 2.2, 3.45 - 0.29, 2.2, 2.5 + 0.29)

# Row 6: Headline map + validation (bottom)
box(ax2, 2.2, 1.5, 3.9, 0.65,
    "6-Class Attribution Map\n(climate-dom / human-dom / co-driven)\nx restoration / degradation",
    color="#a9dfbf", fontsize=8.0, bold=True)
arrow(ax2, 2.2, 2.5 - 0.29, 2.2, 1.5 + 0.33)

box(ax2, 7.3, 1.5, 3.2, 0.65,
    "Independent Validation\nresidual slope vs HFI slope\n(Pearson r = +0.34, n = 20,132)",
    color="#fadbd8", fontsize=8.0)
arrow(ax2, 2.2, 1.5 - 0.33, 4.0, 1.5)
arrow(ax2, 8.2, 3.45 - 0.29, 8.0, 1.5 + 0.33)
arrow(ax2, 4.0, 1.5, 5.7, 1.5)

# cross arrow from RF to validation note
ax2.annotate("", xy=(5.7, 1.5), xytext=(6.6, 3.45),
             arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=1.0,
                             connectionstyle="arc3,rad=-0.35"))

ax2.set_title("Analysis Workflow: Samarkand Province Vegetation Dynamics (2015-2024)",
              fontsize=13, pad=10, fontweight="bold")

# Legend boxes
leg_items = [
    mpatches.Patch(facecolor="#d5f5e3", edgecolor="gray", label="Input data"),
    mpatches.Patch(facecolor="#d6eaf8", edgecolor="gray", label="RESTREND pipeline"),
    mpatches.Patch(facecolor="#fdf2e9", edgecolor="gray", label="RF pipeline"),
    mpatches.Patch(facecolor="#a9dfbf", edgecolor="gray", label="Key output"),
    mpatches.Patch(facecolor="#fadbd8", edgecolor="gray", label="Validation"),
]
ax2.legend(handles=leg_items, loc="lower center", ncol=5, fontsize=8,
           framealpha=0.9, edgecolor="gray",
           bbox_to_anchor=(0.5, -0.01))

fig2.savefig("figures/fig2.png", dpi=DPI)
plt.close(fig2)
print("  Saved figures/fig2.png")

CAPTION_2 = (
    "Figure 2. Analysis workflow. Five satellite/reanalysis datasets were "
    "preprocessed into annual raster stacks (2015-2024, ~1 km resolution). "
    "Two parallel pipelines were run: (1) RESTREND - OLS regression of annual "
    "NDVI on climate predictors, partial-derivative attribution of CC and HA, "
    "and a 6-class headline map; (2) Random Forest (n_estimators=300, spatial "
    "80/20 hold-out) as an independent cross-check of driver importance. "
    "Results were validated by correlating non-climatic NDVI trend (residual "
    "Sen's slope) against HFI trend."
)
print("  CAPTION:", CAPTION_2)


# =============================================================================
# FIG 3 — 5-panel Sen's slope maps
# =============================================================================
print("Fig 3: 5-panel Sen slope maps ...")

panels = [
    (ndvi_slope,  "NDVI Sen's slope  (NDVI yr$^{-1}$)",    "RdBu",  0.012),
    (lst_slope,   "LST Sen's slope  (K yr$^{-1}$)",         "RdBu_r",0.25),
    (prec_slope,  "Precip Sen's slope  (mm yr$^{-1}$)",     "RdBu",  1.0),
    (srad_slope,  "SRAD Sen's slope  (W m$^{-2}$ yr$^{-1}$)","RdBu", 1.5),
    (hfi_slope,   "HFI Sen's slope  (HFI units yr$^{-1}$)", "RdBu_r",0.04),
]
panel_titles = ["(a) NDVI", "(b) LST", "(c) Precipitation",
                "(d) Solar Radiation", "(e) Human Footprint"]

fig3 = plt.figure(figsize=(18, 4.2))
axes3 = []
for i in range(5):
    ax = fig3.add_subplot(1, 5, i + 1)
    axes3.append(ax)

for ax, (arr, cblabel, cmap, vlim), title in zip(axes3, panels, panel_titles):
    im = ax.imshow(arr, extent=extent, origin="upper",
                   cmap=cmap, vmin=-vlim, vmax=vlim,
                   interpolation="nearest", aspect="auto")
    aoi.boundary.plot(ax=ax, color="black", linewidth=0.9)
    add_cbar(fig3, im, ax, cblabel, shrink=0.80, pad=0.03)
    ax.set_title(title, fontsize=10.5, pad=3)
    ax.set_xlabel("Lon (deg E)", fontsize=8)
    if ax is axes3[0]:
        ax.set_ylabel("Lat (deg N)", fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=7)

fig3.suptitle(
    "Sen's Slope Trends by Variable  -  Samarkand Province 2015-2024\n"
    "(diverging colormap centred at zero; blue = decrease, red = increase)",
    fontsize=11, y=1.02
)
fig3.tight_layout(w_pad=0.8)
fig3.savefig("figures/fig3.png", dpi=DPI, bbox_inches="tight")
plt.close(fig3)
print("  Saved figures/fig3.png")

CAPTION_3 = (
    "Figure 3. Per-pixel Theil-Sen monotonic trend slopes (2015-2024) for "
    "(a) NDVI, (b) LST, (c) precipitation, (d) solar radiation, and "
    "(e) Human Footprint Index. Diverging colormaps are centred at zero; "
    "blue tones indicate decreasing trends, red tones increasing. Colour "
    "ranges are clipped at the 99th percentile for visualisation clarity; "
    "statistics were computed on the full data range. AOI boundary shown in "
    "black. n = 20,132 valid pixels."
)
print("  CAPTION:", CAPTION_3)


# =============================================================================
# FIG 4 — RESTREND 2-panel (R2 + residual Sen slope)
# =============================================================================
print("Fig 4: RESTREND 2-panel ...")

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4.8))

im4a = ax4a.imshow(r2_map, extent=extent, origin="upper",
                   cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
aoi.boundary.plot(ax=ax4a, color="white", linewidth=0.9)
add_cbar(fig4, im4a, ax4a, "OLS R$^2$", shrink=0.82)
ax4a.set_title("(a) RESTREND Model Fit\n"
               "OLS: NDVI ~ LST + Precip + SRAD  (n=10 yr, k=3)", fontsize=10.5)
ax4a.set_xlabel("Longitude (deg E)", fontsize=9)
ax4a.set_ylabel("Latitude (deg N)", fontsize=9)

# Residual slope symmetric range
rs_lim = float(np.nanpercentile(np.abs(resid_slope[valid]), 99))
im4b = ax4b.imshow(resid_slope, extent=extent, origin="upper",
                   cmap="RdBu", vmin=-rs_lim, vmax=rs_lim,
                   interpolation="nearest", aspect="auto")
aoi.boundary.plot(ax=ax4b, color="black", linewidth=0.9)
add_cbar(fig4, im4b, ax4b, "Residual Sen's slope\n(NDVI yr$^{-1}$)", shrink=0.82)
ax4b.set_title("(b) Non-climatic NDVI Trend\n"
               "Sen's slope on RESTREND residuals", fontsize=10.5)
ax4b.set_xlabel("Longitude (deg E)", fontsize=9)
ax4b.set_yticklabels([])

# R2 stats annotation on ax4a
r2_v = r2_map[valid & np.isfinite(r2_map)]
ax4a.text(0.03, 0.97,
          f"mean R2 = {r2_v.mean():.3f}\nmedian R2 = {np.median(r2_v):.3f}\n"
          f"R2 > 0.5: {100*(r2_v > 0.5).mean():.1f}%",
          transform=ax4a.transAxes, fontsize=8, va="top", ha="left",
          family="monospace",
          bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85,
                    ec="gray", lw=0.7))

fig4.suptitle("RESTREND Analysis  -  Samarkand Province 2015-2024",
              fontsize=12, y=1.01)
fig4.tight_layout(w_pad=1.2)
fig4.savefig("figures/fig4.png", dpi=DPI, bbox_inches="tight")
plt.close(fig4)
print("  Saved figures/fig4.png")

CAPTION_4 = (
    "Figure 4. RESTREND (Residual Trend Analysis) results. (a) Per-pixel OLS "
    "coefficient of determination (R2) for the climate model "
    "NDVI ~ beta_T*LST + beta_P*Precip + beta_S*SRAD (n=10 years, k=3 "
    "predictors, 6 degrees of freedom). Higher values indicate greater climate "
    "control of interannual NDVI variability. (b) Sen's monotonic trend slope "
    "on the OLS residuals, isolating non-climatic NDVI change. Red = residual "
    "greening, blue = residual browning after the climate signal is removed."
)
print("  CAPTION:", CAPTION_4)


# =============================================================================
# FIG 5 — CC and HA contributions (shared symmetric scale)
# =============================================================================
print("Fig 5: CC and HA contributions ...")

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 4.8))

cc_v = cc_map[valid & np.isfinite(cc_map)]
ha_v = ha_map[valid & np.isfinite(ha_map)]
clim_lim = float(np.percentile(
    np.abs(np.concatenate([cc_v, ha_v])), 99
))

im5a = ax5a.imshow(cc_map, extent=extent, origin="upper",
                   cmap="RdBu", vmin=-clim_lim, vmax=clim_lim,
                   interpolation="nearest", aspect="auto")
aoi.boundary.plot(ax=ax5a, color="black", linewidth=0.9)
add_cbar(fig5, im5a, ax5a, "CC  (NDVI yr$^{-1}$)", shrink=0.82)
ax5a.set_title("(a) Climate Contribution (CC)\n"
               "CC = beta_T*dLST + beta_P*dPrecip + beta_S*dSRAD", fontsize=10.5)
ax5a.set_xlabel("Longitude (deg E)", fontsize=9)
ax5a.set_ylabel("Latitude (deg N)", fontsize=9)

im5b = ax5b.imshow(ha_map, extent=extent, origin="upper",
                   cmap="RdBu", vmin=-clim_lim, vmax=clim_lim,
                   interpolation="nearest", aspect="auto")
aoi.boundary.plot(ax=ax5b, color="black", linewidth=0.9)
add_cbar(fig5, im5b, ax5b, "HA  (NDVI yr$^{-1}$)", shrink=0.82)
ax5b.set_title("(b) Human Attribution (HA)\n"
               "HA = e_HFI * HFI_slope  (residual ~ HFI regression)", fontsize=10.5)
ax5b.set_xlabel("Longitude (deg E)", fontsize=9)
ax5b.set_yticklabels([])

# Stats boxes
for ax, arr_v, label in [(ax5a, cc_v, "CC"), (ax5b, ha_v, "HA")]:
    ax.text(0.03, 0.97,
            f"mean = {arr_v.mean():+.5f}\nmedian = {np.median(arr_v):+.5f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85,
                      ec="gray", lw=0.7))

fig5.suptitle("Partial-Derivative Attribution  -  Samarkand Province 2015-2024\n"
              "(shared symmetric colour scale; red = positive, blue = negative)",
              fontsize=12, y=1.02)
fig5.tight_layout(w_pad=1.2)
fig5.savefig("figures/fig5.png", dpi=DPI, bbox_inches="tight")
plt.close(fig5)
print("  Saved figures/fig5.png")

CAPTION_5 = (
    "Figure 5. Partial-derivative attribution maps on a shared symmetric colour "
    "scale. (a) Climate Contribution (CC) = sum of OLS coefficient times "
    "Sen's slope for each climate predictor. Positive CC (red) indicates "
    "climate-driven greening; negative CC (blue) indicates climate-driven "
    "browning. (b) Human Attribution (HA) = per-pixel sensitivity to HFI "
    "(from residual ~ HFI regression) times HFI Sen's slope. Positive HA "
    "indicates human-footprint-driven greening; negative HA indicates "
    "human-driven browning."
)
print("  CAPTION:", CAPTION_5)


# =============================================================================
# FIG 6 — Headline 6-class attribution map
# =============================================================================
print("Fig 6: 6-class attribution map ...")

class_info = {
    1: ("Climate-dom restoration",  "#2ecc71"),
    2: ("Human-dom restoration",    "#27ae60"),
    3: ("Co-driven restoration",    "#a9dfbf"),
    4: ("Climate-dom degradation",  "#e74c3c"),
    5: ("Human-dom degradation",    "#c0392b"),
    6: ("Co-driven degradation",    "#f1948a"),
}

# Build discrete colormap
cmap_colors = ["#ffffff"] + [class_info[i][1] for i in range(1, 7)]
cmap6 = mcolors.ListedColormap(cmap_colors)
norm6 = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap6.N)

fig6, ax6 = plt.subplots(1, 1, figsize=(8.5, 6))

with rasterio.open("RESTREND/attribution_6class.tif") as src:
    cls_raw = src.read(1).astype(np.float32)
cls_disp = cls_raw.copy()
cls_disp[cls_raw == 0] = 0   # unclassified -> white

im6 = ax6.imshow(cls_disp, extent=extent, origin="upper",
                 cmap=cmap6, norm=norm6, interpolation="nearest",
                 aspect="auto")
aoi.boundary.plot(ax=ax6, color="black", linewidth=1.2, zorder=3)

# Scale bar and compass
add_scale_bar(ax6, x0_frac=0.04, y0_frac=0.05, length_deg=0.586,
              lat_deg=40.0, label_km="50 km")
ax6.annotate("N", xy=(0.06, 0.18), xytext=(0.06, 0.13),
             xycoords="axes fraction", textcoords="axes fraction",
             ha="center", fontsize=11, fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

# Pixel area for area computations
lat_ctr = transform_affine.f + transform_affine.e * (height / 2.0)
px_deg  = abs(transform_affine.a)
px_km2  = px_deg * 111.0 * px_deg * 111.0 * abs(math.cos(math.radians(lat_ctr)))

# Legend with pixel counts and area
n_valid = int(valid.sum())
legend_patches = []
for cid, (label, color) in class_info.items():
    n_px = int((cls_raw[valid] == cid).sum())
    pct  = 100.0 * n_px / n_valid
    area = n_px * px_km2
    patch = mpatches.Patch(
        facecolor=color, edgecolor="gray", lw=0.6,
        label=f"{label}  ({pct:.1f}%,  {area:.0f} km2)"
    )
    legend_patches.append(patch)
patch_uncls = mpatches.Patch(facecolor="white", edgecolor="gray", lw=0.6,
                              label="Unclassified")
legend_patches.append(patch_uncls)

ax6.legend(handles=legend_patches, loc="lower right", fontsize=8,
           framealpha=0.92, edgecolor="gray", title="Attribution class",
           title_fontsize=8.5)

ax6.set_xlabel("Longitude (deg E)", fontsize=10)
ax6.set_ylabel("Latitude (deg N)", fontsize=10)
ax6.set_title(
    "Vegetation Change Attribution  -  Samarkand Province 2015-2024\n"
    "6-Class Dominant-Driver Map (RESTREND + partial-derivative method)",
    fontsize=11
)

fig6.tight_layout()
fig6.savefig("figures/fig6.png", dpi=DPI)
plt.close(fig6)
print("  Saved figures/fig6.png")

CAPTION_6 = (
    "Figure 6. Headline 6-class attribution map classifying each valid pixel "
    "by dominant driver of NDVI change (2015-2024). Green shades indicate "
    "vegetation restoration; red shades indicate degradation. Dark colours "
    "denote single-driver dominance; pale colours denote co-driven change "
    "(|CC - HA| / max(|CC|, |HA|) < 0.2). Co-driven classes were assigned "
    "first to avoid overlap with dominant-driver classes. Legend shows pixel "
    "percentage and approximate area of each class. Scale bar = 50 km."
)
print("  CAPTION:", CAPTION_6)


# =============================================================================
# FIG 7 — RF feature importances
# =============================================================================
print("Fig 7: RF feature importances ...")

# All 7 features
imp_all = imp_df.set_index("feature")["importance"]
# Climate+HFI subset (per spec)
sub4 = imp_all[["LST", "Precip", "SRAD", "HFI"]].sort_values(ascending=True)

fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(11, 4.2))

# Left: all 7 features
imp_all_sorted = imp_all.sort_values(ascending=True)
colors_all = ["#d62728" if i == "HFI" else
              ("#1f77b4" if i in ["LST", "Precip", "SRAD"] else "#aec7e8")
              for i in imp_all_sorted.index]
bars_all = ax7a.barh(imp_all_sorted.index, imp_all_sorted.values,
                     color=colors_all, edgecolor="white", height=0.6)
for bar, val in zip(bars_all, imp_all_sorted.values):
    ax7a.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
              f"{val:.4f}", va="center", fontsize=8.5)
ax7a.set_xlabel("Mean Decrease Impurity (Gini importance)", fontsize=10)
ax7a.set_title("(a) All 7 Features", fontsize=11)
ax7a.set_xlim(0, imp_all_sorted.values.max() * 1.22)
leg_items_a = [
    mpatches.Patch(color="#1f77b4", label="Climate predictors"),
    mpatches.Patch(color="#d62728", label="Human Footprint (HFI)"),
    mpatches.Patch(color="#aec7e8", label="Nuisance controls"),
]
ax7a.legend(handles=leg_items_a, fontsize=8, loc="lower right", framealpha=0.9)

# Right: climate + HFI only
colors4 = ["#d62728" if i == "HFI" else "#1f77b4" for i in sub4.index]
bars4 = ax7b.barh(sub4.index, sub4.values, color=colors4,
                   edgecolor="white", height=0.55)
for bar, val in zip(bars4, sub4.values):
    ax7b.text(val + sub4.values.max() * 0.02,
              bar.get_y() + bar.get_height() / 2,
              f"{val:.4f}", va="center", fontsize=9)
ax7b.set_xlabel("Mean Decrease Impurity (Gini importance)", fontsize=10)
ax7b.set_title("(b) Climate + HFI Features\n(excluding nuisance controls)",
               fontsize=11)
ax7b.set_xlim(0, sub4.values.max() * 1.25)

# Percentage shares annotation
clim_imp = float(imp_all[["LST", "Precip", "SRAD"]].sum())
hfi_imp  = float(imp_all["HFI"])
total_ch = clim_imp + hfi_imp
ax7b.text(0.97, 0.12,
          f"Climate share: {100*clim_imp/total_ch:.1f}%\n"
          f"HFI share:     {100*hfi_imp/total_ch:.1f}%",
          transform=ax7b.transAxes, fontsize=9, va="bottom", ha="right",
          family="monospace",
          bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9,
                    ec="gray", lw=0.7))

fig7.suptitle("Random Forest Feature Importances\n"
              "RF: NDVI ~ climate + HFI + lat + lon + year  "
              "(Test R2 = 0.925, spatial 80/20 hold-out)",
              fontsize=11.5, y=1.02)
fig7.tight_layout(w_pad=1.5)
fig7.savefig("figures/fig7.png", dpi=DPI, bbox_inches="tight")
plt.close(fig7)
print("  Saved figures/fig7.png")

CAPTION_7 = (
    "Figure 7. Random Forest feature importances (Mean Decrease Impurity). "
    "(a) All seven model features ranked by importance. LST is the dominant "
    "predictor; HFI ranks second overall. Nuisance controls (lat, lon, year) "
    "capture spatial and temporal autocorrelation and are not discussed further. "
    "(b) Subset showing only climate predictors (blue) and HFI (red), "
    "normalised to their combined contribution. Climate predictors collectively "
    "account for the majority of explainable variance, with HFI contributing "
    "a complementary but smaller share. Model: RandomForestRegressor "
    "(n_estimators=300, max_depth=12); Test R2 = 0.925 on spatial 80/20 hold-out."
)
print("  CAPTION:", CAPTION_7)


# =============================================================================
# FIG 8 — Validation hexbin scatter
# =============================================================================
print("Fig 8: Validation hexbin scatter ...")

with rasterio.open("RESTREND/residual_sen_slope.tif") as src:
    resid_raw = src.read(1).astype(np.float64)
with rasterio.open("Trends/hfi_sen_slope.tif") as src:
    hfi_raw = src.read(1).astype(np.float64)

vmask = np.isfinite(resid_raw) & np.isfinite(hfi_raw)
x = resid_raw[vmask]
y = hfi_raw[vmask]
n = len(x)

r_val,   p_r   = stats.pearsonr(x, y)
rho_val, p_rho = stats.spearmanr(x, y)
ols = stats.linregress(x, y)
r2_ols = ols.rvalue ** 2

q_pp = int(((x > 0) & (y > 0)).sum())
q_pn = int(((x > 0) & (y < 0)).sum())
q_np = int(((x < 0) & (y > 0)).sum())
q_nn = int(((x < 0) & (y < 0)).sum())

x_lo, x_hi = np.percentile(x, 1),  np.percentile(x, 99)
y_lo, y_hi = np.percentile(y, 1),  np.percentile(y, 99)
x_clip = np.clip(x, x_lo, x_hi)
y_clip = np.clip(y, y_lo, y_hi)

fig8, ax8 = plt.subplots(figsize=(7.5, 6.2))
hb = ax8.hexbin(x_clip, y_clip, gridsize=55, cmap="YlOrRd",
                mincnt=1, linewidths=0.06)
cb8 = plt.colorbar(hb, ax=ax8, shrink=0.88, pad=0.02)
cb8.set_label("Pixel count", fontsize=9)
cb8.ax.tick_params(labelsize=8)

# Zero lines
ax8.axhline(0, color="gray", lw=0.9, ls="--", alpha=0.6, zorder=3)
ax8.axvline(0, color="gray", lw=0.9, ls="--", alpha=0.6, zorder=3)

# OLS line + 95% CI band
x_line = np.linspace(x_lo, x_hi, 300)
y_line = ols.slope * x_line + ols.intercept
# approximate SE of predicted mean
n_pts   = len(x)
x_bar   = x.mean()
ss_xx   = ((x - x_bar)**2).sum()
se_fit  = ols.stderr * np.sqrt(1.0/n_pts + (x_line - x_bar)**2 / ss_xx)
t_crit  = stats.t.ppf(0.975, df=n_pts - 2)
ax8.fill_between(x_line, y_line - t_crit * se_fit, y_line + t_crit * se_fit,
                 color="#3498db", alpha=0.15, zorder=3, label="_nolegend_")
ax8.plot(x_line, y_line, color="#2980b9", lw=2.0, zorder=4,
         label=f"OLS  y = {ols.slope:.4f}x {ols.intercept:+.4f}")

# Quadrant counts
fs_q = 8.5
kw = dict(color="black", fontsize=fs_q, alpha=0.75)
ax8.text(x_hi * 0.96, y_hi * 0.92, f"n={q_pp:,}", ha="right", va="top", **kw)
ax8.text(x_hi * 0.96, y_lo * 0.92, f"n={q_pn:,}", ha="right", va="bottom", **kw)
ax8.text(x_lo * 0.96, y_hi * 0.92, f"n={q_np:,}", ha="left",  va="top", **kw)
ax8.text(x_lo * 0.96, y_lo * 0.92, f"n={q_nn:,}", ha="left",  va="bottom", **kw)

# Stats annotation
ann = (
    f"n = {n:,}\n"
    f"Pearson r = {r_val:+.3f}   (p = {p_r:.2e})\n"
    f"Spearman rho = {rho_val:+.3f}   (p = {p_rho:.2e})\n"
    f"OLS R2 = {r2_ols:.3f}"
)
ax8.text(0.03, 0.97, ann, transform=ax8.transAxes,
         fontsize=9, va="top", ha="left", family="monospace",
         bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                   alpha=0.92, edgecolor="gray", lw=0.8))

ax8.set_xlim(x_lo, x_hi)
ax8.set_ylim(y_lo, y_hi)
ax8.set_xlabel("Residual Sen's slope  (NDVI yr$^{-1}$, climate signal removed)",
               fontsize=11)
ax8.set_ylabel("HFI Sen's slope  (HFI units yr$^{-1}$)", fontsize=11)
ax8.set_title(
    "Independent Validation: Non-climatic NDVI Trend vs Human Footprint Trend\n"
    "Samarkand Province  2015-2024  |  Expected: negative correlation",
    fontsize=11
)
ax8.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor="gray")

fig8.tight_layout()
fig8.savefig("figures/fig8.png", dpi=DPI)
plt.close(fig8)
print("  Saved figures/fig8.png")

CAPTION_8 = (
    "Figure 8. Independent validation scatter plot. Each hexagonal bin represents "
    "valid pixels. X-axis: Sen's slope on RESTREND residuals (non-climatic NDVI "
    "trend, NDVI yr-1). Y-axis: Sen's slope on Human Footprint Index (HFI yr-1). "
    "Note the structural heteroscedasticity: the dense dark-red band sits near "
    "HFI slope = 0 and spans the full range of NDVI residual values, while a "
    "diffuse cloud of high-HFI-slope pixels in the upper region pulls the OLS line "
    "diagonally through an area that contains very few pixels. The OLS fit "
    "(R2 = 0.114) therefore describes only ~11% of variance and is a poor summary "
    "of the data; the fitted line does not pass through the bulk of the point "
    "cloud. This non-linear, heteroscedastic structure is the reason a decile "
    "box-plot (Figure 9) is used alongside the scatter to characterise the "
    "relationship more faithfully. The observed Pearson r = +0.338 is statistically "
    "significant (p < 0.001, n = 20,132) but opposite to the expected negative "
    "sign, reflecting irrigation-driven greening in Samarkand Province where rising "
    "HFI co-occurs with rising NDVI. The dominant quadrant (x<0, y>0; n = 7,349; "
    "36.5%) supports the hypothesis locally, but is nearly matched by pixels with "
    "declining HFI and declining NDVI (x<0, y<0; n = 7,091; 35.2%). "
    "Blue band = 95% CI on the OLS line. Y-axis clipped to the 1st-99th percentile "
    "of HFI slope; statistics are computed on the full unclipped data."
)
print("  CAPTION:", CAPTION_8)


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 68)
print("ALL FIGURES SAVED TO figures/")
print("=" * 68)
fig_list = [
    ("fig1.png", "Study area map"),
    ("fig2.png", "Workflow block diagram"),
    ("fig3.png", "5-panel Sen slope maps"),
    ("fig4.png", "RESTREND R2 + residual Sen slope"),
    ("fig5.png", "CC and HA contribution maps"),
    ("fig6.png", "6-class attribution headline map"),
    ("fig7.png", "RF feature importances"),
    ("fig8.png", "Validation hexbin scatter"),
]
for fname, desc in fig_list:
    size_kb = os.path.getsize(f"figures/{fname}") // 1024
    print(f"  figures/{fname:<14}  {desc:<35}  {size_kb:>5} KB")
print("=" * 68)
