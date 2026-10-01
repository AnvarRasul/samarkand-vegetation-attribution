"""Revised Figures 3, 4, 5 and new Figure 9 (HFI discontinuity) with
explicit, verified colour conventions."""
import os, sys
import numpy as np, rasterio, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
os.chdir(os.environ.get("DATA_ROOT", "."))  # folder holding the data and outputs (see README)
plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "savefig.bbox": "tight"})
d = np.load("analysis_stacks.npz"); valid = d["valid_mask"]; t = d["transform"]
H, W = valid.shape
extent = [t[2], t[2] + W * t[0], t[5] + H * t[4], t[5]]
aoi = gpd.read_file("AOI/samarkand_region.shp")


def load(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype(np.float32)
        if s.nodata is not None and not np.isnan(s.nodata):
            a[a == s.nodata] = np.nan
    a[~valid] = np.nan
    return a


def lim(a, q=98):
    return float(np.nanpercentile(np.abs(a[valid]), q))


def panel(fig, ax, arr, cmap, vl, cblabel, title, lo_txt, hi_txt, ylabel=True):
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=-vl, vmax=vl,
                   interpolation="nearest", aspect="auto")
    aoi.boundary.plot(ax=ax, color="black", linewidth=0.7)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03, extend="both")
    cb.set_label(cblabel + "\n(+) " + hi_txt + "   |   (−) " + lo_txt, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlabel("Longitude (\u00b0E)", fontsize=8.5)
    if ylabel:
        ax.set_ylabel("Latitude (\u00b0N)", fontsize=8.5)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=7.5)


# ---------------- Figure 3 ----------------
P = [("Trends/ndvi_sen_slope.tif", "BrBG", "NDVI yr$^{-1}$", "(a) NDVI", "browning", "greening"),
     ("Trends/lst_sen_slope.tif", "RdBu_r", "\u00b0C yr$^{-1}$", "(b) LST", "cooling", "warming"),
     ("Trends/precip_sen_slope.tif", "BrBG", "mm month$^{-1}$ yr$^{-1}$", "(c) Precipitation", "drier", "wetter"),
     ("Trends/srad_sen_slope.tif", "RdBu_r", "W m$^{-2}$ yr$^{-1}$", "(d) Solar radiation", "decrease", "increase"),
     ("Trends/hfi_sen_slope.tif", "PuOr_r", "HFI units yr$^{-1}$", "(e) Human Footprint", "decrease", "increase")]
fig = plt.figure(figsize=(16, 9.6))
for i, (p, cm, u, ti, lo, hi) in enumerate(P):
    ax = fig.add_subplot(2, 3, i + 1)
    a = load(p)
    panel(fig, ax, a, cm, lim(a), u, ti, lo, hi, ylabel=(i % 3 == 0))
fig.suptitle("Theil\u2013Sen trend slopes of growing-season annual values, Samarkand Province 2015\u20132024\n"
             "Diverging scales centred at zero; each colour-bar label states what positive (+) and negative (−) values mean",
             fontsize=12, y=1.02)
fig.tight_layout(w_pad=1.0, h_pad=2.0)
fig.savefig(f"{OUT}/fig3.png")
plt.close(fig)

# ---------------- Figure 4 ----------------
r2 = load("RESTREND/R2.tif"); rs = load("RESTREND/residual_sen_slope.tif")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5))
im = a1.imshow(r2, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1,
               interpolation="nearest", aspect="auto")
aoi.boundary.plot(ax=a1, color="black", linewidth=0.7)
cb = fig.colorbar(im, ax=a1, shrink=0.82, pad=0.03)
cb.set_label("R$^2$ of climate-only OLS", fontsize=8.5)
a1.set_title("(a) RESTREND model fit\nNDVI ~ LST + P + SRAD (n = 10 years, k = 3)", fontsize=10)
a1.set_xlabel("Longitude (\u00b0E)", fontsize=8.5); a1.set_ylabel("Latitude (\u00b0N)", fontsize=8.5)
v = r2[valid]
a1.text(0.03, 0.97, f"mean R\u00b2 = {v.mean():.3f}\nmedian R\u00b2 = {np.median(v):.3f}\nR\u00b2 > 0.5: {100*(v>0.5).mean():.1f}%",
        transform=a1.transAxes, fontsize=8, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.9, ec="gray", lw=0.7))
panel(fig, a2, rs, "BrBG", lim(rs), "Residual Sen slope (NDVI yr$^{-1}$)",
      "(b) Non-climatic NDVI trend\nTheil\u2013Sen slope of RESTREND residuals",
      "residual browning", "residual greening", ylabel=False)
fig.suptitle("RESTREND analysis \u2013 Samarkand Province 2015\u20132024", fontsize=11.5, y=1.02)
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{OUT}/fig4.png")
plt.close(fig)

# ---------------- Figure 5 ----------------
cc = load("RESTREND/CC.tif"); ha = load("RESTREND/HA.tif")
vl = float(np.nanpercentile(np.abs(np.concatenate([cc[valid], ha[valid]])), 98))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5))
panel(fig, a1, cc, "BrBG", vl, "CC (NDVI yr$^{-1}$)",
      "(a) Climate Contribution (CC)\nCC = \u03b2$_T$\u00b7dLST/dt + \u03b2$_P$\u00b7dP/dt + \u03b2$_S$\u00b7dSRAD/dt",
      "climate-driven browning", "climate-driven greening")
panel(fig, a2, ha, "BrBG", vl, "HA (NDVI yr$^{-1}$)",
      "(b) Human Attribution (HA)\nHA = e\u00b7dHFI/dt (residual ~ HFI regression)",
      "human-attributed browning", "human-attributed greening", ylabel=False)
for ax, arr in [(a1, cc), (a2, ha)]:
    v = arr[valid]
    ax.text(0.03, 0.97, f"mean = {v.mean():+.5f}\nmedian = {np.median(v):+.5f}", transform=ax.transAxes,
            fontsize=8, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.9, ec="gray", lw=0.7))
fig.suptitle("Partial-derivative attribution \u2013 Samarkand Province 2015\u20132024\n"
             "Shared symmetric scale: green = positive (greening), brown = negative (browning)",
             fontsize=11.5, y=1.05)
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{OUT}/fig5.png")
plt.close(fig)

# ---------------- Figure 9: HFI discontinuity ----------------
hfp = d["hfp_annual"].astype(np.float64)
yrs = np.arange(2015, 2025)
lon = t[2] + (np.arange(W) + 0.5) * t[0]
LON = np.tile(lon, (H, 1))
zones = [(65.1, 66.1, "West (65.1\u201366.1\u00b0E)", "#8c510a"),
         (66.1, 66.6, "Centre (66.1\u201366.6\u00b0E)", "#7f7f7f"),
         (66.6, 67.6, "East (66.6\u201367.6\u00b0E)", "#01665e")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
for lo, hi, lab, col in zones:
    m = valid & (LON >= lo) & (LON < hi)
    a1.plot(yrs, [hfp[i][m].mean() for i in range(10)], "-o", color=col, label=lab, lw=2, ms=4.5)
a1.plot(yrs, [hfp[i][valid].mean() for i in range(10)], "--", color="black", lw=1.5, label="Province (valid pixels)")
a1.set_ylim(6, 27)
a1.axvspan(2018.5, 2024.5, color="#fdd49e", alpha=0.35, lw=0)
a1.axvline(2018.5, color="#d7301f", lw=1.2, ls=":")
a1.text(2018.65, 26.5, "2019\u20132024 layers: later figshare\nversions, not described in\nMu et al. (2022)",
        fontsize=7.5, va="top", color="#b30000")
a1.text(2015.1, 26.5, "Period covered by\nMu et al. (2022)", fontsize=7.5, va="top")
a1.set_xticks(yrs); a1.tick_params(labelsize=8)
a1.set_xlabel("Year"); a1.set_ylabel("Mean HFI (index, 0\u201350)")
a1.set_title("(a) Mean annual HFI by longitude zone", fontsize=10)
a1.legend(fontsize=7.5, loc="lower left")
J = hfp[4] - hfp[3]
J[~valid] = np.nan
jl = float(np.nanpercentile(np.abs(J[valid]), 98))
im = a2.imshow(J, extent=extent, origin="upper", cmap="PuOr_r", vmin=-jl, vmax=jl,
               interpolation="nearest", aspect="auto")
aoi.boundary.plot(ax=a2, color="black", linewidth=0.7)
cb = fig.colorbar(im, ax=a2, shrink=0.85, pad=0.03, extend="both")
cb.set_label("HFI$_{2019}$ \u2212 HFI$_{2018}$ (index units)", fontsize=8.5)
a2.set_title("(b) Single-year HFI change, 2018 \u2192 2019", fontsize=10)
a2.set_xlabel("Longitude (\u00b0E)", fontsize=8.5); a2.set_ylabel("Latitude (\u00b0N)", fontsize=8.5)
a2.tick_params(labelsize=7.5)
fig.suptitle("Temporal consistency of the Human Footprint series used in the analysis", fontsize=11.5, y=1.02)
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{OUT}/fig9.png")
plt.close(fig)
print("done")
