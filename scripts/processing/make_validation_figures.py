"""
Two improved validation figures.

fig_validation_v2.png    -- hexbin on white background + marginal histograms
fig_validation_decile.png -- HFI decile boxplot (new diagram type)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import pandas as pd
import rasterio
from scipy import stats

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.35,
    "grid.linewidth":  0.6,
})


# ── data ──────────────────────────────────────────────────────────────────────

def load_data():
    with rasterio.open("RESTREND/residual_sen_slope.tif") as s:
        resid = s.read(1).astype(np.float64)
    with rasterio.open("Trends/hfi_sen_slope.tif") as s:
        hfi = s.read(1).astype(np.float64)
    valid = np.isfinite(resid) & np.isfinite(hfi)
    x = resid[valid]
    y = hfi[valid]
    return x, y


def compute_stats(x, y):
    r,   p_r   = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    ols = stats.linregress(x, y)
    return r, p_r, rho, p_rho, ols


# ── Figure 1: improved hexbin + marginal histograms ───────────────────────────

def fig_hexbin(x, y, r, p_r, rho, p_rho, ols):
    """
    Joint-distribution layout:
      top-left   = x marginal histogram
      bottom-left = main hexbin
      bottom-right = y marginal histogram (rotated)
    """
    n = len(x)

    # Clip range for display (stats use full data)
    xlo, xhi = np.percentile(x, 1),  np.percentile(x, 99)
    ylo, yhi = np.percentile(y, 1),  np.percentile(y, 99)

    fig = plt.figure(figsize=(8.8, 7.2))
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[5, 1], height_ratios=[1, 5],
        hspace=0.06, wspace=0.06,
        left=0.10, right=0.87, top=0.88, bottom=0.10,
    )
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    # colorbar axis
    ax_cb = fig.add_axes([0.89, 0.10, 0.025, 0.55])

    # ---- main hexbin ----
    x_c = np.clip(x, xlo, xhi)
    y_c = np.clip(y, ylo, yhi)

    hb = ax_main.hexbin(
        x_c, y_c,
        gridsize=48,
        cmap="YlOrRd",
        norm=mcolors.LogNorm(vmin=1),
        mincnt=1,
        linewidths=0.05,
        edgecolors="face",
        zorder=3,
    )
    cb = fig.colorbar(hb, cax=ax_cb)
    cb.set_label("Pixel count  (log scale)", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)

    # subtle quadrant fills (behind hexbin)
    ax_main.fill_betweenx([ylo, 0], xlo, 0,  color="#f0e0e0", zorder=0)  # x<0 y<0
    ax_main.fill_betweenx([0, yhi], xlo, 0,  color="#ffd5d5", zorder=0)  # x<0 y>0 (expected)
    ax_main.fill_betweenx([ylo, 0], 0,  xhi, color="#e0f0e0", zorder=0)  # x>0 y<0
    ax_main.fill_betweenx([0, yhi], 0,  xhi, color="#d5e8ff", zorder=0)  # x>0 y>0 (unexpected)

    # zero lines
    ax_main.axhline(0, color="#555555", lw=0.9, ls="--", zorder=2)
    ax_main.axvline(0, color="#555555", lw=0.9, ls="--", zorder=2)

    # OLS line + 95 % pointwise CI
    x_line = np.linspace(xlo, xhi, 400)
    y_line = ols.slope * x_line + ols.intercept
    # SE of y-hat at each x_line: se_yhat = s * sqrt(1/n + (x-xbar)^2 / Sxx)
    xbar = x.mean()
    Sxx  = np.sum((x - xbar) ** 2)
    s    = np.sqrt(np.sum((y - (ols.slope * x + ols.intercept)) ** 2) / (n - 2))
    se_yhat = s * np.sqrt(1 / n + (x_line - xbar) ** 2 / Sxx)
    t95  = stats.t.ppf(0.975, df=n - 2)
    ci_lo = y_line - t95 * se_yhat
    ci_hi = y_line + t95 * se_yhat

    ax_main.fill_between(x_line, ci_lo, ci_hi,
                         color="#e74c3c", alpha=0.18, zorder=4, label="_nolegend_")
    ax_main.plot(x_line, y_line,
                 color="#c0392b", lw=2.2, zorder=5,
                 label=f"OLS  slope = {ols.slope:+.1f}  (95% CI band shown)")

    # quadrant pixel-count labels
    q_np = int(((x < 0) & (y > 0)).sum())
    q_pp = int(((x > 0) & (y > 0)).sum())
    q_nn = int(((x < 0) & (y < 0)).sum())
    q_pn = int(((x > 0) & (y < 0)).sum())

    kw = dict(fontsize=8.2, transform=ax_main.transData, zorder=6)
    ax_main.text(xlo * 0.92, yhi * 0.88,
                 f"Decline + rising HFI\n{q_np:,} px  ({100*q_np/n:.0f}%)",
                 color="#9b2226", ha="left", va="top", **kw)
    ax_main.text(xhi * 0.92, yhi * 0.88,
                 f"Greening + rising HFI\n{q_pp:,} px  ({100*q_pp/n:.0f}%)",
                 color="#1d5fa8", ha="right", va="top", **kw)
    ax_main.text(xlo * 0.92, ylo * 0.88,
                 f"Decline + falling HFI\n{q_nn:,} px  ({100*q_nn/n:.0f}%)",
                 color="#666666", ha="left", va="bottom", **kw)
    ax_main.text(xhi * 0.92, ylo * 0.88,
                 f"Greening + falling HFI\n{q_pn:,} px  ({100*q_pn/n:.0f}%)",
                 color="#2d6a4f", ha="right", va="bottom", **kw)

    ax_main.set_xlim(xlo, xhi)
    ax_main.set_ylim(ylo, yhi)
    ax_main.set_xlabel(
        "Residual Sen's slope   (NDVI yr⁻¹,  climate signal removed)",
        fontsize=11,
    )
    ax_main.set_ylabel(
        "HFI Sen's slope   (HFI units yr⁻¹)",
        fontsize=11,
    )
    ax_main.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax_main.legend(
        loc="lower right",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        fontsize=9, 
        framealpha=0.92)

    # stats annotation
    r2 = ols.rvalue ** 2
    ann = (
        f"n = {n:,}\n"
        f"Pearson r = {r:+.3f}   p = {p_r:.1e}\n"
        f"Spearman ρ = {rho:+.3f}   p = {p_rho:.1e}\n"
        f"OLS R² = {r2:.3f}"
    )
    ax_main.text(
        0.99, 0.01, ann,
        transform=ax_main.transAxes,
        fontsize=9, 
        va="bottom", 
        ha="right",
        bbox=dict(boxstyle="round,pad=0.45", fc="white",
                  ec="#aaaaaa", alpha=0.95, lw=0.8),
    )

    # ---- top marginal: x distribution ----
    ax_top.hist(np.clip(x, xlo, xhi), bins=70, color="#e08030", alpha=0.80,
                density=True, linewidth=0)
    ax_top.axvline(0, color="#555555", lw=0.9, ls="--")
    ax_top.set_ylabel("Density", fontsize=8)
    ax_top.tick_params(labelbottom=False, labelsize=8)
    ax_top.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    for sp in ["top", "right"]:
        ax_top.spines[sp].set_visible(False)

    # ---- right marginal: y distribution (horizontal) ----
    ax_right.hist(np.clip(y, ylo, yhi), bins=70, color="#e08030", alpha=0.80,
                  density=True, orientation="horizontal", linewidth=0)
    ax_right.axhline(0, color="#555555", lw=0.9, ls="--")
    ax_right.set_xlabel("Density", fontsize=8)
    ax_right.tick_params(labelleft=False, labelsize=8)
    for sp in ["top", "right"]:
        ax_right.spines[sp].set_visible(False)

    fig.suptitle(
        "Non-climatic NDVI trend vs Human Footprint trend\n"
        "Samarkand Province  2015–2024   "
        "[positive r = HFI growth associated with less vegetation loss]",
        fontsize=11.5, fontweight="bold", y=0.97,
    )

    fig.savefig("fig_validation_v2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_validation_v2.png")


# ── Figure 2: HFI-slope decile boxplot ───────────────────────────────────────

def fig_decile(x, y, r, ols):
    """
    Divide all pixels into 10 equal-count bins by HFI slope (deciles).
    For each bin show the full residual-slope distribution as a boxplot.
    The positive trend in bin medians makes the direction immediately obvious.
    """
    n = len(x)
    df = pd.DataFrame({"resid": x, "hfi": y})
    df["decile"] = pd.qcut(df["hfi"], q=10, labels=False)  # 0..9

    bin_resid   = [df.loc[df["decile"] == i, "resid"].values for i in range(10)]
    bin_hfi_med = [df.loc[df["decile"] == i, "hfi"].median() for i in range(10)]
    bin_hfi_lo  = [df.loc[df["decile"] == i, "hfi"].min()    for i in range(10)]
    bin_hfi_hi  = [df.loc[df["decile"] == i, "hfi"].max()    for i in range(10)]
    bin_medians = [np.median(g) for g in bin_resid]
    bin_q25     = [np.percentile(g, 25) for g in bin_resid]
    bin_q75     = [np.percentile(g, 75) for g in bin_resid]
    bin_n       = [len(g) for g in bin_resid]

    # colour ramp: blue (low HFI growth) -> red (high HFI growth)
    cmap = plt.cm.get_cmap("coolwarm_r")
    box_colors = [cmap(i / 9) for i in range(10)]

    fig, ax = plt.subplots(figsize=(11, 5.8))

    # ---- boxplots ----
    bp = ax.boxplot(
        bin_resid,
        positions=range(1, 11),
        widths=0.62,
        patch_artist=True,
        notch=False,
        sym="",                          # hide outlier fliers (too many)
        medianprops=dict(color="black", lw=2.0),
        whiskerprops=dict(lw=1.2, color="#444444"),
        capprops=dict(lw=1.2, color="#444444"),
        boxprops=dict(lw=1.0),
    )
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.78)

    # ---- median trend line ----
    pos = list(range(1, 11))
    ax.plot(pos, bin_medians, "o-",
            color="black", lw=2.0, ms=6, zorder=6,
            label="Bin median residual slope")
    ax.fill_between(pos, bin_q25, bin_q75,
                    color="black", alpha=0.08, zorder=5,
                    label="IQR band (Q25-Q75)")

    # ---- zero reference ----
    ax.axhline(0, color="#888888", lw=1.2, ls="--", zorder=4,
               label="Zero (no trend)")

    # ---- annotate median value above each box ----
    for i, (p, m) in enumerate(zip(pos, bin_medians)):
        ax.text(p, m + 0.00015,
                f"{m:+.4f}", ha="center", va="bottom",
                fontsize=6.8, color="#111111")

    # ---- x-axis: two-row labels (decile label + HFI median) ----
    labels = [
        f"D{i+1}\n(HFI {bin_hfi_med[i]:+.2f})"
        for i in range(10)
    ]
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(labels, fontsize=9)

    # coloured x-tick labels to match boxes
    for tick, col in zip(ax.get_xticklabels(), box_colors):
        tick.set_color(col)
        tick.set_fontweight("bold")

    # ---- HFI range bar below x-axis (visual guide to bin extent) ----
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([])
    ax2.set_xlabel(
        "HFI slope decile  D1 = slowest/declining HFI growth   "
        "D10 = fastest HFI growth",
        fontsize=10, labelpad=8,
    )
    ax2.spines["top"].set_visible(True)

    # ---- sample-size annotation in each box (small, inside) ----
    ylo_ax = ax.get_ylim()[0]
    for i, (p, nn) in enumerate(zip(pos, bin_n)):
        ax.text(p, ylo_ax * 0.95,
                f"n={nn:,}", ha="center", va="bottom",
                fontsize=6.5, color="#444444", style="italic")

    # ---- overall stats annotation ----
    r2 = ols.rvalue ** 2
    ax.text(
        0.015, 0.97,
        f"Pearson r = {r:+.3f}   OLS slope = {ols.slope:+.1f}   OLS R² = {r2:.3f}\n"
        f"Positive trend: HFI growth → less vegetation loss (or greening)",
        transform=ax.transAxes, fontsize=9.5,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec="#aaaaaa", alpha=0.93, lw=0.8),
    )

    ax.set_ylabel(
        "Residual Sen's slope   (NDVI yr⁻¹,  climate removed)",
        fontsize=11,
    )
    ax.set_xlabel("")   # handled by tick labels
    
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        fontsize=9.5, 
        framealpha=0.93)
    
    ax.set_title(
        "Residual NDVI slope by HFI trend decile\n"
        "Each bin contains ~2,013 pixels  —  colour: blue = declining HFI, "
        "red = fastest HFI growth",
        fontsize=11.5, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("fig_validation_decile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_validation_decile.png")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    x, y = load_data()
    r, p_r, rho, p_rho, ols = compute_stats(x, y)

    print(f"n={len(x):,}  Pearson r={r:+.4f}  Spearman rho={rho:+.4f}")
    print(f"OLS slope={ols.slope:+.4f}  intercept={ols.intercept:+.4f}")

    fig_hexbin(x, y, r, p_r, rho, p_rho, ols)
    fig_decile(x, y, r, ols)
    print("Done.")


if __name__ == "__main__":
    main()
