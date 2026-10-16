#!/usr/bin/env python
"""Publication figure: ICESat-2 photon Kd validated against in-situ optics.

Two panels, both with IS-2 photon Kd on the y-axis (in-situ reference on x):
  (a) in-situ Kd  (profiler Kd_PAR + BGC-Argo Kd490)      -- results/paired_is2_insitu_kd.csv
                                                             results/paired_is2_argo_kd.csv
  (b) in-situ near-surface turbidity                      -- results/paired_is2_turbidity.csv

Built with the publication-figures kit (figkit): >=10pt type, semantic palette,
no suptitle, PNG + vector PDF at 300 dpi.

Panel (a) -- points are grouped by PROFILER TYPE, not by the buoy/ship manifest
tag, because the manifest tag hides the physics: the 64 "buoy" rows contain both
surface-piercing profilers (biased one way) and deep/offshore profilers (biased
the other), so a buoy-vs-ship legend mislabels them.
    surface-piercing  issp | shsp | sp001 | paradj   -> in-situ > IS-2 (spectral)
    deep / offshore   parada | paradk | osp | wfp    -> in-situ < IS-2 (depth)
    glider / ship near-surface casts                 -> in-situ > IS-2
    BGC-Argo floats (clear open ocean)               -> near 1:1
Color family encodes the direction of the bias: warm = in-situ above IS-2
(points below the 1:1 line), blue = in-situ below IS-2 (above the line),
green = unbiased. Only the 1:1 line is drawn; the class
median ratios are given in the legend.

Panel (b) -- 478 turbidity matchups. Per-pass scatter is wide, so a
quantile-binned median (with IQR whiskers) carries the trend.

Deliberately NOT in the figure (all returned by main() for the caption / text):
  - within-class Spearman rho in (a). The panel's question is magnitude agreement
    and whether the offset tracks measurement configuration; a near-zero
    within-class rho invites "there is no correlation" on a panel whose argument
    never rested on one.
  - the pass-level p-value in (b). 478 passes come from 47 stations, so it
    overstates independence (pseudoreplication); the station-level rho is the
    defensible statistic.
  - the 5 km sensitivity test. It is a spatial-mismatch question, not a headline
    result, and belongs in the text where it can be discussed.

The two panels are NOT interchangeable evidence: (a) is a direct comparison of Kd
against Kd, (b) is independent optical consistency (turbidity is not a direct
validation variable for Kd). Both put IS-2 Kd on the y-axis over a SHARED range
(see KDLIM), so vertical position is comparable between panels. (Axes swapped
2026-10-16: in-situ reference on x, IS-2 retrieval on y, per validation convention.)

Usage:
  python fig_is2_validation.py
"""
import os
import re
import sys
import argparse
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # vendored figkit.py
import figkit as fk

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

KDCOL = "is2_kd_coherent_med_8km"       # IS-2 Kd, coherent model, 8 km median
# 8 km = largest radius fully inside the 8 km half-width photon fetch box
# (03d_is2kd_sliderule.py --fetch-km). Columns added by reaggregate_radius.py.
KDCOL5 = "is2_kd_coherent_med_5km"      # 5 km sensitivity variant (printed only)

KD_LABEL = "ICESat-2 photon $K_{dph}$ (m$^{-1}$)"

# Shared IS-2 Kd limits (y-axis of both panels, and x too in square panel (a)):
# both panels plot the SAME IS-2 variable, so they use the same range and vertical
# position IS comparable between them. Default keeps all
# 87 (a) points and drops 12/478 (2.5%) of (b) at the extreme-clear end -- all
# retrieved Kd < 0.003 m^-1, i.e. noise-level. Widen to (2e-4, 3.0) to clip nothing
# (costs (a) about a third of its vertical resolution). Statistics always use the
# full data regardless of this window.
KDLIM = (0.001, 2.0)

# Optional low-Kd cutoff, OFF by default. NOTE: do not reach for a pure-water Kd
# limit here. The pipeline reports kd as the raw decay rate of log(photon_counts)
# vs depth (Kd_analysis.fit_beers_law_bma: kd = -slope, no two-way factor applied),
# so a threshold derived for one-way diffuse Kd does not transfer to this quantity
# without first fixing the convention. Kept as a general-purpose knob for whatever
# cutoff the analysis ends up justifying; when set, the figure is annotated so a
# filtered panel can never be mistaken for an unfiltered one.

# Profiler-type classification from the OOI/IOOS station id.
SURFACE_RE = re.compile(r"issp|shsp|sp001|paradj", re.I)
DEEP_RE = re.compile(r"parada|paradk|osp|wfp", re.I)

# Panel (a) series: key -> (label, color, marker). Order sets legend + z-order.
# Muted Tol palette (shared with the appendix figures): warm = in-situ above
# IS-2, blue = in-situ below IS-2, green = unbiased.
SERIES = {
    # key: (legend stem, colour, marker, fixed?)  Fixed = sampled at a fixed
    # location (filled markers): moorings, and ship casts taken on station.
    # Moving platforms (open markers): gliders and drifting Argo floats.
    "surface": ("Moored surface-piercing profilers", "#CC6677", "o", True),
    "deep": ("Moored offshore profilers", "#4477AA", "s", True),
    "ship": ("Ship casts on station", "#DDAA33", "v", True),
    "glider": ("Gliders", "#882255", "D", False),
    "argo": ("BGC-Argo floats", "#117733", "^", False),
}

# Station IDs that duplicate another ID's records (same site served by two ERDDAP
# servers; identical passes and values). The first-listed twin is kept.
DUPLICATE_IDS = {
    "WQB-05", "WQB-04", "tiburon-co2",                    # turbidity (b)
    "ioos-gliderdac-bios-jack-20190725T1346-delayed",     # glider Kd (a)
}
# Stations excluded from (b) on data grounds (user decision 2026-10-12).
EXCLUDED_TURBIDITY_IDS = {
    "exploratorium-turner",   # reports RFU (fluorescence units), not a turbidity unit
}

TURB_POINT = "0.72"        # pale grey: panel (b) has one category only
TURB_TREND = "#4477AA"     # muted blue binned-median trend (same blue as (a))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def classify_profiler(station_id: str) -> str:
    """Map a station id to its profiler class."""
    if SURFACE_RE.search(station_id):
        return "surface"
    if DEEP_RE.search(station_id):
        return "deep"
    return "castnear"


def load_kd_pairs(res_dir=RES):
    """Panel (a) input: one tidy frame of is2_kd, insitu_kd, series."""
    bs = pd.read_csv(os.path.join(res_dir, "paired_is2_insitu_kd.csv"))
    bs = bs.rename(columns={KDCOL: "is2_kd"})
    bs = bs[~bs.station_id.isin(DUPLICATE_IDS)]
    bs["series"] = bs.station_id.map(classify_profiler)
    bs.loc[bs.series == "castnear", "series"] = "glider"
    if "kind" in bs.columns:
        bs.loc[(bs["kind"] == "ship"), "series"] = "ship"
    bs = bs[["station_id", "series", "is2_kd", "insitu_kd"]]

    ar = pd.read_csv(os.path.join(res_dir, "paired_is2_argo_kd.csv"))
    ar = ar.rename(columns={"kind": "series"})
    ar["station_id"] = [f"argo_{i:03d}" for i in range(len(ar))]
    ar = ar[["station_id", "series", "is2_kd", "insitu_kd"]]

    parts = [bs, ar]
    gl = load_ooi_glider_pairs(res_dir)
    if gl is not None:
        parts.append(gl)
    d = pd.concat(parts, ignore_index=True)
    d = d[np.isfinite(d.is2_kd) & np.isfinite(d.insitu_kd)
          & (d.is2_kd > 0) & (d.insitu_kd > 0)]
    return d.reset_index(drop=True)


def load_ooi_glider_pairs(res_dir=RES, max_gap_h=6.0):
    """OOI Endurance + Pioneer glider matchups, re-aggregated at 8 km
    (glider_reaggregate_8km.py). One point per (granule, glider deployment): the
    median glider Kd_PAR and median 8 km ICESat-2 Kd over that pass's clusters
    within +/-max_gap_h. Returns None if the 8 km table has not been built."""
    f = os.path.join(res_dir, "glider_is2_8km.csv")
    if not os.path.exists(f):
        return None
    g = pd.read_csv(f)
    g = g[(g.dt_gap_hr.abs() <= max_gap_h) & g.is2_kd_8km.notna() & (g.is2_kd_8km > 0)
          & g.kd_par.notna() & (g.kd_par > 0)]
    ind = (g.groupby(["region", "granule", "dataset"])
             .agg(is2_kd=("is2_kd_8km", "median"), insitu_kd=("kd_par", "median"))
             .reset_index())
    ind["station_id"] = ind["dataset"] + "|" + ind["granule"]
    ind["series"] = "glider"
    return ind[["station_id", "series", "is2_kd", "insitu_kd", "region"]]


def load_turbidity_pairs(res_dir=RES):
    """Panel (b) input: is2_kd (8 km), is2_kd5 (5 km), turb, station_id."""
    d = pd.read_csv(os.path.join(res_dir, "paired_is2_turbidity.csv"))
    d = d[~d.station_id.isin(DUPLICATE_IDS | EXCLUDED_TURBIDITY_IDS)]
    d = d.rename(columns={KDCOL: "is2_kd", KDCOL5: "is2_kd5", "surf_turb": "turb"})
    for c in ("is2_kd", "is2_kd5", "turb"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[np.isfinite(d.is2_kd) & np.isfinite(d.turb) & (d.is2_kd > 0) & (d.turb > 0)]
    return d[["station_id", "is2_kd", "is2_kd5", "turb"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def series_stats(g):
    """n, median in-situ/IS-2 ratio, Spearman rho and p for one series."""
    x, y = g.is2_kd.values, g.insitu_kd.values
    rho, p = (spearmanr(x, y) if len(x) >= 3 else (np.nan, np.nan))
    return dict(n=len(x), ratio=float(np.median(y / x)), rho=rho, p=p)


def quantile_bin_medians(x, y, nbins=6):
    """Median y (with IQR) in equal-count bins of x. Returns arrays."""
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    xm, ym, lo, hi = [], [], [], []
    for i in range(nbins):
        m = (x >= edges[i]) & (x <= edges[i + 1] if i == nbins - 1 else x < edges[i + 1])
        if m.sum() < 5:
            continue
        xm.append(np.median(x[m]))
        ym.append(np.median(y[m]))
        lo.append(np.percentile(y[m], 25))
        hi.append(np.percentile(y[m], 75))
    xm, ym = np.array(xm), np.array(ym)
    return xm, ym, np.array(lo), np.array(hi)


def across_station_rho(d, ycol):
    """Spearman rho between per-station median IS-2 Kd and per-station median y."""
    st = d.groupby("station_id").agg(x=("is2_kd", "median"), y=(ycol, "median")).dropna()
    rho, p = spearmanr(st.x, st.y)
    return len(st), rho, p


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_kd(ax, d, xlim=KDLIM):
    """Panel (a): in-situ Kd (x) vs IS-2 Kd (y), grouped by profiler class.

    Square with equal aspect on the shared Kd range, so the 1:1 line sits at 45 deg.
    Plain journal styling: thin dashed 1:1 line named in the legend, open-edged
    markers, in-axes legend carrying the class median ratios.
    """
    lo, hi = xlim
    stats = {k: series_stats(d[d.series == k]) for k in SERIES if (d.series == k).any()}

    edge = np.array([lo, hi])
    ax.plot(edge, edge, color="0.35", ls="--", lw=1.0, zorder=1, label="1:1")

    for key, (label, color, marker, moored) in SERIES.items():
        g = d[d.series == key]
        if g.empty:
            continue
        # one format for every class: matchups and median in-situ / ICESat-2 ratio
        # (station / deployment / float counts are given in Table S2)
        # Class name wrapped and stats on their own line, so the legend fits the
        # data-free strip left of the points (see the legend call below).
        lab = "\n".join(textwrap.wrap(label, 21)
                        + [f"(n = {len(g)}, ratio {stats[key]['ratio']:.2f})"])
        # fixed-site symbols drawn above the (numerous) open mobile symbols
        ax.scatter(g.insitu_kd, g.is2_kd, marker=marker, s=38 if moored else 30,
                   linewidths=1.0 if moored else 0.9,
                   facecolors=(color if moored else "none"), edgecolors=color,
                   alpha=0.9 if moored else 0.8, label=lab, zorder=4 if moored else 3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("In-situ $K_d$ (m$^{-1}$)")
    ax.set_ylabel(KD_LABEL)
    # Upper left: with in-situ Kd on x the points fill a vertical band over the
    # full IS-2 range, and the only data-free space is the strip left of it
    # (in-situ Kd < ~0.02); the 1:1 line stays below this box. Short handles and
    # tight padding keep it ~10 pt clear of the leftmost (BGC-Argo) markers.
    ax.legend(loc="upper left", fontsize=9.5, frameon=True, framealpha=0.9,
              edgecolor="0.8", handlelength=1.2, handletextpad=0.3, borderpad=0.3,
              borderaxespad=0.4)

    return stats


def station_medians(t):
    """Per-station median and IQR of IS-2 Kd and turbidity, plus pass count."""
    g = t.groupby("station_id")
    q = lambda c, f: g[c].quantile(f)
    return pd.DataFrame(dict(
        n=g.size(),
        kd=g.is2_kd.median(), kd_lo=q("is2_kd", .25), kd_hi=q("is2_kd", .75),
        tb=g.turb.median(), tb_lo=q("turb", .25), tb_hi=q("turb", .75),
    )).reset_index()


def panel_turbidity(ax, t, xlim=KDLIM, tlo=0.03, thi=60.0):
    """Panel (b): station-median turbidity (x) vs station-median IS-2 Kd (y).

    Plotted at station level because the passes are not independent (478 passes
    from 47 stations, five stations supplying a third of them) and because
    turbidity is often reported to whole NTU, which turns a pass-level scatter
    into vertical stripes. The station-level Spearman rho is the statistic the
    text relies on, so this is the view that shows it.
    """
    lo, hi = xlim
    st = station_medians(t)
    ax.errorbar(st.tb, st.kd,
                xerr=[st.tb - st.tb_lo, st.tb_hi - st.tb],
                yerr=[st.kd - st.kd_lo, st.kd_hi - st.kd],
                fmt="none", ecolor=TURB_TREND, elinewidth=0.8, alpha=0.35, zorder=2)
    ax.scatter(st.tb, st.kd, s=34, facecolors=TURB_TREND, edgecolors=TURB_TREND,
               linewidths=0.8, alpha=0.85, zorder=3,
               label=f"Station median $\\pm$ IQR (n = {len(st)} stations)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(tlo, thi); ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)   # square, same size as panel (a), so the shared y-axes line up
    ax.set_xlabel("Near-surface turbidity (NTU or FNU)")
    ax.set_ylabel(KD_LABEL)
    ax.legend(loc="lower left", fontsize=9.5, frameon=True, framealpha=0.9,
              edgecolor="0.8", handletextpad=0.4, borderaxespad=0.6)

    rho, p = spearmanr(t.is2_kd, t.turb)
    n_st, rho_st, p_st = across_station_rho(t, "turb")
    t5 = t[np.isfinite(t.is2_kd5) & (t.is2_kd5 > 0)]
    rho5, p5 = spearmanr(t5.is2_kd5, t5.turb)
    # Station level only: passes at one station are not independent, so a
    # pass-level rho (still returned for the text) would overstate certainty.
    box = f"Spearman $\\rho$ = {rho_st:+.2f} ({n_st} stations)"
    ax.text(0.03, 0.97, box, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, linespacing=1.4, zorder=6)
    off = int(((st.tb < tlo) | (st.tb > thi) | (st.kd < lo) | (st.kd > hi)).sum())
    return dict(n=len(t), rho=rho, p=p, n_st=n_st, rho_st=rho_st, p_st=p_st,
                n5=len(t5), rho5=rho5, p5=p5, st_off=off,
                tb_range=(float(st.tb.min()), float(st.tb.max())))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(d, t, xlim=KDLIM, kd_floor=0.0):
    fk.set_pub_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.0, 5.6))

    stats_a = panel_kd(ax_a, d, xlim=xlim)
    stats_b = panel_turbidity(ax_b, t, xlim=xlim)

    if kd_floor > 0:
        # never let a filtered figure look unfiltered
        for ax in (ax_a, ax_b):   # IS-2 Kd is the y-axis in both panels
            ax.axhline(kd_floor, color="0.55", ls="-.", lw=1.0, zorder=1)
        ax_a.text(0.98, 0.02,
                  f"retrievals with $K_d$ < {kd_floor:g} m$^{{-1}}$ excluded",
                  transform=ax_a.transAxes, ha="right", va="bottom", fontsize=9.5,
                  color="0.35", zorder=6)
    # Same "(a) Title" headings as the appendix figures (figkit.panel_label).
    fk.panel_label(ax_a, "a", "Comparison with in-situ $K_d$")
    fk.panel_label(ax_b, "b", "Relationship with turbidity")
    fig.tight_layout(w_pad=2.5)
    return fig, stats_a, stats_b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=RES)
    ap.add_argument("--output", default=os.path.join(RES, "fig_is2_validation.png"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--kd-floor", type=float, default=0.0,
                    help="Drop matchups with retrieved IS-2 Kd below this value "
                         "(m^-1) from both panels and all statistics. Default 0 = "
                         "keep everything (see the note at KD floor in this file).")
    ap.add_argument("--kdlim", "--xlim", dest="kdlim", default=f"{KDLIM[0]},{KDLIM[1]}",
                    help="Shared IS-2 Kd axis limits as lo,hi (default %(default)s). "
                         "Use 2e-4,3.0 to clip nothing.")
    args = ap.parse_args()

    d = load_kd_pairs(args.input_dir)
    t = load_turbidity_pairs(args.input_dir)
    xlim = tuple(float(v) for v in args.kdlim.split(","))
    if args.kd_floor > 0:
        n_a0, n_b0 = len(d), len(t)
        d = d[d.is2_kd >= args.kd_floor].reset_index(drop=True)
        t = t[t.is2_kd >= args.kd_floor].reset_index(drop=True)
        print(f"kd-floor {args.kd_floor} m^-1 applied: (a) {n_a0 - len(d)}/{n_a0} "
              f"dropped, (b) {n_b0 - len(t)}/{n_b0} dropped")
        xlim = (max(xlim[0], args.kd_floor * 0.8), xlim[1])
    fig, stats_a, stats_b = make_figure(d, t, xlim=xlim, kd_floor=args.kd_floor)
    fk.save_fig(fig, args.output, dpi=args.dpi)

    off_a = int(((d.is2_kd < xlim[0]) | (d.is2_kd > xlim[1])).sum())
    off_b = int(((t.is2_kd < xlim[0]) | (t.is2_kd > xlim[1])).sum())
    print(f"shared IS-2 Kd window {xlim}: (a) {off_a}/{len(d)} outside, "
          f"(b) {off_b}/{len(t)} outside (statistics use ALL data)")
    print(f"(a) Kd pairs: {len(d)}   data span "
          f"[{d.is2_kd.min():.4f}, {d.is2_kd.max():.3f}]")
    for key, s in stats_a.items():
        print(f"      {key:9s} n={s['n']:>3d}  in-situ/IS-2={s['ratio']:.2f}  "
              f"rho={s['rho']:+.2f} (p={s['p']:.3f})")
    n_st, rho_st, p_st = across_station_rho(d, "insitu_kd")
    prof = d[d.series != "argo"]
    n_p, rho_p, p_p = across_station_rho(prof, "insitu_kd")
    print(f"      across-station all n={n_st} rho={rho_st:+.2f} (p={p_st:.3f}) | "
          f"profilers only n={n_p} rho={rho_p:+.2f} (p={p_p:.3f})")
    print(f"(b) turbidity pairs: {stats_b['n']}   data span "
          f"[{t.is2_kd.min():.5f}, {t.is2_kd.max():.3f}]")
    print(f"      per-pass rho={stats_b['rho']:+.2f} (p={stats_b['p']:.2e}) | "
          f"per-station n={stats_b['n_st']} rho={stats_b['rho_st']:+.2f} "
          f"(p={stats_b['p_st']:.4f}) | 5 km rho={stats_b['rho5']:+.2f}")
    print(f"      station medians: turbidity range {stats_b['tb_range']}, "
          f"{stats_b['st_off']} station(s) outside the plotted window")
    print(f"wrote {args.output} and {os.path.splitext(args.output)[0]}.pdf")


if __name__ == "__main__":
    main()
