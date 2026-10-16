#!/usr/bin/env python3
"""Step 04: pair ICESat-2 Kd (step 03d) with the in-situ ground truth (steps 03a-c).

Writes the three tables behind Fig. 5 and Table S:

  results/paired_is2_insitu_kd.csv   buoy/mooring + ship-cast Kd_PAR  vs IS-2 Kd
  results/paired_is2_turbidity.csv   near-surface turbidity           vs IS-2 Kd
  results/paired_is2_argo_kd.csv     BGC-Argo Kd(490)                 vs IS-2 Kd

Join key for buoys/ships/turbidity is the ERDDAP file of the pass (``erddap_csv``
in the IS-2 table == ``source_file`` in the in-situ table): one file holds the
in-situ data within +/-6 h of one ICESat-2 pass at one station. Only IS-2 rows
with status ``ok`` are paired.

In-situ value per pass
  Kd_PAR     median Kd of the casts in the file with |r| >= --min-abs-r (0.9)
  turbidity  median of readings with |z| <= --turb-max-depth (5 m) and
             0 < turb < --turb-max (500), excluding --drop-units (RFU)
  Argo       kd_490_Xing of the float profile (Begouen Demeaux & Boss dataset).
             The Argo source has one row per float x satellite matchup, so a
             profile appears several times; rows are collapsed to one point per
             profile (median IS-2 Kd over its rows, which share one pass).
             --argo-max-gap-h keeps only profiles whose pass is within that many
             hours (granule start time vs float time); default: no extra cut
             (the step-02 search window, +/-12 h).

Station duplicates served by two ERDDAP servers under different IDs (e.g.
WQB-05 / wqb_05) and the plotted subset are handled in fig_is2_validation.py.

    python 04_pair_insitu.py                      # paper tables
    python 04_pair_insitu.py --argo-max-gap-h 6   # Argo restricted to +/-6 h
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KDCOL = "is2_kd_coherent_med_8km"


def base(s):
    return s.astype(str).map(os.path.basename)


def ok_rows(path):
    d = pd.read_csv(path)
    d = d[d.status == "ok"].copy()
    d["src_base"] = base(d.erddap_csv)
    return d


def pair_kd(res, min_abs_r):
    parts = []
    for kind, tag, kd_file in [("buoy", "buoys", "03a_buoy_kd.csv"),
                               ("ship", "ships", "ship_kd.csv")]:
        is2 = ok_rows(os.path.join(res, f"is2_kd_sliderule_{tag}.csv"))
        k = pd.read_csv(os.path.join(res, kd_file))
        k = k[k.rval.abs() >= min_abs_r].copy()
        k["src_base"] = base(k.source_file)
        g = (k.groupby("src_base")
               .agg(insitu_kd=("kd", "median"), n_casts=("kd", "size"))
               .reset_index())
        m = is2.merge(g, on="src_base", how="inner")
        m.insert(m.columns.get_loc("src_base") + 1, "kind", kind)
        print(f"  {kind:5s}: {len(is2)} IS-2 ok passes, {len(m)} with in-situ Kd "
              f"(|r| >= {min_abs_r})")
        parts.append(m)
    return pd.concat(parts, ignore_index=True)


def pair_turbidity(res, max_depth, turb_max, drop_units):
    is2 = ok_rows(os.path.join(res, "is2_kd_sliderule_turbidity.csv"))
    t = pd.read_csv(os.path.join(res, "turbidity_compiled.csv"))
    t["src_base"] = base(t.source_file)
    keep = (t.z.abs() <= max_depth) & (t.turb > 0) & (t.turb < turb_max)
    if drop_units and "turb_units" in t.columns:
        bad = t.turb_units.fillna("").str.upper().isin([u.upper() for u in drop_units])
        n_bad = t.loc[keep & bad, "src_base"].nunique()
        if n_bad:
            print(f"  turbidity: dropped {n_bad} file(s) reporting {drop_units}")
        keep &= ~bad
    g = (t[keep].groupby("src_base")
                .agg(surf_turb=("turb", "median"), n_turb=("turb", "size"))
                .reset_index())
    m = is2.merge(g, on="src_base", how="inner")
    print(f"  turbidity: {len(is2)} IS-2 ok passes, {len(m)} with near-surface "
          f"turbidity (|z| <= {max_depth} m, 0 < turb < {turb_max}), "
          f"{m.station_id.nunique()} station IDs")
    return m


def pair_argo(res, max_gap_h):
    is2 = pd.read_csv(os.path.join(res, "is2_kd_sliderule_argo.csv"))
    ins = pd.read_csv(os.path.join(res, "02_match_argo_profiles.csv"))
    is2 = is2[is2.status == "ok"].copy()
    # time between the float profile and the granule start (~ the overpass)
    g_t = pd.to_datetime(is2.granule.str.slice(6, 20), format="%Y%m%d%H%M%S", utc=True)
    f_t = pd.to_datetime(is2.t0, utc=True, format="ISO8601") + (
        pd.to_datetime(is2.t1, utc=True, format="ISO8601")
        - pd.to_datetime(is2.t0, utc=True, format="ISO8601")) / 2
    is2["gap_h"] = ((g_t - f_t).dt.total_seconds() / 3600).abs()
    is2["k"] = list(zip(is2.station_lat.round(4), is2.station_lon.round(4)))
    ins["k"] = list(zip(ins.lat_float.round(4), ins.lon_float.round(4)))
    ins_u = ins.drop_duplicates("k")[["k", "ID_float", "dt_float", "kd_490_Xing"]]
    agg = (is2.groupby("k")
              .agg(station_lat=("station_lat", "first"),
                   station_lon=("station_lon", "first"),
                   is2_kd=(KDCOL, "median"),
                   n_rows=(KDCOL, "size"),
                   min_dist_km=("min_dist_km", "min"),
                   gap_h=("gap_h", "min"))
              .reset_index())
    m = agg.merge(ins_u, on="k", how="left").rename(columns={"kd_490_Xing": "insitu_kd"})
    m["kind"] = "argo"
    m = m[m.is2_kd.notna() & m.insitu_kd.notna()].copy()
    n0 = len(m)
    if max_gap_h is not None:
        m = m[m.gap_h <= max_gap_h]
    print(f"  argo : {is2.k.nunique()} profiles with IS-2 Kd, {n0} with float Kd(490)"
          + (f", {len(m)} within +/-{max_gap_h:g} h" if max_gap_h is not None else "")
          + f" ({m.ID_float.nunique()} floats)")
    return m[["kind", "ID_float", "dt_float", "station_lat", "station_lon", "is2_kd",
              "insitu_kd", "n_rows", "min_dist_km", "gap_h"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=RES)
    ap.add_argument("--min-abs-r", type=float, default=0.9,
                    help="Minimum |r| of a Kd_PAR cast fit (default 0.9)")
    ap.add_argument("--turb-max-depth", type=float, default=5.0)
    ap.add_argument("--turb-max", type=float, default=500.0)
    ap.add_argument("--drop-units", default="RFU",
                    help="Comma-separated turbidity units to exclude (default RFU)")
    ap.add_argument("--argo-max-gap-h", type=float, default=None)
    a = ap.parse_args()
    res = a.results_dir
    drop = [u for u in a.drop_units.split(",") if u.strip()]

    kd = pair_kd(res, a.min_abs_r)
    kd.to_csv(os.path.join(res, "paired_is2_insitu_kd.csv"), index=False)
    tb = pair_turbidity(res, a.turb_max_depth, a.turb_max, drop)
    tb.to_csv(os.path.join(res, "paired_is2_turbidity.csv"), index=False)
    ar = pair_argo(res, a.argo_max_gap_h)
    ar.to_csv(os.path.join(res, "paired_is2_argo_kd.csv"), index=False)
    print(f"wrote paired tables to {res}: Kd {len(kd)}, turbidity {len(tb)}, argo {len(ar)}")


if __name__ == "__main__":
    main()
