#!/usr/bin/env python
"""Re-aggregate IS-2 Kd at a new station radius from the kept per-pass CSVs.

The original 03d run aggregated each pass at 5 and 10 km only (03d now also
writes 8 km by default). The photon fetch box is an 8 km half-width square, so a
10 km circle is only partly covered; 8 km is the largest radius fully inside the
box. This script re-reads the per-pass *_AddedKdDataset_strongBeams_Further.csv
files under _sliderule_kd_runs/ (no new SlideRule calls) and adds or refreshes the
suffixed <r>km columns in

  results/is2_kd_sliderule_<tag>.csv          (tag = buoys, ships, turbidity, argo)

Then rerun 04_pair_insitu.py to rebuild the paired tables.

    python reaggregate_radius.py --radius 8
"""
import argparse
import glob
import os
from types import SimpleNamespace

import pandas as pd

import importlib
sr = importlib.import_module("03d_is2kd_sliderule")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RUNS = os.path.join(HERE, "_sliderule_kd_runs")
TAGS = ("buoys", "ships", "turbidity", "argo")
KEY = ["station_id", "t0", "rgt"]


def label_for(row):
    """Same label 03d_is2kd_sliderule.py used for the per-pass work dir."""
    return (f"{row.station_id}_{int(row.rgt)}_{str(row.t0)[:10]}"
            .replace(":", "").replace("/", "-"))


def index_csvs(tag):
    """label -> per-pass CSV path, searching every array-task dir of this tag."""
    idx = {}
    for p in glob.glob(os.path.join(RUNS, f"{tag}_*", "*",
                                    "*_AddedKdDataset_strongBeams_Further.csv")):
        lab = os.path.basename(os.path.dirname(p))
        idx[lab] = p          # later task dirs win, matching merge_sr_parts keep="last"
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=8.0)
    a = ap.parse_args()
    r = a.radius
    rk = f"{int(r) if r.is_integer() else r}km"
    newcols = [f"n_bins_{rk}", f"is2_kd_med_{rk}", f"is2_kd_coherent_med_{rk}",
               f"n_trusted_{rk}", f"is2_kd_coherent_trusted_med_{rk}"]
    print(f"radius {r} km -> columns {newcols}", flush=True)

    for tag in TAGS:
        path = os.path.join(RES, f"is2_kd_sliderule_{tag}.csv")
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in newcols if c in df.columns])
        idx = index_csvs(tag)
        rows, missing = [], 0
        for row in df.itertuples(index=False):
            rec = {k: getattr(row, k) for k in KEY}
            if row.status == "ok":
                p = idx.get(label_for(row))
                if p is None:
                    missing += 1
                else:
                    out = sr.aggregate_near_station(
                        p, SimpleNamespace(lat=row.station_lat, lon=row.station_lon), [r])
                    if out:
                        rec.update({c: out[c] for c in newcols})
            rows.append(rec)
        add = pd.DataFrame(rows)
        df = pd.concat([df.reset_index(drop=True), add[newcols].reset_index(drop=True)]
                       if set(newcols) <= set(add.columns) else [df], axis=1)
        df.to_csv(path, index=False)
        ok = df[df.status == "ok"]
        have = ok[f"is2_kd_coherent_med_{rk}"].notna().sum()
        print(f"[{tag}] ok rows {len(ok)}, per-pass CSV missing {missing}, "
              f"with {rk} Kd {have}", flush=True)
    print("Now rerun 04_pair_insitu.py to rebuild the paired tables.", flush=True)


if __name__ == "__main__":
    main()
