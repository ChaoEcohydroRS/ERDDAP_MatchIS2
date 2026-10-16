#!/usr/bin/env python3
"""Merge the per-array-task part files from run_is2kd_sliderule_array.sbatch.

Concatenates results/_parts/is2_kd_<tag>_part_*.csv into one CSV, de-duplicated on
(station_id, t0, rgt), and prints a quick yield summary.

Usage:
  python merge_sr_parts.py --tag buoys
"""
import argparse
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="buoys")
    ap.add_argument("--parts-dir", default=os.path.join(ROOT, "results", "_parts"))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    out = args.output or os.path.join(ROOT, "results", f"is2_kd_sliderule_{args.tag}.csv")

    parts = sorted(glob.glob(os.path.join(args.parts_dir, f"is2_kd_{args.tag}_part_*.csv")))
    if not parts:
        raise SystemExit(f"No part files matching is2_kd_{args.tag}_part_*.csv in {args.parts_dir}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["station_id", "t0", "rgt"], keep="last")
    df = df.sort_values(["station_id", "t0"]).reset_index(drop=True)
    df.to_csv(out, index=False)

    n = len(df)
    ok = int((df["status"] == "ok").sum()) if "status" in df.columns else 0
    print(f"Merged {len(parts)} parts -> {out}")
    print(f"  rows={n}  ok={ok} ({100*ok/max(n,1):.0f}%)")
    if "status" in df.columns:
        print("  status counts:", df["status"].value_counts().to_dict())
    if ok:
        okrows = df[df["status"] == "ok"]
        for col in ("is2_kd_coherent_med_5km", "is2_kd_coherent_med_10km"):
            if col in okrows.columns:
                v = okrows[col].dropna()
                if len(v):
                    print(f"  {col}: n={len(v)} median={v.median():.4f} "
                          f"IQR=[{v.quantile(.25):.4f},{v.quantile(.75):.4f}]")


if __name__ == "__main__":
    main()
