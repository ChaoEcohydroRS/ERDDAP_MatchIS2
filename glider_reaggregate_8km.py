#!/usr/bin/env python
"""Re-aggregate the OOI glider <-> ICESat-2 matchups at an 8 km radius.

The glider study (A_OOI_Glider_Kd_WA_OR) stored ICESat-2 Kd medians at 5 and 10 km
only. Panel (a) of fig_is2_validation uses 8 km, so this re-reads the kept per-pass
Kd CSVs (no SlideRule calls) and computes, for every paired glider cluster, the
median kd_coherent of all bins with Kd > 0 within RADIUS km of the glider position
(ungated, as in panel (a)). Same pipeline recipe as the ERDDAP matchups
(--enable_gebco_filter --poisson_weighted --coherent_model_selection
--min_seabed_depth 3, SlideRule input, July 2026 code), so values are comparable.

Self-check: the 10 km median is recomputed the same way and compared with the value
stored in the glider study's paired table.

Output (rewritten after every finished file, so a killed job leaves partial results):
  results/glider_is2_8km.csv
"""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

# Data dir of the OOI glider study (A_OOI_Glider_Kd_WA_OR); not part of this repo.
G = os.environ.get("GLIDER_DATA", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "A_OOI_Glider_Kd_WA_OR", "data"))
SETS = {  # region: (paired table, per-pass run dir)
    "OOI Endurance (WA/OR)": (f"{G}/processed/glider_is2_paired.csv", f"{G}/interim/matchcalc_runs_boot200"),
    "OOI Pioneer (Mid-Atlantic Bight)": (f"{G}/processed/mab_glider_is2_paired.csv", f"{G}/interim/mab_matchcalc_runs_boot200"),
}
RADIUS = float(os.environ.get("RADIUS_KM", 8.0))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "glider_is2_8km.csv")


def haversine_km(lat0, lon0, lat, lon):
    p0, p = np.radians(lat0), np.radians(lat)
    a = (np.sin((p - p0) / 2) ** 2
         + np.cos(p0) * np.cos(p) * np.sin(np.radians(lon - lon0) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def work(region, run_dir, granule, dataset, rows):
    d = os.path.join(run_dir, f"{granule}__{dataset}")
    hits = [f for f in os.listdir(d) if "AddedKdDataset" in f and f.endswith(".csv")] if os.path.isdir(d) else []
    out = []
    if not hits:
        for r in rows:
            out.append({**r, "region": region, "n_is2_8km": 0, "is2_kd_8km": np.nan,
                        "is2_kd_10km_recomputed": np.nan, "status": "no_kd_file"})
        return out
    k = pd.read_csv(os.path.join(d, sorted(hits)[0]), low_memory=False,
                    usecols=lambda c: c in ("latitude", "longitude", "kd_coherent", "kd"))
    col = "kd_coherent" if "kd_coherent" in k.columns else "kd"
    kd = pd.to_numeric(k[col], errors="coerce")
    ok = kd.notna() & (kd > 0)
    lat, lon, kd = k.latitude[ok].to_numpy(), k.longitude[ok].to_numpy(), kd[ok].to_numpy()
    for r in rows:
        dist = haversine_km(r["glider_lat"], r["glider_lon"], lat, lon)
        m8, m10 = dist <= RADIUS, dist <= 10.0
        out.append({**r, "region": region, "n_is2_8km": int(m8.sum()),
                    "is2_kd_8km": float(np.median(kd[m8])) if m8.any() else np.nan,
                    "is2_kd_10km_recomputed": float(np.median(kd[m10])) if m10.any() else np.nan,
                    "status": "ok"})
    return out


def main():
    keep = ["granule", "dataset", "cluster", "glider_lat", "glider_lon", "pass_t0",
            "dt_gap_hr", "kd_par", "kd_par_r2", "kd_par_n_dives", "is2_kd_10km",
            "glider_z_top", "glider_z_bot", "par_dt_gap_hr"]
    jobs = []
    for region, (paired, run_dir) in SETS.items():
        p = pd.read_csv(paired)
        print(f"[plan] {region}: {len(p)} paired clusters, "
              f"{p.groupby(['granule','dataset']).ngroups} per-pass files, radius {RADIUS} km", flush=True)
        for (g, ds), grp in p.groupby(["granule", "dataset"]):
            jobs.append((region, run_dir, g, ds, grp[keep].to_dict("records")))
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"[plan] {len(jobs)} files across {workers} workers", flush=True)
    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result(); results.extend(res)
            j = futs[f]
            n8 = sum(1 for r in res if np.isfinite(r["is2_kd_8km"]))
            print(f"[{i}/{len(jobs)}] {j[0][:14]} {j[2]} {j[3]}: {len(res)} clusters, "
                  f"{n8} with 8 km Kd ({time.time()-t0:.0f}s)", flush=True)
            if i % 25 == 0 or i == len(jobs):
                pd.DataFrame(results).to_csv(OUT, index=False)
    df = pd.DataFrame(results); df.to_csv(OUT, index=False)
    both = df.dropna(subset=["is2_kd_10km", "is2_kd_10km_recomputed"])
    agree = np.isclose(both.is2_kd_10km, both.is2_kd_10km_recomputed, rtol=1e-6).mean() if len(both) else np.nan
    print(f"[check] 10 km recomputed vs stored: {agree*100:.1f}% identical over {len(both)} clusters", flush=True)
    s = df[(df.dt_gap_hr.abs() <= 6) & df.is2_kd_8km.notna() & df.kd_par.notna()]
    for region, g in s.groupby("region"):
        ind = g.groupby(["granule", "dataset"]).agg(is2=("is2_kd_8km", "median"), par=("kd_par", "median"))
        print(f"[result] {region}: |dt|<=6h pairs {len(g)}, independent {len(ind)}, "
              f"median ratio {np.median(ind.par/ind.is2):.2f}", flush=True)
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
