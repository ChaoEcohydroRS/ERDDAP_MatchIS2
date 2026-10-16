#!/usr/bin/env python3
"""Step 03d (SlideRule): IS-2 Kd for each station-matched ATL03 pass, via SlideRule.

Replaces the deprecated SlideRule-fork ``03d_is2kd_buoys.py``. This driver does NOT
re-implement the Kd pipeline — it calls the validated ``main.py --input_source
sliderule`` entry point from the sibling repo as a subprocess, so it can never drift
from the algorithm that produced the figures / the reproducibility PASS (2026-07-24:
raw kd bias 0.0000, r=0.94 vs a local v007 run). For each manifest row it:

  1. runs main.py on the station bbox / rgt / ±window (SlideRule server-side subset,
     no granule download; ATL03 v007; recipe = the figure recipe),
  2. reads the per-beam AddedKdDataset CSV,
  3. keeps bins within --radius-km of the station and aggregates kd_coherent
     (+ raw kd, + the figure "trusted" gate), pairing with station metadata.

Manifest schema (results/02_match_*_manifest.csv):
  station_id,source,lat,lon,granule_filename,rgt,cycle,segment,t0,t1,
  poly_W,poly_S,poly_E,poly_N,erddap_csv,erddap_rows

Usage:
  python 03d_is2kd_sliderule.py \
      --manifest results/02_match_buoys_manifest.csv \
      --output   results/is2_kd_sliderule_buoys.csv \
      [--radius-km 5] [--limit N | --start I --end J] [--with-atl09]

Resume: skips manifest rows already in logs/03d_sr_progress.json. Safe to re-run.
Heavy: one main.py run per row (~5-15 min each). Do a --limit smoke test first,
then scale via an sbatch array (each task a --start/--end slice). Never the login node.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
# The Kd pipeline repo (ChaoEcohydroRS/RiverMouths_IS2KdMassiveTest) and the
# workspace holding Dataset/Bathy/gebco_2024_geotiff/ and Dataset/Shorelines/.
# Set by environment variable, or pass --kdph-dir / --workspace-path.
KDPH_DIR_DEFAULT = os.environ.get(
    "IS2KD_REPO", os.path.join(os.path.dirname(ROOT), "RiverMouths_IS2KdMassiveTest"))
WORKSPACE_DEFAULT = os.environ.get("IS2KD_WORKSPACE", os.path.dirname(ROOT)) + os.sep
PYTHON = os.environ.get("IS2KD_PYTHON", sys.executable)

# Validated figure recipe (2026-06-25 minsb3 runs; reproduced via SlideRule 2026-07-24).
RECIPE = ["--enable_gebco_filter", "--poisson_weighted",
          "--coherent_model_selection", "--min_seabed_depth", "3"]
ALL_STRONG = "gt1l,gt2l,gt3l,gt1r,gt2r,gt3r"


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, f"03d_sr_{ts}.log")),
                  logging.StreamHandler(sys.stdout)])
    return logging.getLogger("03d_sr")


def load_progress(p):
    return json.load(open(p)) if os.path.exists(p) else {"done": [], "failed": []}


def save_progress(p, state):
    json.dump(state, open(p, "w"), indent=2)


def haversine_km(lat0, lon0, lat, lon):
    R = 6371.0
    p0, p1 = np.radians(lat0), np.radians(lat)
    dphi = np.radians(lat - lat0)
    dlmb = np.radians(lon - lon0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def station_bbox(lat, lon, half_km):
    """Station-centered bbox (W,S,E,N). The manifest's own poly_* is station-centered
    but only ~±4 km, so a track matched at up to the ~5 km criterion falls just OUTSIDE
    it (verified: wqb_04 track sits 4.4 km W of the station, 0.4 km past the box edge →
    0 photons). Build from the station ± half_km instead; aggregation keeps only bins
    within --radius-km."""
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(0.15, np.cos(np.radians(lat))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def run_main_sliderule(row, out_dir, label, args, logger):
    """Invoke main.py --input_source sliderule for one row. Returns (rc, csv_path)."""
    os.makedirs(out_dir, exist_ok=True)
    W, S, E, N = station_bbox(row.lat, row.lon, args.fetch_km)
    bbox = f"{W},{S},{E},{N}"
    # Normalise timestamps to clean ISO 'Z' — SlideRule's CMR query fails on fractional
    # seconds / '+00:00' offsets (e.g. the argo manifest '...040000293+00:00' → 0 photons).
    sr_t0 = pd.to_datetime(row.t0, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    sr_t1 = pd.to_datetime(row.t1, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    cmd = [PYTHON, "main.py",
           "--workspace_path", args.workspace_path,
           "--input_source", "sliderule",
           f"--sr_bbox={bbox}",              # '=' so a leading '-' lon is not read as a flag
           "--sr_t0", sr_t0, "--sr_t1", sr_t1,
           "--sr_rgt", str(int(row.rgt)),
           "--sr_label", label,
           "--sr_atl09_fields", ("high_rate/backg_c" if args.with_atl09 else ""),
           "--other_data_path", args.other_data_path,
           "--shoreline_data", args.shoreline_data,
           "--output_path", out_dir,
           "--target_beams", ALL_STRONG,
           "--no_plot"] + RECIPE
    if args.with_atl09:
        cmd += ["--min_noise_floor_photons_per_bin", str(args.min_noise_floor)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=args.kdph_dir, capture_output=True, text=True,
                              timeout=args.timeout_per_row)
    except subprocess.TimeoutExpired:
        logger.warning("  TIMEOUT after %ss", args.timeout_per_row)
        return 124, None
    dt = time.time() - t0
    csv_path = os.path.join(out_dir, f"{label}_AddedKdDataset_strongBeams_Further.csv")
    if proc.returncode != 0 or not os.path.exists(csv_path):
        # main.py exits 2/3/4 for no-photons / no-subsurface / no-beams (not a crash)
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
        logger.info("  main.py rc=%d (%.0fs) — %s", proc.returncode, dt, tail[0][:120])
        return proc.returncode, None
    logger.info("  main.py OK (%.0fs) -> %s", dt, os.path.basename(csv_path))
    return 0, csv_path


def aggregate_near_station(csv_path, row, radii):
    """Kd stats for bins within each radius (km) of the station.

    Returns min_dist_km + beams (nearest bin / max radius) plus, for each r in
    ``radii``, suffixed columns  n_bins_<r>km / is2_kd_med_<r>km /
    is2_kd_coherent_med_<r>km / n_trusted_<r>km / is2_kd_coherent_trusted_med_<r>km.
    None if the CSV has no valid kd at all.
    """
    kd = pd.read_csv(csv_path)
    kd = kd[kd["kd"].notna()].copy()
    if kd.empty:
        return None
    kd["dist_km"] = haversine_km(row.lat, row.lon,
                                 kd["latitude"].to_numpy(), kd["longitude"].to_numpy())
    kc = "kd_coherent" if "kd_coherent" in kd.columns else "kd"
    # figure "trusted" gate: bootstrap CI succeeded, good surface fit, tight CI
    is_trusted = (kd["kd_std"].notna()
                  & (kd.get("surface_fit_quality", 1.0) >= 0.75)
                  & ((kd["kd_ci_high"] - kd["kd_ci_low"]) <= 0.5))
    out = {"min_dist_km": float(kd["dist_km"].min()),
           "beams_maxr": ",".join(sorted(kd[kd["dist_km"] <= max(radii)]["beam_id"]
                                          .astype(str).unique()))}
    for r in radii:
        near = kd[kd["dist_km"] <= r]
        tr = kd[(kd["dist_km"] <= r) & is_trusted]
        rk = f"{int(r) if float(r).is_integer() else r}km"
        out[f"n_bins_{rk}"] = int(len(near))
        out[f"is2_kd_med_{rk}"] = float(near["kd"].median()) if len(near) else np.nan
        out[f"is2_kd_coherent_med_{rk}"] = float(near[kc].median()) if len(near) else np.nan
        out[f"n_trusted_{rk}"] = int(len(tr))
        out[f"is2_kd_coherent_trusted_med_{rk}"] = float(tr[kc].median()) if len(tr) else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="results/02_match_buoys_manifest.csv")
    ap.add_argument("--output", default="results/is2_kd_sliderule_buoys.csv")
    ap.add_argument("--radii", type=str, default="5,8,10",
                    help="Comma-separated station radii (km) to aggregate Kd at; each gets "
                         "its own suffixed output columns. Default '5,8,10'. The paper uses "
                         "8 km, the largest circle fully inside the 8 km fetch box.")
    ap.add_argument("--fetch-km", type=float, default=8.0,
                    help="Half-width of the station-centered SlideRule bbox (km). Must "
                         "exceed the step-02 match radius so the matched track is inside; "
                         "the manifest poly_* (~±4 km) is too tight. Default 8.")
    ap.add_argument("--kdph-dir", default=KDPH_DIR_DEFAULT)
    ap.add_argument("--workspace-path", default=WORKSPACE_DEFAULT)
    ap.add_argument("--other-data-path", default="Dataset/")
    ap.add_argument("--shoreline-data", default="Shorelines/ne_10m_land/ne_10m_land.shp")
    ap.add_argument("--work-dir", default=os.path.join(ROOT, "_sliderule_kd_runs"),
                    help="Per-row main.py output dir (large subsurface CSVs deleted after).")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    ap.add_argument("--with-atl09", action="store_true",
                    help="Enable ATL09 backg_c (CMR+h5) + daytime Poisson-floor gate. "
                         "Off by default = the validated figure recipe.")
    ap.add_argument("--min-noise-floor", type=float, default=2.0,
                    help="Poisson-floor threshold when --with-atl09 (daytime only).")
    ap.add_argument("--timeout-per-row", type=int, default=1800)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=-1, help="Process at most N rows (smoke test).")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="Keep the large per-row subsurface photon CSVs (default: delete).")
    args = ap.parse_args()

    logger = setup_logger(args.log_dir)
    # Progress file is derived from --output so parallel SLURM array tasks (each a
    # distinct --output part file) never race on a shared progress json.
    progress_path = os.path.join(
        args.log_dir, os.path.basename(args.output).replace(".csv", "") + ".progress.json")

    radii = [float(x) for x in str(args.radii).split(",") if x.strip()]
    man = pd.read_csv(args.manifest)
    end = args.end if args.end != -1 else len(man)
    man = man.iloc[args.start:end].reset_index(drop=True)
    if args.limit != -1:
        man = man.iloc[:args.limit]
    logger.info("Manifest %s -> %d rows (start=%d end=%d limit=%s), radii=%s km, "
                "fetch=%.1f km, atl09=%s", args.manifest, len(man), args.start, end,
                args.limit, radii, args.fetch_km, args.with_atl09)

    prog = load_progress(progress_path)
    done = set(prog["done"])
    results = []
    prev = pd.read_csv(args.output) if os.path.exists(args.output) else None
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    def write_output():
        """Rewrite the output with every row so far, so a killed job keeps them."""
        out_df = pd.DataFrame(results)
        if prev is not None:
            out_df = pd.concat([prev, out_df], ignore_index=True).drop_duplicates(
                subset=["station_id", "t0", "rgt"], keep="last")
        out_df.to_csv(args.output, index=False)
        return out_df
    for i, row in enumerate(man.itertuples(index=False)):
        key = f"{row.station_id}|{row.t0}|{int(row.rgt)}"
        if key in done:
            logger.info("[%d/%d] SKIP (done): %s", i + 1, len(man), key)
            continue
        logger.info("[%d/%d] %s  station=(%.3f,%.3f) rgt=%d",
                    i + 1, len(man), key, row.lat, row.lon, int(row.rgt))
        label = f"{row.station_id}_{int(row.rgt)}_{str(row.t0)[:10]}".replace(":", "").replace("/", "-")
        out_dir = os.path.join(args.work_dir, label)
        rec = {"station_id": row.station_id, "station_lat": row.lat, "station_lon": row.lon,
               "t0": row.t0, "t1": row.t1, "rgt": int(row.rgt),
               "granule": row.granule_filename,
               "erddap_csv": getattr(row, "erddap_csv", "")}
        rc, csv_path = run_main_sliderule(row, out_dir, label, args, logger)
        if rc == 0 and csv_path:
            agg = aggregate_near_station(csv_path, row, radii)
            if agg is None:
                rec.update({"status": "no_valid_kd"})
            else:
                rec.update({"status": "ok", **agg})
                r0 = f"{int(radii[0]) if float(radii[0]).is_integer() else radii[0]}km"
                n0, kc0 = agg.get(f"n_bins_{r0}", 0), agg.get(f"is2_kd_coherent_med_{r0}")
                logger.info("  -> %d bins <= %s, kd_coherent_med=%s (trusted n=%s); min_dist=%.1f km",
                            n0, r0, f"{kc0:.4f}" if n0 else "NA",
                            agg.get(f"n_trusted_{r0}"), agg["min_dist_km"])
            prog["done"].append(key)
        else:
            rec.update({"status": f"main_rc_{rc}"})
            if key not in prog["failed"]:
                prog["failed"].append(key)
        results.append(rec)
        write_output()
        save_progress(progress_path, prog)
        # reclaim disk: the subsurface photon CSV is ~14 MB/row
        if not args.keep_intermediate and os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                if "subsurface_photons" in f:
                    try:
                        os.remove(os.path.join(out_dir, f))
                    except OSError:
                        pass

    if not results:
        logger.info("No new rows processed.")
        return
    out_df = write_output()
    ok = (out_df["status"] == "ok").sum()
    logger.info("Wrote %s: %d rows (%d ok). Failed keys: %d",
                args.output, len(out_df), ok, len(prog["failed"]))


if __name__ == "__main__":
    main()
