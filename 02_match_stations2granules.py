#!/usr/bin/env python3
"""
02_match_stations2granules.py

Unified CLI script that replaces the four 02_match_*2granules_ipx.ipynb notebooks
written by Hannah Glover (kept in legacy/).
Matches ground-truth stations (buoys, ships, turbidity, argo) to ICESat-2 ATL03
granules and downloads matching data.

Usage:
    python 02_match_stations2granules.py --source buoys
    python 02_match_stations2granules.py --source ships --start-index 0 --end-index 100
    python 02_match_stations2granules.py --source turbidity --dry-run
    python 02_match_stations2granules.py --source argo --argo-csv data/Dataset_Kd_Paper_2018.csv

Dependencies: icepyx>=2.0.0, earthaccess, erddapy, pandas, numpy, tqdm
"""

import argparse
import json
import logging
import os
import sys
import time
import glob
from datetime import datetime, timedelta, timezone

import earthaccess
import numpy as np
import pandas as pd
from erddapy import ERDDAP
from tqdm import tqdm

import icepyx as ipx


# ---------------------------------------------------------------------------
# Source configurations
# ---------------------------------------------------------------------------
SOURCE_CONFIGS = {
    "buoys": {
        "search_hrs": 6,
        "search_km": 4,
        "pattern": "A",            # CMR-first
        "filter": lambda df: df[
            (df.buoy == True)
            & ((df.phot == True) | (df.photo == True) | (df.radiation == True))
        ],
        "description": "Buoy PAR / photosynthetic radiation stations",
    },
    "ships": {
        "search_hrs": 6,
        "search_km": 4,
        "pattern": "B",            # CMR-first (was ERDDAP-first)
        "filter": lambda df: df[
            (df.radiation == True)
            & (df.buoy == False)
        ],
        "description": "Ship radiation casts (all servers)",
    },
    "turbidity": {
        "search_hrs": 6,
        "search_km": 4,
        "pattern": "A",            # CMR-first
        "filter": lambda df: df[
            (df.buoy == True)
            & ((df.turbid == True) | (df.ssc == True) | (df.turb == True))
        ],
        "description": "Turbidity / SSC buoy stations",
    },
    "argo": {
        "search_hrs": 12,
        "search_km": 4,
        "pattern": "C",            # CSV-direct
        "filter": None,            # handled separately
        "description": "BGC-Argo Kd floats (post 2018-10-01)",
    },
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def setup_logging(log_dir: str, source: str) -> logging.Logger:
    """Configure file + console logging and return the logger."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"02_match_{source}_{ts}.log")

    logger = logging.getLogger(f"match_{source}")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_file)
    return logger


def buoy_bound_box(lat: float, lon: float, buffer_km: float):
    """
    Create a bounding box around a point.

    Returns
    -------
    poly_ipx : list of (lon, lat) tuples  — for icepyx.Query
    bbox     : tuple (W, S, E, N)         — for earthaccess.search_data
    """
    lat_buff = buffer_km / 111.0
    lon_buff = buffer_km / (111.0 * np.cos(np.radians(lat)))

    minx = lon - lon_buff
    miny = lat - lat_buff
    maxx = lon + lon_buff
    maxy = lat + lat_buff

    poly_ipx = [
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny),
    ]
    bbox = (minx, miny, maxx, maxy)
    return poly_ipx, bbox


def fname2datetime(fname: str) -> datetime:
    """Extract UTC datetime from an ATL03 granule filename (e.g. ATL03_20190101...)."""
    y = int(fname[6:10])
    m = int(fname[10:12])
    d = int(fname[12:14])
    H = int(fname[14:16])
    M = int(fname[16:18])
    S = int(fname[18:20])
    return datetime(y, m, d, H, M, S, tzinfo=timezone.utc)


def fname2rgt_cycle_segment(fname: str):
    """
    Extract RGT, cycle, and segment from an ATL03 filename.

    ATL03 filename format: ATL03_YYYYMMDDHHMMSS_TTTTCCSS_VVV_RR.h5
      TTTT = 4-digit Reference Ground Track
      CC   = 2-digit cycle number
      SS   = 2-digit segment number
    """
    # Characters after second underscore: TTTTCCSS
    track_str = fname[21:29]
    try:
        rgt = int(track_str[0:4])
        cycle = int(track_str[4:6])
        segment = int(track_str[6:8])
        return rgt, cycle, segment
    except (ValueError, IndexError):
        return None, None, None


# ---------------------------------------------------------------------------
# Temporal / ERDDAP helpers
# ---------------------------------------------------------------------------
def _parse_temporal_range(t_start_str, t_end_str, pad_hrs):
    """
    Parse station time_coverage_start / time_coverage_end strings into a
    (datetime, datetime) tuple padded by *pad_hrs* on each side.

    Handles ISO-8601 strings (with or without trailing 'Z') and NaN / None
    end times (falls back to UTC now).
    """
    def _to_dt(s):
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return None
        s = str(s).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        elif "+" not in s and s[-1].isdigit():
            s += "+00:00"
        return datetime.fromisoformat(s)

    dt_start = _to_dt(t_start_str)
    if dt_start is None:
        return None  # can't build a range without a start

    dt_end = _to_dt(t_end_str)
    if dt_end is None:
        dt_end = datetime.now(timezone.utc)

    return (dt_start - timedelta(hours=pad_hrs),
            dt_end + timedelta(hours=pad_hrs))


def _parse_erddap_times(erddap_df):
    """
    Find the time column in an ERDDAP DataFrame (name contains 'time' and
    '(UTC)'), parse it to timezone-aware datetime, and return a Series.
    """
    time_col = None
    for col in erddap_df.columns:
        if "time" in col.lower() and "utc" in col.lower():
            time_col = col
            break
    if time_col is None:
        # fallback: first column with 'time' in the name
        for col in erddap_df.columns:
            if "time" in col.lower():
                time_col = col
                break
    if time_col is None:
        return pd.Series(dtype="datetime64[ns, UTC]")

    return pd.to_datetime(erddap_df[time_col], utc=True)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def search_cmr_granules(bbox, version: str = "007", temporal=None, logger=None):
    """
    Search CMR for ATL03 granules that intersect *bbox*.

    Parameters
    ----------
    temporal : tuple of (datetime, datetime), optional
        If provided, restricts the CMR search to this time window.

    Returns a list of earthaccess DataGranule objects.
    """
    W, S, E, N = bbox
    kwargs = dict(short_name="ATL03", version=version, bounding_box=(W, S, E, N))
    if temporal:
        kwargs["temporal"] = temporal
    results = earthaccess.search_data(**kwargs)
    if logger:
        logger.debug("CMR returned %d granules for bbox %s", len(results), bbox)
    return results


def granule_filename(granule) -> str:
    """Extract the canonical ATL03 filename from an earthaccess DataGranule."""
    # earthaccess DataGranule stores filename in data_links or native_id
    try:
        links = granule.data_links()
        if links:
            return os.path.basename(links[0])
    except Exception:
        pass
    # fallback: native_id
    try:
        return granule["meta"]["native-id"]
    except Exception:
        return str(granule)


def download_atl03(spatial_extent, date_range, output_dir: str,
                   version: str = "007", max_retries: int = 3, logger=None):
    """
    Download ATL03 granules via icepyx for the given spatial extent and
    date range.  Returns list of downloaded file paths.

    Skips download if matching .h5 files already exist in output_dir.
    """
    short_name = "ATL03"
    region = ipx.Query(short_name, spatial_extent, date_range, version=version)

    # Check available granules first
    try:
        region.avail_granules()
    except Exception as exc:
        if logger:
            logger.warning("No granules available: %s", exc)
        return []

    for attempt in range(1, max_retries + 1):
        try:
            region.download_granules(output_dir)
            break
        except Exception as exc:
            if logger:
                logger.warning("Download attempt %d/%d failed: %s",
                               attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(30)
            else:
                if logger:
                    logger.error("Download failed after %d attempts", max_retries)
                return []

    # Return whatever .h5 files now exist in output_dir
    return glob.glob(os.path.join(output_dir, "processed_ATL03_*.h5"))


def download_erddap_data(server_url: str, dataset_id: str, constraints: dict,
                         max_retries: int = 3, logger=None):
    """
    Download tabular data from an ERDDAP server.

    Returns a pandas DataFrame or None on failure.
    """
    e = ERDDAP(server=server_url, protocol="tabledap", response="csv")
    e.dataset_id = dataset_id
    e.constraints = constraints

    for attempt in range(1, max_retries + 1):
        try:
            df = e.to_pandas()
            return df
        except Exception as exc:
            if logger:
                logger.debug("ERDDAP attempt %d/%d for %s: %s",
                             attempt, max_retries, dataset_id, exc)
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # exponential backoff: 2, 4, 8 …
    return None


def load_station_dataframe(pkl_path: str, source: str, logger=None):
    """Load the step-01 station pickle and apply the source filter.

    Step 01 tags datasets with ``photo`` and ``turbid``; the source filters also
    reference ``phot`` / ``turb``, which are the same tags under a second name
    (they were added by hand in the original run). They are derived here so
    step 01 output can be used directly.

    The same dataset ID is often served by several mirror ERDDAP servers
    (e.g. sensors.ioos.us and cencoos.org). Only the first server listed is
    kept, so each dataset is matched once; otherwise every pass is written to
    the manifest once per mirror.
    """
    # The pickle was written with numpy 2; the CSV twin from step 01 loads anywhere.
    FF = pd.read_csv(pkl_path) if pkl_path.endswith(".csv") else pd.read_pickle(pkl_path)
    if "phot" not in FF.columns and "photo" in FF.columns:
        FF["phot"] = FF["photo"]
    if "turb" not in FF.columns and "turbid" in FF.columns:
        FF["turb"] = FF["turbid"]
    cfg = SOURCE_CONFIGS[source]
    FF = cfg["filter"](FF).copy()
    n_all = len(FF)
    FF = FF.drop_duplicates(subset="sites", keep="first")
    FF.reset_index(drop=True, inplace=True)
    if logger and len(FF) < n_all:
        logger.info("Dropped %d mirror copies of the same dataset ID (%d -> %d)",
                    n_all - len(FF), n_all, len(FF))

    for col in ("geospatial_lat_min", "geospatial_lon_min",
                "geospatial_lat_max", "geospatial_lon_max"):
        FF[col] = pd.to_numeric(FF[col])
    FF["photon_data"] = False

    if logger:
        logger.info("Loaded %d stations for source=%s from %s",
                     len(FF), source, pkl_path)
    return FF


def load_argo_dataframe(csv_path: str, logger=None):
    """Load Argo Kd CSV (Zenodo), convert Matlab datenum, filter post-IS2."""
    df = pd.read_csv(csv_path)
    df["dt_float"] = pd.to_datetime(
        df["dt_float"] - 719529, unit="d", utc=True
    ).round("s")
    df = df[df["dt_float"] > datetime(2018, 10, 1, tzinfo=timezone.utc)]
    df.reset_index(drop=True, inplace=True)
    df["check_sum"] = False
    if logger:
        logger.info("Loaded %d Argo rows (post 2018-10-01) from %s",
                     len(df), csv_path)
    return df


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------
def load_progress(progress_path: str) -> dict:
    """Load progress JSON; returns empty dict if file doesn't exist."""
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            return json.load(f)
    return {"last_completed_index": -1, "matches": 0, "errors": 0}


def save_progress(progress_path: str, progress: dict):
    """Atomically write progress JSON."""
    tmp = progress_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, progress_path)


# ---------------------------------------------------------------------------
# Pattern A — CMR-first (buoys, turbidity)
# ---------------------------------------------------------------------------
def match_pattern_a(df, cfg, output_dir, version, dry_run,
                    start_idx, end_idx, progress, progress_path,
                    source, logger, manifest=None):
    """
    For each station:
      1. Build bounding box → search CMR for all ATL03 granules
      2. For each granule time → check ERDDAP has ground-truth within ±search_hrs
      3. If match → save ERDDAP CSV; download ATL03 (or record to manifest)
    """
    search_hrs = cfg["search_hrs"]
    search_km = cfg["search_km"]

    for jj in tqdm(range(start_idx, end_idx), desc=f"Pattern A ({source})"):
        if jj <= progress.get("last_completed_index", -1):
            continue

        station_id = df["sites"][jj]
        logger.info("Station %d/%d  %s", jj, end_idx, station_id)

        try:
            lat = df["geospatial_lat_min"][jj]
            lon = df["geospatial_lon_min"][jj]
            poly_ipx, bbox = buoy_bound_box(lat, lon, search_km)

            # --- Build temporal range from station metadata ---
            t_cov_start = df["time_coverage_start"][jj]
            t_cov_end = df["time_coverage_end"][jj]
            temporal = _parse_temporal_range(t_cov_start, t_cov_end, search_hrs)

            # --- Step 1: CMR search (with temporal filter when available) ---
            granules = search_cmr_granules(
                bbox, version=version, temporal=temporal, logger=logger,
            )
            if not granules:
                logger.info("  No CMR granules for station %s", station_id)
                progress["last_completed_index"] = jj
                save_progress(progress_path, progress)
                continue

            # Extract granule filenames and datetimes (paired)
            granule_fnames = [granule_filename(g) for g in granules]
            icesat_pairs = []  # list of (filename, datetime)
            for fn in granule_fnames:
                try:
                    icesat_pairs.append((fn, fname2datetime(fn)))
                except Exception:
                    logger.debug("  Could not parse datetime from %s", fn)

            if not icesat_pairs:
                logger.info("  Could not parse times from CMR results")
                progress["last_completed_index"] = jj
                save_progress(progress_path, progress)
                continue

            icesat_times = [pair[1] for pair in icesat_pairs]

            # --- Step 2: One ERDDAP call per station ---
            # Use the span of actual CMR granule times (± search_hrs) instead
            # of the full station lifetime to avoid ERDDAP OUT_OF_MEMORY on
            # long-running, high-frequency stations.
            server_url = df["url"][jj]
            erddap_t_start = (min(icesat_times) - timedelta(hours=search_hrs)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            erddap_t_end = (max(icesat_times) + timedelta(hours=search_hrs)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

            erddap_df = download_erddap_data(
                server_url, station_id,
                {"time>=": erddap_t_start, "time<=": erddap_t_end},
                logger=logger,
            )
            if erddap_df is None or erddap_df.empty:
                logger.info("  No ERDDAP data for station %s — skipping",
                            station_id)
                progress["last_completed_index"] = jj
                save_progress(progress_path, progress)
                continue

            # Parse ERDDAP time column once for local filtering
            erddap_times = _parse_erddap_times(erddap_df)

            # --- Step 3: For each granule time, filter locally then download ---
            for granule_fname, t in icesat_pairs:
                t_start = t - timedelta(hours=search_hrs)
                t_end = t + timedelta(hours=search_hrs)

                # Local pandas filter instead of per-granule ERDDAP call
                if not erddap_times.empty:
                    mask = (erddap_times >= t_start) & (erddap_times <= t_end)
                    matched_rows = erddap_df[mask]
                    if matched_rows.empty:
                        continue
                else:
                    # No parseable time column — fall back to accepting match
                    matched_rows = erddap_df

                logger.info("  ERDDAP match at %s for station %s (%d rows)",
                            t, station_id, len(matched_rows))
                progress["matches"] = progress.get("matches", 0) + 1

                # Save ERDDAP ground-truth CSV (needed by Step 03)
                csv_name = f"data_{station_id}_{t_start}.csv"
                csv_path = os.path.join(output_dir, csv_name)
                matched_rows.to_csv(csv_path, index=False)
                logger.info("  Saved ERDDAP CSV: %s", csv_path)

                # Record match to manifest for batch download
                rgt, cycle, segment = fname2rgt_cycle_segment(granule_fname)
                if manifest is not None:
                    manifest.append({
                        "station_id": station_id,
                        "source": source,
                        "lat": lat,
                        "lon": lon,
                        "granule_filename": granule_fname,
                        "rgt": rgt,
                        "cycle": cycle,
                        "segment": segment,
                        "t0": t_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "t1": t_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "poly_W": bbox[0],
                        "poly_S": bbox[1],
                        "poly_E": bbox[2],
                        "poly_N": bbox[3],
                        "erddap_csv": csv_path,
                        "erddap_rows": len(matched_rows),
                    })

                if dry_run:
                    logger.info("  [DRY RUN] would download ATL03")
                    continue

                # Download ATL03
                download_atl03(
                    poly_ipx, [t_start, t_end], output_dir,
                    version=version, logger=logger,
                )

        except Exception as exc:
            logger.error("  Error on station %d (%s): %s", jj, station_id, exc)
            progress["errors"] = progress.get("errors", 0) + 1

        progress["last_completed_index"] = jj
        save_progress(progress_path, progress)


# ---------------------------------------------------------------------------
# Pattern B — CMR-first with temporal filter (ships)
# ---------------------------------------------------------------------------
def match_pattern_b(df, cfg, output_dir, version, dry_run,
                    start_idx, end_idx, progress, progress_path,
                    source, logger, manifest=None):
    """
    For each station:
      1. Build spatial + temporal window from pickle metadata
      2. Search CMR for ATL03 granules in that window
      3. If CMR finds granules → download ERDDAP ground-truth CSV + ATL03 via icepyx
    """
    search_hrs = cfg["search_hrs"]
    search_km = cfg["search_km"]

    for jj in tqdm(range(start_idx, end_idx), desc=f"Pattern B ({source})"):
        if jj <= progress.get("last_completed_index", -1):
            continue

        station_id = df["sites"][jj]
        if jj % 100 == 0:
            logger.info("Station %d/%d  %s", jj, end_idx, station_id)

        try:
            # Step 1: build spatial + temporal search from pickle metadata
            lat = df["geospatial_lat_min"][jj]
            lon = df["geospatial_lon_min"][jj]
            poly_ipx, bbox = buoy_bound_box(lat, lon, search_km)

            t_cov_start = df["time_coverage_start"][jj]
            t_cov_end = df["time_coverage_end"][jj]
            temporal = _parse_temporal_range(t_cov_start, t_cov_end, search_hrs)

            t = datetime.fromisoformat(
                str(t_cov_start).replace("Z", "+00:00")
            )
            t_start = t - timedelta(hours=search_hrs)
            t_end = t + timedelta(hours=search_hrs)

            # Step 2: CMR search with temporal filter
            granules = search_cmr_granules(
                bbox, version=version, temporal=temporal, logger=logger,
            )
            if not granules:
                logger.debug("  No CMR granules for station %s", station_id)
                progress["last_completed_index"] = jj
                save_progress(progress_path, progress)
                continue

            # Extract first granule filename for RGT info
            g_fname = granule_filename(granules[0])
            rgt, cycle, segment = fname2rgt_cycle_segment(g_fname)

            # Step 3: download ERDDAP ground-truth data (needed by Step 03)
            server_url = df["url"][jj]
            t_start_str = t_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            t_end_str = t_end.strftime("%Y-%m-%dT%H:%M:%SZ")

            erddap_df = download_erddap_data(
                server_url, station_id,
                {"time>=": t_start_str, "time<=": t_end_str},
                logger=logger,
            )

            progress["matches"] = progress.get("matches", 0) + 1

            # Save ERDDAP ground-truth CSV (needed by Step 03)
            csv_path = ""
            erddap_rows = 0
            if erddap_df is not None and not erddap_df.empty:
                csv_name = f"data_{station_id}_{t_start}.csv"
                csv_path = os.path.join(output_dir, csv_name)
                erddap_df.to_csv(csv_path, index=False)
                erddap_rows = len(erddap_df)
                logger.info("  Saved ERDDAP CSV: %s", csv_path)

            # Record match to manifest for batch download
            if manifest is not None:
                manifest.append({
                    "station_id": station_id,
                    "source": source,
                    "lat": lat,
                    "lon": lon,
                    "granule_filename": g_fname,
                    "rgt": rgt,
                    "cycle": cycle,
                    "segment": segment,
                    "t0": t_start_str,
                    "t1": t_end_str,
                    "poly_W": bbox[0],
                    "poly_S": bbox[1],
                    "poly_E": bbox[2],
                    "poly_N": bbox[3],
                    "erddap_csv": csv_path,
                    "erddap_rows": erddap_rows,
                })

            if dry_run:
                logger.info("  [DRY RUN] CMR found %d granules for station %s at %s",
                            len(granules), station_id, t)
                progress["last_completed_index"] = jj
                save_progress(progress_path, progress)
                continue

            # Step 4: download ATL03
            download_atl03(
                poly_ipx, [t_start, t_end], output_dir,
                version=version, logger=logger,
            )

        except Exception as exc:
            logger.error("  Error on station %d (%s): %s", jj, station_id, exc)
            progress["errors"] = progress.get("errors", 0) + 1

        progress["last_completed_index"] = jj
        save_progress(progress_path, progress)


# ---------------------------------------------------------------------------
# Pattern C — CSV-direct (argo)
# ---------------------------------------------------------------------------
def match_pattern_c(df, cfg, output_dir, version, dry_run,
                    start_idx, end_idx, progress, progress_path,
                    source, logger, manifest=None):
    """
    For each row in the Argo CSV:
      1. Use lat/lon/time directly
      2. Search earthaccess for ATL03 within ±search_hrs, ±search_km
      3. Download if found (unless dry_run); mark matched rows
    """
    search_hrs = cfg["search_hrs"]
    search_km = cfg["search_km"]

    for jj in tqdm(range(start_idx, end_idx), desc=f"Pattern C ({source})"):
        if jj <= progress.get("last_completed_index", -1):
            continue

        if jj % 100 == 0:
            logger.info("Row %d/%d", jj, end_idx)

        try:
            lat = df["lat_float"][jj]
            lon = df["lon_float"][jj]
            poly_ipx, bbox = buoy_bound_box(lat, lon, search_km)

            t_center = df["dt_float"][jj]
            t_start = t_center - timedelta(hours=search_hrs)
            t_end = t_center + timedelta(hours=search_hrs)

            # Search CMR using earthaccess (works in dry_run too)
            granules = search_cmr_granules(bbox, version, temporal=(t_start, t_end), logger=logger)

            if not granules:
                logger.debug("  No granules for row %d at %.2f,%.2f", jj, lat, lon)
                progress["last_completed_index"] = jj
                save_progress(progress_path, progress)
                continue

            # Found matches!
            logger.info("  Found %d granule(s) for row %d at %.2f,%.2f  %s",
                        len(granules), jj, lat, lon, t_center)
            df.at[jj, "check_sum"] = True
            progress["matches"] = progress.get("matches", 0) + 1

            # Add to manifest
            for g in granules:
                fname = granule_filename(g)
                rgt, cycle, segment = fname2rgt_cycle_segment(fname)
                if manifest is not None:
                    manifest.append({
                        "station_id": f"argo_row_{jj}",
                        "source": source,
                        "lat": lat,
                        "lon": lon,
                        "granule_filename": fname,
                        "rgt": rgt,
                        "cycle": cycle,
                        "segment": segment,
                        "t0": t_start.isoformat(),
                        "t1": t_end.isoformat(),
                        "poly_W": bbox[0],
                        "poly_S": bbox[1],
                        "poly_E": bbox[2],
                        "poly_N": bbox[3],
                        "erddap_csv": "",
                        "erddap_rows": 0,
                    })

            # Download ATL03 (unless dry_run)
            if not dry_run:
                download_atl03(
                    poly_ipx, [t_start, t_end], output_dir,
                    version, logger, dry_run=False
                )

        except Exception as exc:
            logger.error("  Error on row %d: %s", jj, exc)
            progress["errors"] = progress.get("errors", 0) + 1

        progress["last_completed_index"] = jj
        save_progress(progress_path, progress)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Match ground-truth stations to ICESat-2 ATL03 granules "
                    "and download matching data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_match_stations2granules.py --source buoys --dry-run
  python 02_match_stations2granules.py --source ships --start-index 0 --end-index 100
  python 02_match_stations2granules.py --source argo --argo-csv data/Dataset_Kd_Paper_2018.csv
""",
    )
    parser.add_argument(
        "--source", required=True,
        choices=["buoys", "ships", "turbidity", "argo"],
        help="Which data source to process.",
    )
    parser.add_argument(
        "--pkl-path", default="results/Saved_all_relevant_stations.pkl",
        help="Step-01 station table, .pkl or .csv (buoys/ships/turbidity).",
    )
    parser.add_argument(
        "--argo-csv", default="data/Dataset_Kd_Paper_2018.csv",
        help="Path to Argo Kd CSV (argo source only).",
    )
    parser.add_argument(
        "--output-dir", default="data/",
        help="Directory to save downloaded ATL03 .h5 files and ERDDAP CSVs.",
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="Start index in the station/row list (inclusive).",
    )
    parser.add_argument(
        "--end-index", type=int, default=-1,
        help="End index in the station/row list (exclusive). "
             "-1 means process all.",
    )
    parser.add_argument(
        "--log-dir", default="logs/",
        help="Directory for log files and progress JSON.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Search and match but do not download ATL03 files.",
    )
    parser.add_argument(
        "--version", default="007",
        help="ATL03 version string (default: 007).",
    )
    args = parser.parse_args()

    # --- Setup ---
    source = args.source
    cfg = SOURCE_CONFIGS[source]
    logger = setup_logging(args.log_dir, source)
    logger.info("Source: %s — %s", source, cfg["description"])
    logger.info("ATL03 version: %s", args.version)
    logger.info("Search window: ±%d hrs, ±%d km",
                cfg["search_hrs"], cfg["search_km"])

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Authenticate earthdata (needed by both earthaccess and icepyx) ---
    try:
        earthaccess.login(strategy="environment")
        logger.info("Authenticated via earthaccess (environment)")
    except Exception:
        try:
            earthaccess.login(strategy="netrc")
            logger.info("Authenticated via earthaccess (netrc)")
        except Exception as exc:
            logger.error("earthdata authentication failed: %s", exc)
            logger.error("Set EARTHDATA_USERNAME / EARTHDATA_PASSWORD env vars "
                         "or configure ~/.netrc")
            sys.exit(1)

    # --- Load data ---
    if source == "argo":
        df = load_argo_dataframe(args.argo_csv, logger=logger)
    else:
        if not os.path.exists(args.pkl_path):
            logger.error("Pickle file not found: %s", args.pkl_path)
            sys.exit(1)
        df = load_station_dataframe(args.pkl_path, source, logger=logger)

    # --- Index range ---
    n = len(df)
    start_idx = args.start_index
    end_idx = n if args.end_index == -1 else min(args.end_index, n)
    logger.info("Processing indices [%d, %d) of %d total rows",
                start_idx, end_idx, n)

    # --- Progress file ---
    progress_path = os.path.join(args.log_dir, f"02_match_{source}_progress.json")
    progress = load_progress(progress_path)
    if progress.get("last_completed_index", -1) >= start_idx:
        logger.info("Resuming from index %d (progress file: %s)",
                     progress["last_completed_index"] + 1, progress_path)

    # --- Manifest: collects match parameters for batch download ---
    manifest = []

    # --- Dispatch to pattern ---
    if cfg["pattern"] == "A":
        match_pattern_a(
            df, cfg, args.output_dir, args.version, args.dry_run,
            start_idx, end_idx, progress, progress_path, source, logger,
            manifest=manifest,
        )
    elif cfg["pattern"] == "B":
        match_pattern_b(
            df, cfg, args.output_dir, args.version, args.dry_run,
            start_idx, end_idx, progress, progress_path, source, logger,
            manifest=manifest,
        )
    elif cfg["pattern"] == "C":
        df = match_pattern_c(
            df, cfg, args.output_dir, args.version, args.dry_run,
            start_idx, end_idx, progress, progress_path, source, logger,
            manifest=manifest,
        )
        # Save matched rows
        matched = df[df["check_sum"] == True]
        out_csv = "results/02_match_argo_profiles.csv"
        matched.to_csv(out_csv, index=False)
        logger.info("Saved %d matched Argo rows to %s", len(matched), out_csv)

    # --- Save manifest CSV ---
    if manifest:
        # A partial index range (array task) gets its own file, so tasks never
        # overwrite each other; concatenate the pieces afterwards.
        suffix = "" if (start_idx == 0 and end_idx == n) else f"_{start_idx}_{end_idx}"
        manifest_path = os.path.join(
            "results", f"02_match_{source}_manifest{suffix}.csv"
        )
        manifest_df = pd.DataFrame(manifest)
        n_rows = len(manifest_df)
        manifest_df = manifest_df.drop_duplicates(
            subset=["station_id", "granule_filename"], keep="first")
        if len(manifest_df) < n_rows:
            logger.info("Dropped %d duplicate (station, granule) rows",
                        n_rows - len(manifest_df))
        manifest_df.to_csv(manifest_path, index=False)
        logger.info("Saved manifest with %d matches to %s",
                     len(manifest_df), manifest_path)

    # --- Summary ---
    logger.info("Done.  matches=%d  errors=%d",
                progress.get("matches", 0), progress.get("errors", 0))


if __name__ == "__main__":
    main()
