#!/usr/bin/env python3
"""
Step 03b: Calculate Kd from ship PAR profile data.

Reads the ships manifest (Step 02 output) to identify which ERDDAP CSVs
contain ship CTD/PAR profiles, then calculates Kd using log-linear regression:
    ln(PAR) = -Kd * depth + intercept

Usage:
    python 03b_calc_kd_ships.py \
        --manifest results/02_match_ships_manifest.csv \
        --input-dir data/ \
        --output results/ship_kd.csv \
        --max-depth 20

Output columns: time, lat, lon, kd, rval, stderr, n_points, source_file
"""

import argparse
import os
import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def find_column(df, candidates):
    """
    Find a column by trying multiple possible names.

    Three passes (in order of strictness):
      1. Exact match
      2. Case-insensitive exact match
      3. Case-insensitive startswith match — catches ERDDAP columns that
         append units, e.g. 'photosynthetically_available_radiation (umol m-2 s-1)'
         is found by candidate 'photosynthetically_available_radiation'
    """
    lower_cols = {c.lower(): c for c in df.columns}

    for col in candidates:
        # Pass 1: exact
        if col in df.columns:
            return col
        # Pass 2: case-insensitive exact
        if col.lower() in lower_cols:
            return lower_cols[col.lower()]

    for col in candidates:
        # Pass 3: startswith (handles unit suffixes added by ERDDAP)
        col_lower = col.lower()
        for df_col in df.columns:
            if df_col.lower().startswith(col_lower):
                return df_col

    return None


def calc_kd(z, par):
    """
    Calculate Kd from depth and PAR arrays via log-linear regression.

    Returns (kd, rval, stderr) or (nan, nan, nan) if fit fails.
    """
    z   = np.abs(np.array(z,   dtype=float))
    par = np.array(par, dtype=float)

    valid = (par > 0) & ~np.isnan(par) & ~np.isnan(z)
    z, par = z[valid], par[valid]

    if len(z) < 3:
        return np.nan, np.nan, np.nan

    try:
        fit = stats.linregress(x=z, y=np.log(par))
        return np.abs(fit.slope), fit.rvalue, fit.stderr
    except Exception:
        return np.nan, np.nan, np.nan


def identify_casts(df, time_col, gap_minutes=10):
    """
    Split rows into separate casts by detecting time gaps > gap_minutes.
    Returns a Series of integer cast IDs aligned to df.index.
    """
    times = pd.to_datetime(df[time_col], errors='coerce')
    if times.isna().all():
        return pd.Series(0, index=df.index)

    dt = times.diff()
    cast, casts = 0, []
    for gap in dt:
        if pd.notna(gap) and gap > timedelta(minutes=gap_minutes):
            cast += 1
        casts.append(cast)
    return pd.Series(casts, index=df.index)


# ---------------------------------------------------------------------------
# File processor
# ---------------------------------------------------------------------------

def process_par_file(filepath, max_depth=20):
    """
    Process one ERDDAP CSV and return a list of Kd result dicts (one per cast).
    Returns [] if the file has no usable PAR profile data.
    """
    try:
        df = pd.read_csv(filepath, header=0)
    except Exception as e:
        print(f"  Error reading {os.path.basename(filepath)}: {e}")
        return []

    time_col  = find_column(df, ['time (UTC)', 'time', 'datetime', 'date_time'])
    lat_col   = find_column(df, ['latitude (degrees_north)', 'latitude', 'lat'])
    lon_col   = find_column(df, ['longitude (degrees_east)', 'longitude', 'lon'])
    depth_col = find_column(df, ['depth (m)', 'z (m)', 'depth', 'z', 'depth_reading (m)'])
    par_col   = find_column(df, [
        # Short names
        'par', 'PAR', 'Par',
        # Common ERDDAP display names
        'PAR_Irradiance', 'par_irradiance',
        'downwelling_par', 'downwelling_PAR',
        'scalar_par', 'scalar_PAR',
        # CF standard names / OOI PARAD instrument names (PARADA, PARADJ, PARADK)
        'downwelling_photosynthetic_photon_radiance_in_sea_water',
        'downwelling_photosynthetic_photon_flux_in_sea_water',
        'estimated_downwelling_photosynthetic_photon_flux_in_sea_water',
        'estimated_downwelling_photosynthetic_photon_flux',
        'photosynthetically_available_radiation',
        'photosynthetically_active_radiation',
        'photosynthetic_photon_flux_density',
        # With units in the column name (ERDDAP style)
        'par (umol m-2 s-1)', 'par (uE m-2 s-1)',
        'par (microeinsteins m-2 s-1)', 'PAR (umol m-2 s-1)',
        'par (W m-2)', 'PAR (W m-2)',
        'irradiance (W m-2)', 'irradiance',
    ])

    missing = [n for n, c in [
        ('time', time_col), ('lat', lat_col), ('lon', lon_col),
        ('depth', depth_col), ('PAR', par_col),
    ] if not c]

    if missing:
        print(f"  Skipping {os.path.basename(filepath)}: missing columns {missing}")
        print(f"    Available columns: {list(df.columns)}")
        return []

    df['_z']   = pd.to_numeric(df[depth_col], errors='coerce').abs()
    df['_par'] = pd.to_numeric(df[par_col],   errors='coerce')
    df['_lat'] = pd.to_numeric(df[lat_col],   errors='coerce')
    df['_lon'] = pd.to_numeric(df[lon_col],   errors='coerce')
    df = df[df['_z'] < max_depth].copy()

    if len(df) < 3:
        print(f"  Skipping {os.path.basename(filepath)}: too few points after depth filter")
        return []

    df['_cast'] = identify_casts(df, time_col)

    results = []
    for cast_id in df['_cast'].unique():
        sub = df[df['_cast'] == cast_id]
        if len(sub) < 3:
            continue
        kd, rval, stderr = calc_kd(sub['_z'].values, sub['_par'].values)
        if np.isnan(kd):
            continue
        results.append({
            'time':        sub[time_col].iloc[0],
            'lat':         sub['_lat'].iloc[0],
            'lon':         sub['_lon'].iloc[0],
            'kd':          kd,
            'rval':        rval,
            'stderr':      stderr,
            'n_points':    len(sub),
            'source_file': os.path.basename(filepath),
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 03b: Calculate Kd from ship PAR profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Method:
  Log-linear regression: ln(PAR) = -Kd * depth + intercept
  Multiple casts within one file are split by time gaps > 10 min.
  Only depths shallower than --max-depth are used.
""",
    )
    parser.add_argument(
        '--manifest', default='results/02_match_ships_manifest.csv',
        help='Ships manifest from Step 02 (default: results/02_match_ships_manifest.csv)',
    )
    parser.add_argument(
        '--input-dir', default='data/',
        help='Directory containing data_*.csv files (default: data/)',
    )
    parser.add_argument(
        '--output', default='results/ship_kd.csv',
        help='Output CSV path (default: results/ship_kd.csv)',
    )
    parser.add_argument(
        '--max-depth', type=float, default=20,
        help='Maximum depth (m) to include in Kd fit (default: 20)',
    )
    args = parser.parse_args()

    # --- Load manifest and get file list ---
    if not os.path.exists(args.manifest):
        print(f"ERROR: manifest not found: {args.manifest}")
        return

    manifest = pd.read_csv(args.manifest)
    if 'erddap_csv' not in manifest.columns:
        print("ERROR: manifest has no 'erddap_csv' column")
        return

    # erddap_csv may be stored as full path; normalise to basename
    csv_basenames = (
        manifest['erddap_csv']
        .dropna()
        .apply(os.path.basename)
        .unique()
    )
    csv_basenames = [f for f in csv_basenames if f]
    print(f"Manifest: {args.manifest}")
    print(f"  {len(manifest)} manifest rows → {len(csv_basenames)} unique ERDDAP CSV file(s)")
    print(f"Max depth: {args.max_depth} m\n")

    # --- Process each file ---
    all_results = []
    files_found = 0
    files_with_kd = 0

    for fname in sorted(csv_basenames):
        fpath = os.path.join(args.input_dir, fname)
        if not os.path.exists(fpath):
            print(f"  NOT FOUND: {fpath} — skipping")
            continue
        files_found += 1
        rows = process_par_file(fpath, max_depth=args.max_depth)
        if rows:
            all_results.extend(rows)
            files_with_kd += 1
            print(f"  {fname}: {len(rows)} Kd value(s)")
        else:
            print(f"  {fname}: no Kd (no PAR data or poor fit)")

    # --- Save ---
    if not all_results:
        print("\nNo Kd values calculated from any ship file.")
        print("Check that ship CSVs contain PAR profile data (depth + PAR columns).")
        return

    df_out = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df_out.to_csv(args.output, index=False)

    print(f"\n=== Summary ===")
    print(f"Files found         : {files_found} / {len(csv_basenames)}")
    print(f"Files with valid Kd : {files_with_kd} / {files_found}")
    print(f"Total Kd values     : {len(df_out)}")
    print(f"Output              : {args.output}")
    print(f"\nKd stats:")
    print(f"  Min    = {df_out['kd'].min():.4f}")
    print(f"  Max    = {df_out['kd'].max():.4f}")
    print(f"  Mean   = {df_out['kd'].mean():.4f}")
    print(f"  Median = {df_out['kd'].median():.4f}")
    good = (df_out['rval'].abs() > 0.8).sum()
    print(f"\nGood fits (|r| > 0.8): {good}/{len(df_out)} ({100*good/len(df_out):.1f}%)")


if __name__ == '__main__':
    main()
