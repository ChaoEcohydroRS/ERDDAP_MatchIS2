#!/usr/bin/env python3
"""
Step 03c: Compile turbidity data from ERDDAP CSVs.

Reads the turbidity manifest (Step 02 output) to identify which ERDDAP CSVs
contain turbidity measurements, extracts near-surface values (|depth| <= 5 m),
and writes a single compiled CSV.

Depth convention: ERDDAP datasets report either a positive-down ``depth`` or a
positive-up ``z`` (negative below the surface). The near-surface test therefore
uses the absolute value. (The original run used ``z < max_depth``, which kept
every reading of a negative-z dataset down to 33 m; fixed here.) Files with no
depth column are assumed to be at the surface (z = 0); note that some servers
also report a placeholder z = 0 for subsurface instruments.

The output is rewritten after every file, so a killed run keeps its progress.

Usage:
    python 03c_compile_turbidity.py \
        --manifest results/02_match_turbidity_manifest.csv \
        --input-dir data/ \
        --output results/turbidity_compiled.csv \
        --max-depth 5

Output columns: time, lat, lon, z, turb, turb_units, source_file
"""

import argparse
import os
import re
import pandas as pd
import numpy as np


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
         append units, e.g. 'turbidity (1)' is found by candidate 'turbidity'
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


def extract_turbidity(filepath, max_depth=5.0):
    """
    Extract turbidity data from one ERDDAP CSV file.

    Returns a DataFrame with columns [time, lat, lon, z, turb, source_file],
    or None if the file cannot be used.
    """
    try:
        df_full = pd.read_csv(filepath, header=0)
    except Exception as e:
        print(f"  Error reading {os.path.basename(filepath)}: {e}")
        return None

    time_col  = find_column(df_full, ['time (UTC)', 'time', 'datetime', 'date_time'])
    lat_col   = find_column(df_full, ['latitude (degrees_north)', 'latitude', 'lat'])
    lon_col   = find_column(df_full, ['longitude (degrees_east)', 'longitude', 'lon'])
    depth_col = find_column(df_full, ['depth (m)', 'z (m)', 'depth_reading (m)', 'depth', 'z'])
    turb_col  = find_column(df_full, [
        'turbidity (ntus)', 'turbidity (NTU)', 'turbidity',
        'sea_water_turbidity (NTU)', 'sea_water_turbidity (RFU)',
        'sea_water_turbidity', 'turb', 'NTU (ntu)',
    ])

    missing = [n for n, c in [
        ('time', time_col), ('lat', lat_col), ('lon', lon_col),
        ('turbidity', turb_col),
    ] if not c]

    if missing:
        print(f"  Skipping {os.path.basename(filepath)}: missing columns {missing}")
        print(f"    Available: {list(df_full.columns)}")
        return None

    df = pd.DataFrame()
    df['time'] = df_full[time_col]
    df['lat']  = pd.to_numeric(df_full[lat_col],  errors='coerce')
    df['lon']  = pd.to_numeric(df_full[lon_col],  errors='coerce')
    df['turb'] = pd.to_numeric(df_full[turb_col], errors='coerce')

    if depth_col:
        df['z'] = pd.to_numeric(df_full[depth_col], errors='coerce')
    else:
        df['z'] = 0.0  # assume surface if no depth column

    # Units as given in the column header, e.g. 'turbidity (NTU)' -> 'NTU'.
    # Lets later steps drop fluorescence units (RFU), which are not turbidity.
    m = re.search(r'\(([^)]*)\)\s*$', turb_col)
    df['turb_units'] = m.group(1) if m else ''
    df['source_file'] = os.path.basename(filepath)

    # Keep only near-surface rows (|z| handles both depth conventions)
    df = df[df['z'].abs() <= max_depth]

    if len(df) == 0:
        return None

    return df[['time', 'lat', 'lon', 'z', 'turb', 'turb_units', 'source_file']]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 03c: Compile turbidity data from ERDDAP CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Only rows with |depth| <= --max-depth (default 5 m) are kept.
Files that lack a turbidity column are reported and skipped.
""",
    )
    parser.add_argument(
        '--manifest', default='results/02_match_turbidity_manifest.csv',
        help='Turbidity manifest from Step 02 (default: results/02_match_turbidity_manifest.csv)',
    )
    parser.add_argument(
        '--input-dir', default='data/',
        help='Directory containing data_*.csv files (default: data/)',
    )
    parser.add_argument(
        '--output', default='results/turbidity_compiled.csv',
        help='Output CSV path (default: results/turbidity_compiled.csv)',
    )
    parser.add_argument(
        '--max-depth', type=float, default=5.0,
        help='Maximum |depth| (m) to include (default: 5, the window used for Fig. 5b)',
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

    # --- Process each file (output appended per file, so partial runs are kept) ---
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)
    files_found = 0
    files_with_turb = 0
    n_written = 0

    for i, fname in enumerate(sorted(csv_basenames), 1):
        fpath = os.path.join(args.input_dir, fname)
        if not os.path.exists(fpath):
            print(f"  NOT FOUND: {fpath} — skipping", flush=True)
            continue
        files_found += 1
        df = extract_turbidity(fpath, max_depth=args.max_depth)
        if df is not None:
            df = df.dropna(subset=['lat', 'lon', 'turb'])
        if df is not None and len(df) > 0:
            df.to_csv(args.output, mode='a', header=(n_written == 0), index=False)
            n_written += len(df)
            files_with_turb += 1
            print(f"  [{i}/{len(csv_basenames)}] {fname}: {len(df)} row(s)", flush=True)
        else:
            print(f"  [{i}/{len(csv_basenames)}] {fname}: no usable turbidity data", flush=True)

    # --- Summary ---
    if n_written == 0:
        print("\nNo turbidity data found in any file.")
        return

    df_out = pd.read_csv(args.output)

    print(f"\n=== Summary ===")
    print(f"Files found              : {files_found} / {len(csv_basenames)}")
    print(f"Files with turbidity data: {files_with_turb} / {files_found}")
    print(f"Total observations       : {len(df_out)}")
    print(f"Output                   : {args.output}")
    print(f"\nData extent:")
    print(f"  Lat  : {df_out['lat'].min():.3f} to {df_out['lat'].max():.3f}")
    print(f"  Lon  : {df_out['lon'].min():.3f} to {df_out['lon'].max():.3f}")
    print(f"  Turb : {df_out['turb'].min():.4f} to {df_out['turb'].max():.4f}")


if __name__ == '__main__':
    main()
