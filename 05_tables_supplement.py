#!/usr/bin/env python3
"""Step 05: build the two supplementary tables from the same inputs as Fig. 5.

  results/tableS_kd_matchups.csv        Fig. 5a matchups grouped by platform
  results/tableS_turbidity_stations.csv Fig. 5b stations grouped by network

Counts come from the fig_is2_validation loaders, so the tables always match the
plotted points (same duplicate removal and exclusions).

    python 05_tables_supplement.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_is2_validation as F  # noqa: E402

RES = os.path.join(HERE, "results")

# station-id pattern -> (network, region); first match wins
NETWORKS = [
    (r"^pmel_co2_moorings_", "NOAA PMEL CO2 moorings", "Open ocean and continental shelves"),
    (r"^(edu_calpoly|edu_humboldt|l01-elkhorn|rtcco2buoy|san-luis-bay|tomales-bay|exploratorium)",
     "University and regional shore stations", "California coast"),
    (r"^ooi-cp", "OOI Coastal Pioneer Array", "Mid-Atlantic Bight shelf"),
    (r"^(wqb|WQB)[-_]", "PacIOOS water-quality buoys", "Hawaii"),
    (r"^gov-ndbc-", "NOAA NDBC buoys",
     "Mobile Bay (Gulf of Mexico), Chesapeake Bay, Great South Bay (Long Island)"),
    (r"^(biscayne|clbf1)", "Florida coastal stations", "Tampa Bay and Biscayne Bay"),
    (r"^LOBO_", "UMaine LOBO buoys", "Gulf of Maine"),
    (r"^[A-Z]\d\d_optics", "NERACOOS buoys", "Gulf of Maine"),
    (r"^nerrs_kac", "NERRS Kachemak Bay", "Cook Inlet, Alaska"),
]


def network_of(station_id):
    for pat, net, reg in NETWORKS:
        if re.search(pat, station_id):
            return net, reg
    return "Other", ""


def argo_region(lat, lon):
    """Coarse ocean-basin boxes, adequate for the matched float positions."""
    if 30 < lat < 46 and -6 < lon < 37:
        return "Mediterranean Sea"
    if 53 < lat < 66 and 10 < lon < 31:
        return "Baltic Sea"
    if (lon < -70 and lat < 9) or lon > 120:
        return "Pacific Ocean"
    return "Atlantic Ocean"


def ratio(g):
    return float(np.median(g.insitu_kd / g.is2_kd))


def kd_table(res):
    d = F.load_kd_pairs(res)
    ar = pd.read_csv(os.path.join(res, "paired_is2_argo_kd.csv"))
    rows = []

    def add(platform, program, qty, sites, g):
        rows.append(dict(platform=platform, program_region=program, insitu_quantity=qty,
                         sites_or_platforms=sites, matchups=len(g),
                         median_ratio=round(ratio(g), 2)))

    g = d[d.series == "surface"]
    add("Moored surface-piercing profilers", "OOI Endurance Array; Oregon and Washington shelf",
        "Kd(PAR)", f"{g.station_id.nunique()} stations", g)
    g = d[d.series == "deep"]
    add("Moored offshore profilers",
        "OOI Regional Cabled Array; Oregon offshore, slope base and Axial Seamount",
        "Kd(PAR)", f"{g.station_id.nunique()} stations", g)
    g = d[d.series == "ship"]
    cruises = sorted({s.split("_")[0] for s in g.station_id})
    add("Ship casts on station", f"West Florida Shelf (cruises {' and '.join(cruises)})",
        "Kd(PAR)", f"{g.station_id.nunique()} stations", g)
    gl = d[d.series == "glider"]
    for region, label, program in [
            ("OOI Endurance (WA/OR)", "Gliders, Washington/Oregon",
             "OOI Endurance Array; Washington and Oregon shelf"),
            ("OOI Pioneer (Mid-Atlantic Bight)", "Gliders, Mid-Atlantic Bight",
             "OOI Coastal Pioneer Array; Mid-Atlantic Bight")]:
        g = gl[gl.region == region]
        deps = g.station_id.str.split("|").str[0].nunique()
        add(label, program, "Kd(PAR)", f"{deps} deployments", g)
    g = gl[gl.region.isna()]
    add("Gliders, Bermuda", "BIOS gliders; Bermuda", "Kd(PAR)",
        f"{g.station_id.nunique()} deployments", g)
    g = d[d.series == "argo"]
    reg = ar.apply(lambda r: argo_region(r.station_lat, r.station_lon), axis=1).value_counts()
    add("BGC-Argo floats", ", ".join(f"{k} ({v})" for k, v in reg.items()), "Kd(490)",
        f"{len(ar)} profiles from {ar.ID_float.nunique()} floats", g)
    t = pd.DataFrame(rows)
    t.loc[len(t)] = dict(platform="Total", program_region="", insitu_quantity="",
                         sites_or_platforms="", matchups=int(t.matchups.sum()), median_ratio=np.nan)
    return t


def turbidity_table(res):
    d = F.load_turbidity_pairs(res)
    st = d.groupby("station_id").size().rename("passes").reset_index()
    st[["network", "region"]] = st.station_id.apply(lambda s: pd.Series(network_of(s)))
    t = (st.groupby(["network", "region"], sort=False)
           .agg(stations=("station_id", "size"), icesat2_passes=("passes", "sum"))
           .reset_index())
    order = {net: i for i, (_, net, _) in enumerate(NETWORKS)}
    t = t.sort_values("network", key=lambda s: s.map(order).fillna(99)).reset_index(drop=True)
    t.loc[len(t)] = dict(network="Total", region="", stations=int(t.stations.sum()),
                         icesat2_passes=int(t.icesat2_passes.sum()))
    if (st.network == "Other").any():
        print("WARNING: unassigned stations:", st[st.network == "Other"].station_id.tolist())
    return t


def main():
    kd = kd_table(RES)
    kd.to_csv(os.path.join(RES, "tableS_kd_matchups.csv"), index=False)
    tb = turbidity_table(RES)
    tb.to_csv(os.path.join(RES, "tableS_turbidity_stations.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(kd.to_string(index=False))
        print()
        print(tb.to_string(index=False))


if __name__ == "__main__":
    main()
