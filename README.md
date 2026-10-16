# ERDDAP_MatchIS2

Find in-situ ocean-optics data on every public ERDDAP server, match it to
ICESat-2 overpasses, and compare in-situ light attenuation (K<sub>d</sub>) and
turbidity with ICESat-2 photon K<sub>dph</sub>. This repository holds the code and
the result tables behind the in-situ comparison (Fig. 5 and Tables S1–S2) of the
manuscript.

## Credits

- **Hannah Glover** wrote the original ERDDAP search, matchup and K<sub>d</sub>
  workflow as Jupyter notebooks (June–July 2025). They are kept, with outputs
  stripped, in [`legacy/`](legacy/).
- **Chao Wang** refactored them into the numbered scripts, added the SlideRule
  ICESat-2 K<sub>d</sub> stage (03d), the pairing (04), the tables (05) and the figure.
- ERDDAP server list: [IrishMarineInstitute/awesome-erddap](https://github.com/IrishMarineInstitute/awesome-erddap)
  (commit `df70c66`), copied to `config/erddaps.json`.
- BGC-Argo K<sub>d</sub>: Begouen Demeaux, C. (2024), *BGC-Argo matchups with Ocean
  Color Satellite Sensors and MERRA-2 updated for 2023* (v2), Zenodo,
  [10.5281/zenodo.8228242](https://doi.org/10.5281/zenodo.8228242), described in
  Begouen Demeaux et al. (2023, *Optics Express*). The matched
  profiles are included as `results/02_match_argo_profiles.csv` (CC BY 4.0, redistributed
  with attribution); download the full dataset from Zenodo to rerun step 02 for Argo.
- ICESat-2 photons are retrieved with [SlideRule Earth](https://slideruleearth.io),
  and K<sub>dph</sub> is computed by the Kd pipeline
  [ChaoEcohydroRS/RiverMouths_IS2KdMassiveTest](https://github.com/ChaoEcohydroRS/RiverMouths_IS2KdMassiveTest).

## Pipeline

| Step | Script | Does | Output (in `results/`) |
|---|---|---|---|
| 01 | `01_find_erddap_stations.py` | Lists every dataset with data after 1 Oct 2018 on each server, searches variable names for `chl turb sedi par phot secchi ssc`, tags platform type | `initial_erddap_stations.*`, `Saved_all_relevant_stations.pkl`, `erddap_station_search_results_labeled.csv` |
| 02 | `02_match_stations2granules.py` | For each dataset: ATL03 granules within a ±4 km box (NASA CMR via earthaccess), then the ERDDAP data within ±6 h of each pass (±12 h for Argo) | `02_match_{buoys,ships,turbidity,argo}_manifest.csv`, `02_match_argo_profiles.csv`, ERDDAP files in `data/` |
| 03a/b | `03a_calc_kd_buoys.py`, `03b_calc_kd_ships.py` | K<sub>d</sub>(PAR) per cast: log-linear fit of PAR vs depth over the upper 20 m | `03a_buoy_kd.csv`, `ship_kd.csv` |
| 03c | `03c_compile_turbidity.py` | Turbidity readings within 5 m of the surface | `turbidity_compiled.csv` |
| 03d | `03d_is2kd_sliderule.py` (+ `merge_sr_parts.py`) | ICESat-2 K<sub>dph</sub> per pass via SlideRule; median `kd_coherent` of 500 m bins within 5, 8 and 10 km of the station | `is2_kd_sliderule_{buoys,ships,turbidity,argo}.csv` |
| 04 | `04_pair_insitu.py` | Pairs each pass with its in-situ value | `paired_is2_{insitu_kd,turbidity,argo_kd}.csv` |
| 05 | `05_tables_supplement.py` | Supplementary tables from the plotted data | `tableS_kd_matchups.csv`, `tableS_turbidity_stations.csv` |
| fig | `fig_is2_validation.py` | Fig. 5 (a: vs in-situ K<sub>d</sub>, b: vs turbidity) | `fig_is2_validation.{png,pdf}` |

Glider matchups (OOI Endurance and Pioneer) come from a separate glider study;
`results/glider_is2_8km.csv` is included and `glider_reaggregate_8km.py` shows
how it was built (`GLIDER_DATA` points to that study's data).

The paper uses the **8 km** radius: the largest circle fully inside the 8 km
half-width SlideRule fetch box. `reaggregate_radius.py` adds another radius
from the kept per-pass files without new SlideRule calls.

## Quick start: rebuild the figure and tables from the included results

```bash
conda env create -f environment.yml && conda activate erddap_is2
python 04_pair_insitu.py          # paired tables
python 05_tables_supplement.py    # Tables S1-S2
python fig_is2_validation.py      # Fig. 5
```

Takes seconds and needs no network access. For BGC-Argo restricted to passes
within ±6 h of the profile, use `python 04_pair_insitu.py --argo-max-gap-h 6`.

## Full rerun

Needs an [Earthdata login](https://urs.earthdata.nasa.gov) (`~/.netrc` or
`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`) and, for step 03d:

- the Kd pipeline at commit **`e997c61`** (branch `refactor/bma-polyfit`), with
  its own Python environment:
  ```bash
  git clone -b refactor/bma-polyfit https://github.com/ChaoEcohydroRS/RiverMouths_IS2KdMassiveTest
  git -C RiverMouths_IS2KdMassiveTest checkout e997c61
  ```
- a workspace directory holding the GEBCO 2024 GeoTIFF tiles in
  `Dataset/Bathy/gebco_2024_geotiff/` and Natural Earth 10 m land in
  `Dataset/Shorelines/ne_10m_land/ne_10m_land.shp`.

Point to these with `IS2KD_REPO`, `IS2KD_PYTHON` and `IS2KD_WORKSPACE`. The
defaults assume both repositories sit side by side in the workspace. Then, from
the repo root (SLURM wrappers in `slurm/`; edit the environment lines for your
cluster):

```bash
mkdir -p logs
sbatch slurm/01_search.sbatch                                  # ~1 h
sbatch --export=ALL,SOURCE=buoys slurm/02_match.sbatch         # also ships, turbidity, argo
sbatch slurm/03_ground_truth.sbatch
sbatch slurm/03d_is2kd_array.sbatch                            # see header for other sources
python merge_sr_parts.py --tag buoys                           # after the array completes
python 04_pair_insitu.py && python 05_tables_supplement.py && python fig_is2_validation.py
```

ERDDAP servers add, move and retire datasets, so rerunning 01–02 today will not
return exactly the same counts as the original run (Feb 2026).

## Numbers in this version

Original run (Feb–Oct 2026), recounted from the included tables.

**Search:** 62 servers listed, 36 responded, 32,706 datasets with data after
1 Oct 2018, 6,552 tagged. The tagged list has 91 PAR-type and 161 turbidity
fixed-asset entries, which are 88 and 146 unique datasets once mirror copies on
several servers are merged.

**K<sub>d</sub>(PAR):** 967 coincident passes at 62 datasets → 324 cast fits
from 125 files (247 with |r| > 0.8; pairing uses |r| ≥ 0.9). 519 passes gave an
ICESat-2 K<sub>dph</sub>, 64 of them with an in-situ K<sub>d</sub>; plus 4 ship casts.

**Turbidity:** 2,883 coincident passes at 127 datasets → 1,027 passes from 59
datasets with near-surface turbidity (43,986 readings within 5 m, 0–500 NTU/FNU,
fluorescence units excluded) → 491 with a valid ICESat-2 K<sub>dph</sub> → 461 passes at
46 stations after merging three sites served under two IDs.

**BGC-Argo:** the source has 11,680 float × satellite rows (6,182 profiles from
225 floats); 5,129 rows (2,428 profiles) are after 1 Oct 2018. The ±12 h, ±4 km
search found 40 profiles; 36 gave an ICESat-2 K<sub>dph</sub>; 20 profiles from
15 floats have a float K<sub>d</sub>(490). Within ±6 h: 24 → 22 → 11 profiles
from 9 floats.

**Fig. 5:** (a) 196 matchups; (b) 461 passes at 46 stations,
per-station Spearman ρ = 0.47.

## Changes from the original run

The code here fixes issues found while preparing the paper; each was checked
against the original outputs (`_verify/`, not committed).

1. **Mirror datasets** (02): a dataset ID served by two servers was matched twice,
   duplicating 95 of 1,062 buoy and 242 of 3,125 turbidity manifest rows. The
   station list is now deduplicated by dataset ID, and manifest rows by
   (station, granule). The included manifests are deduplicated. In-situ results
   are unaffected, because 03a–c already read each file once.
2. **Turbidity depth window** (03c): `z < 2` kept every reading of datasets that
   store depth as negative `z` (down to 33 m) and dropped positive-depth sensors
   at 2–5 m. Now `|z| <= 5 m`. This adds three NERACOOS Gulf of Maine buoys
   (A01, E05, J04; sensors at 3 and 5 m; 26 passes) to Fig. 5b: 435 → 461 passes,
   43 → 46 stations. No other pass changed.
3. **Units** (03c/04): turbidity units are recorded, and fluorescence-unit (RFU)
   files are excluded automatically.
4. **Step 01 → 02 hand-off**: 02 now reads step-01 output directly (the `phot` and
   `turb` tags used by its filters were added by hand in the original run; they
   equal `photo` and `turbid`).
5. **Ship casts**: `ship_kd.csv` is regenerated from the current ship manifest
   (26 fits from 15 files; the original was computed from an earlier manifest).
   The 4 paired ship casts are identical.
6. **Pairing scripted** (04): the original pairing was done ad hoc; 04 reproduces
   it exactly (68 K<sub>d</sub> pairs, 20 Argo pairs).
7. **Partial results**: 03c and 03d rewrite their output after every file or row,
   so a killed job keeps its progress. 03d also writes the 8 km radius directly.

## Caveats

- Some ERDDAP servers report `z = 0` as a placeholder for subsurface instruments
  (e.g. OOI Pioneer RID and MFD turbidity sensors), so they pass the near-surface
  test.
- The time gap for Argo uses the ATL03 granule start time as the overpass time,
  which is accurate to a few minutes.
- Field depths, units and QC differ between providers; turbidity is used as
  reported (NTU or FNU).

## License

Code: [MIT](LICENSE). The data tables in `results/` derive from third-party
sources and keep their providers' terms: the BGC-Argo subset is CC BY 4.0
(Begouen Demeaux, 2024), and the ERDDAP-derived tables follow each data
provider's terms.
