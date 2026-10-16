# Original notebooks (Hannah Glover, 2025)

The first version of this workflow, written by Hannah Glover in June–July 2025.
The numbered scripts in the repository root are a refactor of these notebooks.
Outputs are stripped, and one local data path was replaced with `data`.
Otherwise the code is unchanged.

| Notebook | Became |
|---|---|
| `01_find_all_stations_github.ipynb` | `01_find_erddap_stations.py` |
| `02_match_{buoys,ships,turbidity,argo}2granules_ipx.ipynb` | `02_match_stations2granules.py` |
| `03_calckd_buoys.ipynb` | `03a_calc_kd_buoys.py`, `03b_calc_kd_ships.py` |
| `03_load_turbidity_data.ipynb` | `03c_compile_turbidity.py` |
| `03_proc_icesat4matlab.ipynb` | `03d_is2kd_sliderule.py` (ICESat-2 Kd now via SlideRule) |
| `04_plot_kd2kd_comparison.ipynb`, `04_plot_kd2turb_comparison.ipynb` | `04_pair_insitu.py`, `fig_is2_validation.py` |

They used icepyx downloads and an earlier version of the Kd code, so they
will not reproduce the current results as they stand.
