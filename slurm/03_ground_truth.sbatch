#!/bin/bash
# Steps 03a-c: in-situ Kd_PAR (buoys, ships) and near-surface turbidity from the
# ERDDAP files in data/. Fast (~1 min).
#SBATCH --job-name=erddap_03_truth
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module load anaconda && source activate "${CONDA_ENV:-erddap_is2}"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
python 03a_calc_kd_buoys.py    --manifest results/02_match_buoys_manifest.csv --input-dir data/
python 03b_calc_kd_ships.py    --manifest results/02_match_ships_manifest.csv --input-dir data/
python 03c_compile_turbidity.py --manifest results/02_match_turbidity_manifest.csv --input-dir data/ --max-depth 5
