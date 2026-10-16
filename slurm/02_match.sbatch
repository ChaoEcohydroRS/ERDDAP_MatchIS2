#!/bin/bash
# Step 02: match stations to ICESat-2 ATL03 granules (CMR via earthaccess) and
# download the coincident ERDDAP data. Needs Earthdata credentials (~/.netrc or
# EARTHDATA_USERNAME / EARTHDATA_PASSWORD).
#   sbatch --export=ALL,SOURCE=buoys     slurm/02_match.sbatch
#   sbatch --export=ALL,SOURCE=ships     slurm/02_match.sbatch
#   sbatch --export=ALL,SOURCE=turbidity slurm/02_match.sbatch
#   sbatch --export=ALL,SOURCE=argo      slurm/02_match.sbatch   (needs data/Dataset_Kd_Paper_2018.csv)
# DRY_RUN=1 (default) searches without downloading ATL03 granules; the paper used
# SlideRule for the photons, so granule downloads are not needed.
#SBATCH --job-name=erddap_02_match
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module load anaconda && source activate "${CONDA_ENV:-erddap_is2}"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data results
SOURCE="${SOURCE:-buoys}"
FLAGS=""; [ "${DRY_RUN:-1}" -eq 1 ] && FLAGS="--dry-run"
python 02_match_stations2granules.py --source "$SOURCE" --output-dir data/ \
    --log-dir logs/ --version 007 $FLAGS
