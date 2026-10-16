#!/bin/bash
# Step 01: enumerate ERDDAP servers and tag candidate datasets (network-bound, ~1 h).
# Submit from the repo root:  mkdir -p logs && sbatch slurm/01_search.sbatch
#SBATCH --job-name=erddap_01_search
#SBATCH --cpus-per-task=32          # 30 metadata-request threads
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
# --- cluster-specific environment (edit for your system) ---
module load anaconda && source activate "${CONDA_ENV:-erddap_is2}"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
python 01_find_erddap_stations.py --step1-force-rerun --step2-force-rerun
