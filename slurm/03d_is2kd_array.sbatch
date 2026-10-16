#!/bin/bash
# Step 03d: ICESat-2 Kd for every manifest row via SlideRule, as a SLURM array.
# Each task runs a --start/--end slice serially (one Kd-pipeline run per row; the
# bootstrap is single-threaded, so 1 cpu per task). Each task writes its own part
# file (rewritten after every row) and progress json; merge afterwards:
#   python merge_sr_parts.py --tag buoys
#
# Requires the Kd pipeline repo ChaoEcohydroRS/RiverMouths_IS2KdMassiveTest at
# commit e997c61 (branch refactor/bma-polyfit) and outbound internet on compute nodes.
#   IS2KD_REPO       path to that repo         (default: ../RiverMouths_IS2KdMassiveTest)
#   IS2KD_PYTHON     python of its environment (default: the python running 03d)
#   IS2KD_WORKSPACE  dir holding Dataset/Bathy/gebco_2024_geotiff/ and Dataset/Shorelines/
#
#   sbatch slurm/03d_is2kd_array.sbatch                                  # buoys, 967 rows
#   sbatch --export=ALL,TAG=ships,NROWS=85 --array=0-9%10 slurm/03d_is2kd_array.sbatch
#   sbatch --export=ALL,TAG=turbidity,NROWS=2883 --array=0-49%10 slurm/03d_is2kd_array.sbatch
#   sbatch --export=ALL,TAG=argo,NROWS=87 --array=0-4 slurm/03d_is2kd_array.sbatch
#SBATCH --job-name=erddap_03d_is2kd
#SBATCH --array=0-19%10            # <=10 concurrent tasks, gentle on the shared SlideRule service
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G                  # measured peaks 1-3 GB; headroom for outliers
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
module load anaconda && source activate "${CONDA_ENV:-erddap_is2}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
TAG=${TAG:-buoys}
MANIFEST=${MANIFEST:-results/02_match_${TAG}_manifest.csv}
NROWS=${NROWS:-$(( $(wc -l < "$MANIFEST") - 1 ))}
NTASKS=${SLURM_ARRAY_TASK_COUNT:-20}
CHUNK=$(( (NROWS + NTASKS - 1) / NTASKS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK )); END=$(( START + CHUNK ))
if [ "$START" -ge "$NROWS" ]; then echo "task $SLURM_ARRAY_TASK_ID beyond manifest"; exit 0; fi
mkdir -p results/_parts logs
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID rows [$START,$END) of $NROWS  $(date)"
python 03d_is2kd_sliderule.py --manifest "$MANIFEST" \
  --output "results/_parts/is2_kd_${TAG}_part_${SLURM_ARRAY_TASK_ID}.csv" \
  --start "$START" --end "$END" --radii 5,8,10 --fetch-km 8 --timeout-per-row 900 \
  --work-dir "$PWD/_sliderule_kd_runs/${TAG}_${SLURM_ARRAY_TASK_ID}"
echo "task $SLURM_ARRAY_TASK_ID done $(date)"
