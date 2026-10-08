#!/bin/bash
#SBATCH --job-name=qpdk-dataset
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%A_%a.out
set -euo pipefail

generator=$1
output=$2
workdir=$3
shift 3
if [[ ${SLURM_ARRAY_TASK_MIN:?} != 0 || ${SLURM_ARRAY_TASK_STEP:?} != 1 || ${SLURM_ARRAY_TASK_MAX:?} != $((SLURM_ARRAY_TASK_COUNT - 1)) ]]; then
    echo "Use contiguous array indices starting at zero" >&2
    exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1

uv run --script "$generator" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --processes "$SLURM_CPUS_PER_TASK" \
    --output "$output/shard-$SLURM_ARRAY_TASK_ID" --workdir "$workdir/shard-$SLURM_ARRAY_TASK_ID" "$@"
