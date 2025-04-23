#!/bin/sh
#SBATCH --output=out/plots.out
#SBATCH --error=err/plots.err
#SBATCH --job-name=plots
#SBATCH --partition=short
#SBATCH --time=30
#SBATCH --array=[0-150]


# Activate your conda environment
source /data/sferrone/miniconda3/etc/profile.d/conda.sh
conda activate gcs


perturber_index=$((SLURM_ARRAY_TASK_ID / 25))
host_index=$((SLURM_ARRAY_TASK_ID % 25))

echo "Processing: array_id=$SLURM_ARRAY_TASK_ID, perturber_index=$perturber_index, host_index=$host_index"


# Run script
srun python3 plot_tail_coordinates.py $perturber_index $host_index