#!/bin/sh
#SBATCH --output=img_%A_%a.out  # Output file for each array job
#SBATCH --error=img%A_%a.err   # Error file for each array job
#SBATCH --job-name=christmas
#SBATCH --partition=short
#SBATCH --time=29
#SBATCH --array=[0-24]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


srun python tree_in_time.py $SLURM_ARRAY_TASK_ID