#!/bin/sh
#SBATCH --output=img_%A_%a.out  # Output file for each array job
#SBATCH --error=img%A_%a.err   # Error file for each array job
#SBATCH --job-name=christmas
#SBATCH --partition=medium
#SBATCH --time=300
#SBATCH --array=[0-24]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10 


### I'm trying to dynamically allocate the number of CPUs to the number of tasks based on the node 


srun python tree_in_time.py $SLURM_ARRAY_TASK_ID