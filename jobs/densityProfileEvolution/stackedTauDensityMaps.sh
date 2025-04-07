#!/bin/sh
#SBATCH --output=out/stacked_%A_%a.out  # Output file for each array job
#SBATCH --error=err/stacked_%A_%a.err   # Error file for each array job
#SBATCH --job-name=stacked
#SBATCH --partition=short
#SBATCH --time=30
## NOT YET SBATCH --array=[0-24]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10 


### I'm trying to dynamically allocate the number of CPUs to the number of tasks based on the node 


srun python stackedTauDensityMaps.py 1