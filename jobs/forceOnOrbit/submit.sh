#!/bin/sh
#SBATCH --output=FOORB.out
#SBATCH --error=FOORB.err
#SBATCH --job-name=FOORB
#SBATCH --partition=short
#SBATCH --time=59
#SBATCH --nodes=1 --ntasks-per-node=15



srun python mp_compute.py
