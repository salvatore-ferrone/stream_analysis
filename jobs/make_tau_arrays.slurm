#!/bin/sh 
#SBATCH --job-name=tau
#SBATCH --output=./errs/tau_%A_%a.out
#SBATCH --error=./outs/tau_%A_%a.out
#SBATCH --partition=medium
#SBATCH --time=1330
#SBATCH --mail-user=salvatore.ferrone@obspm.fr
#SBATCH --mail-type=ALL
#SBATCH --array=[0-49]
#SBATCH --cpus-per-task=10


source /obs/sferrone/py-env-gc/bin/activate

python3 make_tau_arrays.py $SLURM_ARRAY_TASK_ID

