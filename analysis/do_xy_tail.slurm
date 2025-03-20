#!/bin/sh 
#SBATCH --job-name=plots
#SBATCH --output=./outs/%x_%A_%a.out
#SBATCH --error=./errs/%x_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=10
#SBATCH --mail-user=salvatore.ferrone@obspm.fr
#SBATCH --mail-type=ALL
#SBATCH --array=[0-49]


module purge
module load python
source /obs/sferrone/py-env/bin/activate

python3 xy_tail_coordinates.py $SLURM_ARRAY_TASK_ID