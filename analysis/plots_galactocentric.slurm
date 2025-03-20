#!/bin/sh 
#SBATCH --job-name=plots
#SBATCH --output=./outs/%x_%A_%a.out
#SBATCH --error=./errs/%x_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=10
#SBATCH --mail-user=salvatore.ferrone@obspm.fr
#SBATCH --mail-type=ALL
#SBATCH --array=[1-31]


path_to_files="/obs/sferrone/stream_analysis/jobs/Pal5-all-perturber-suspects.txt"


# print line N of file 
module purge
module load python
source /obs/sferrone/py-env/bin/activate



perturberName=$(head -n $SLURM_ARRAY_TASK_ID $path_to_files | tail -n 1)

echo $perturberName

python3 plots_galactocentric_loop.py $perturberName


