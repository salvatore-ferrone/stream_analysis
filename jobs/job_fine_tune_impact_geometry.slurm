#!/bin/sh 
#SBATCH --job-name=fine_time
#SBATCH --output=./outs/fine_time.out
#SBATCH --error=./errs/fine_time.err
#SBATCH --partition=medium
#SBATCH --time=180
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

python3 fine_tune_loop.py $perturberName


