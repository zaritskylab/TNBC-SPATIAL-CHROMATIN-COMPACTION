#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like so ##SBATCH
#SBATCH --partition main                         ### specify partition name where to run a job. short - 7 days time limit; debug – for testing - 2 hours and 1 job at a time
#SBATCH --time 3-11:30:00                      ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name create_median_and_lifetime_distribution      ### name of the job. replace my_job with your desired job name
#SBATCH --array=0-4                             
#SBATCH --output out/%x_%A_%a.out               # e.g., logs/extract_features_123456_0.out
#SBATCH --mail-user=user@post.bgu.ac.il      ### users email for sending job status notifications
#SBATCH --mail-type=END,FAIL            ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=64G				### ammount of RAM memory
#SBATCH --cpus-per-task=15

### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

### Start you code below ####
module load anaconda              ### load anaconda module
source activate tnbc_flim_test    ### activate a conda environment, replace my_env with your conda environment

cd "$SLURM_SUBMIT_DIR"
cd "TNBC-SPATIAL-CHROMATIN-COMPACTION"
PY="flim_analysis/feature_extraction/create_distribution_and_median.py"

# --- commands array (index by SLURM_ARRAY_TASK_ID) ---
commands=(
    "python -u ${PY} core --max-val 13 --bin-range 1.3"         ### lifetime distribution with 10 bins
    "python -u ${PY} core --max-val 13 --bin-range 0.73"        ### lifetime distribution with 18 bins
    "python -u ${PY} core --max-val 13 --bin-range 0.585"       ### lifetime distribution with 23 bins
    "python -u ${PY} core --max-val 13 --bin-range 0.383"       ### lifetime distribution with 34 bins
    "python -u ${PY} core --max-val 13 --bin-range 0.31"        ### lifetime distribution with 42 bins
)

# --- run selected command ---
echo "Running: ${commands[$SLURM_ARRAY_TASK_ID]}"
eval "${commands[$SLURM_ARRAY_TASK_ID]}"
