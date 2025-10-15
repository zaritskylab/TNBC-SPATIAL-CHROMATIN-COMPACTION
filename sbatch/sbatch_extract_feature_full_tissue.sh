#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like so ##SBATCH
#SBATCH --partition main                         ### specify partition name where to run a job. short - 7 days time limit; debug – for testing - 2 hours and 1 job at a time
#SBATCH --time 3-11:30:00                      ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name extract_features                   ### name of the job. replace my_job with your desired job name
#SBATCH --output my_job-id-%J.out                ### output log for running job - %J is the job number variable
#SBATCH --mail-user=reutme@post.bgu.ac.il      ### users email for sending job status notifications
#SBATCH --mail-type=END,FAIL            ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=128G				### ammount of RAM memory
#SBATCH --cpus-per-task=15

### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

### Start you code below ####
module load anaconda              ### load anaconda module
source activate flim_stardist         ### activating Conda environment, environment must be configured before running the job

sample_type=$1

python /home/reutme/TNBC_FLIM/flim_analysis/feature_extraction/extract_feature_full_tissue_main.py $sample_type


