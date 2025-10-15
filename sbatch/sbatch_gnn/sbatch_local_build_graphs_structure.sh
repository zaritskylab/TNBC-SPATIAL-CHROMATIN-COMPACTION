#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like so ##SBATCH
#SBATCH --partition main                         ### specify partition name where to run a job. short - 7 days time limit; debug – for testing - 2 hours and 1 job at a time
#SBATCH --time 3-11:30:00                      ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name=build_graphs                   ### name of the job. replace my_job with your desired job name
#SBATCH --output /home/reutme/out/my_job-id-%A_%a.out  # Output log (%A: Job Array ID, %a: Task ID)
#SBATCH --mail-user=reutme@post.bgu.ac.il      ### users email for sending job status notifications
#SBATCH --mail-type=END,FAIL            ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=64G				### ammount of RAM memory
#SBATCH --cpus-per-task=6

### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID

### Start you code below ####
module load anaconda              ### load anaconda module
source activate flim_torch       ## activating Conda environment, environment must be configured before running the job

python /home/reutme/TNBC_FLIM/flim_analysis/gnn_clssification/build_graphs/local_build_graph_structure_main.py