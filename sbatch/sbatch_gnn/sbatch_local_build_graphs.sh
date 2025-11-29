#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like so ##SBATCH
#SBATCH --partition main                         ### specify partition name where to run a job. short - 7 days time limit; debug – for testing - 2 hours and 1 job at a time
#SBATCH --time 3-11:30:00                      ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name=build_graphs                   ### name of the job. replace my_job with your desired job name
#SBATCH --output out/my_job-id-%A_%a.out  # Output log (%A: Job Array ID, %a: Task ID)
#SBATCH --mail-user=reutme@post.bgu.ac.il      ### users email for sending job status notifications
#SBATCH --mail-type=END,FAIL            ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=128G				### ammount of RAM memory
#SBATCH --cpus-per-task=6

### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID

### Start you code below ####
module load anaconda              ### load anaconda module
source activate tnbc_flim_test    ### activate a conda environment, replace my_env with your conda environment

cd "$SLURM_SUBMIT_DIR"
cd "TNBC-SPATIAL-CHROMATIN-COMPACTION"
PY="flim_analysis/gnn_classification/build_graphs/build_graph_main.py"

graph_type=$1                      ### graph_type must be: gnn | structure_gnn | shuffling_gnn

echo "Running graph type: ${graph_type}"

if [[ "$graph_type" == "shuffling_gnn" ]]; then
    python -u "${PY}" "${graph_type}" --patch-size 1500 --overlap 0.75 --feature_type 'lifetime' --max_dist 30 --n_seeds 1
else
    python -u "${PY}" "${graph_type}" --patch-size 1500 --overlap 0.75 --feature_type 'lifetime' --max_dist 30
fi