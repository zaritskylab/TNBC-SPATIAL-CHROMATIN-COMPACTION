#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 7-00:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name classification_tissue_dists			### name of the job
#SBATCH --output job-%A_%a.out			       ### output log for running job - %A for array job ID, %a for array index
#SBATCH --gpus=0				### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1

# Note: the following 4 lines are commented out
##SBATCH --mail-user=user@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=64G				### ammount of RAM memory, allocating more than 60G requires IT team's permission
#SBATCH --array=0-4	            ### define job array with indices

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Array of Python commands
commands=(
  "python -u -m flim_analysis.distribution_classification.treatment_classification_tissue_wise --dist_csv_name features_lifetime_distribution_data_max_val_13_bins_amount_10_bin_range_1.3.csv  --n_seeds 100 --n_permutations 1000"
  "python -u -m flim_analysis.distribution_classification.treatment_classification_tissue_wise --dist_csv_name features_lifetime_distribution_data_max_val_13_bins_amount_18_bin_range_0.73.csv --n_seeds 100 --n_permutations 1000"
  "python -u -m flim_analysis.distribution_classification.treatment_classification_tissue_wise --dist_csv_name features_lifetime_distribution_data_max_val_13_bins_amount_23_bin_range_0.585.csv --n_seeds 100 --n_permutations 1000"
  "python -u -m flim_analysis.distribution_classification.treatment_classification_tissue_wise --dist_csv_name features_lifetime_distribution_data_max_val_13_bins_amount_34_bin_range_0.383.csv --n_seeds 100 --n_permutations 1000"
  "python -u -m flim_analysis.distribution_classification.treatment_classification_tissue_wise --dist_csv_name features_lifetime_distribution_data_max_val_13_bins_amount_42_bin_range_0.31.csv --n_seeds 100 --n_permutations 1000"
)

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate tnbc_flim   ### activate a conda environment, replace my_env with your conda environment
cd $SLURM_SUBMIT_DIR

### Run the command corresponding to the array index
eval ${commands[$SLURM_ARRAY_TASK_ID]}