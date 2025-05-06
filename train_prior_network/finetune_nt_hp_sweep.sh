#!/bin/bash

# Check if the necessary arguments are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: USER, Singularity overlay, or Singularity image not provided."
    echo "Usage: ./hp_sweep.sh <USER> <singularity_overlay> <singularity_img>"
    exit 1
fi

USER=$1
SINGULARITY_OVERLAY=$2
SINGULARITY_IMG=$3
PARENT_OUTPUT_DIR=$4

SLURM_SCRIPT="./train_prior_network/finetune_nt.slurm"
CONDA_ENV=/ext3/pmf-prior-network

# hyperparameter sweep:
gradient_accumulation_steps=(16 32 64 128)
class_weights=("[0.1,1.0]" "[0.2,1.0]" "[0.3,1.0]" "[0.4,1.0]" "[0.5,1.0]" "[0.6,1.0]" "[0.7,1.0]" "[0.8,1.0]" "[0.9,1.0]" "[1.0,1.0]") 
learning_rate=(0.00001)
downsample_rate=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for LR in ${learning_rate[@]}; do
    for CLASS_WEIGHT in "${class_weights[@]}"; do
        for DOWNSAMPLE_RATE in ${downsample_rate[@]}; do
            for G_ACCUM in ${gradient_accumulation_steps[@]}; do
                CLEAN_CLASS_WEIGHT=$(echo "${CLASS_WEIGHT}" | sed 's/,/_/g' | sed 's/\[//g' | sed 's/\]//g')
                RUN_NAME="batching-cw_${CLEAN_CLASS_WEIGHT}-dsr_${DOWNSAMPLE_RATE}-gas_${G_ACCUM}"
                OUTPUT_DIR="${PARENT_OUTPUT_DIR}${RUN_NAME}"
                OPT="training_args.learning_rate=${LR} \
                     training_args.output_dir=${OUTPUT_DIR} script_args.wandb_run_name=${RUN_NAME} \
                     script_args.class_weights=${CLASS_WEIGHT} script_args.downsample_rate=${DOWNSAMPLE_RATE} \
                     training_args.gradient_accumulation_steps=${G_ACCUM}"
                
                # Submit the job
                sbatch --job-name=${RUN_NAME} ${SLURM_SCRIPT} "${USER}" "${SINGULARITY_OVERLAY}" "${SINGULARITY_IMG}" "${OPT}"
                echo "Launched training job for ${RUN_NAME}"
            done
        done
    done
done