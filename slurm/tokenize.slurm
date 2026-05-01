#!/bin/bash
#SBATCH --job-name=glm-prior-tokenize
#SBATCH --open-mode=append
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500GB
#SBATCH --partition=cpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@nyu.edu

set -euo pipefail

ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

source "$ROOT/venvs/central-rusty9/bin/activate"
module purge
module load modules/2.4-20250724
module load python/3.12
module load gcc/14.2.0

cd "$ROOT"

python -m train_prior_network.tokenize_sequences "$@"
