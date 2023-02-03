#!/usr/bin/env bash
#SBATCH --output=evaluation_jobs/%2j_ge_evaluate.out
#SBATCH --time=38:10:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600
module load mamba
source activate Framing_py39
srun python tune_and_evaluate_models.py --config_path_yaml evaluate/config.yaml
