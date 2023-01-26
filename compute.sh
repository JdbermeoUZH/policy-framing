#!/usr/bin/env bash
#SBATCH --output=new_jobs/%2jen_tune_best_models_tune_preproc.out
#SBATCH --time=38:10:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600
module load mamba
source activate Framing_py39
srun python benchmark_subtask_2.py --config_path_yaml config.yaml
