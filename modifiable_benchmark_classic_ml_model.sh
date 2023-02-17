#!/usr/bin/env bash
#SBATCH --output=benchmark_jobs/%2j_modifiable_job_classical_ml_model.out
#SBATCH --time=43:59:59
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600

module load mamba
source activate Framing_py39

srun python benchmark_subtask_2.py \
  --config_path_yaml config.yaml\
  --languages $languages\
  --analysis_unit $analysis_unit\
  --preprocessing_hyperparam_module $preprocessing_hyperparam_module\
  --tune_preprocessing_params $tune_preprocessing_params\
  --n_samples $n_samples\
  --model_hyperparam_module $model_hyperparam_module\
  --model_list $model_list\
  --default_params $default_params\
  --experiment_base_name $experiment_base_name
