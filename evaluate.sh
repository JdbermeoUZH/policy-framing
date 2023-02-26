#!/usr/bin/env bash
#SBATCH --output=evaluation_jobs/%2j_ge_evaluate.out
#SBATCH --time=38:10:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600
module load mamba
source activate Framing_py39
srun python tune_and_evaluate_classical_ml_models.py\
 --config_path_yaml config_classic_models_final_evaluation.yaml\
 --languages $languages\
 --analysis_unit $analysis_unit\
 --preprocessing_hyperparam_module $preprocessing_hyperparam_module\
 --model_hyperparam_module $model_hyperparam_module\
 --output_dir $output_dir\
 --metric_file_prefix $metric_file_prefix