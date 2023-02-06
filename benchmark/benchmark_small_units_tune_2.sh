#!/usr/bin/env bash
#SBATCH --output=benchmark_jobs/%2j_benchmark_small_units_tune.out
#SBATCH --time=43:59:59
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600
module load mamba
source activate Framing_py39
cd ../
srun python benchmark_subtask_2.py \
  --config_path_yaml config.yaml\
  --analysis_unit title title_and_first_paragraph title_and_5_sentences title_and_10_sentences\
  --experiment_base_name benchmark_tune_2\
  --tune_preprocessing_params 1\
  --n_samples 15\
