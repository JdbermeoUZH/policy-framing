#!/usr/bin/env bash
#SBATCH --output=benchmark_jobs/%2j_benchmark_raw_text_fixed.out
#SBATCH --time=43:59:59
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600
module load mamba
source activate Framing_py39
cd ../
srun python benchmark_subtask_2.py \
  --config_path_yaml config.yaml\
  --analysis_unit raw_text\
  --experiment_base_name benchmark_fixed\
  --tune_preprocessing_params 0\
