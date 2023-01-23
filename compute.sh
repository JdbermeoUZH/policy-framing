#!/usr/bin/env bash
#SBATCH --output=new_jobs/%2j_ru_preproc_pipeline_tunning_df_ind_3gram.out
#SBATCH --time=38:10:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600
module load mamba
source activate Framing_py39
srun python benchmark_subtask_2.py --config_path_yaml config.yaml
