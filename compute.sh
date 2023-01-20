#!/usr/bin/env bash
#SBATCH --output=new_jobs/ru_mlb_ind_reg_part_1_1gram_dfv2_%j.out
#SBATCH --time=32:10:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600
module load mamba
source activate Framing_py39
srun python benchmark_subtask_2.py --config_path_yaml config.yaml
