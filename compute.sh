#!/usr/bin/env bash
#SBATCH --output=tunning_jobs/ru_rf%j.out
#SBATCH --time=36:10:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600
module load generic
module load anaconda3
source activate Framing
srun python benchmark_subtask_2.py --config_path_yaml config.yaml
