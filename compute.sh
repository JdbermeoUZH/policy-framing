#!/usr/bin/env bash
#SBATCH --output=new_jobs/it_rf_%j.out
#SBATCH --time=32:10:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600
module load generic
module load anaconda3
source activate Framing
srun python benchmark_subtask_2.py --config_path_yaml config.yaml
