#!/usr/bin/env bash
#SBATCH --output=benchmark_jobs/%2j_fr_benchmark_1st_sent_each_p_fixed.out
#SBATCH --time=43:59:59
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600
module load mamba
source activate Framing_py39
srun python benchmark_subtask_2.py --config_path_yaml config.yaml
