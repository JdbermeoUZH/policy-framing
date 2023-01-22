#!/usr/bin/env bash
#SBATCH --output=new_jobs/preproc_pipeline_tunning_doc_freq_3gram_chained%2j.out
#SBATCH --time=38:10:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=61600
module load mamba
source activate Framing_py39
srun python benchmark_subtask_2.py --config_path_yaml config.yaml
