#!/usr/bin/env bash
#SBATCH --output=llm_benchmark_jobs/%2j_xlm_roberta_512_minib_4_gradsteps_4.out
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --gres gpu:1
module load gpu
module load mamba
source activate Framing_HF_2
srun python benchmark_llms.py \
  --config_path_yaml config_llm_benchmark.yaml\
  --gradient_accumulation_steps 4\
  --max_length_padding 512\
  --minibatch_size 4
