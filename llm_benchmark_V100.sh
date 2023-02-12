#!/usr/bin/env bash
#SBATCH --output=llm_jobs/%2j_GPT-NEO-1.3B.out
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --constraint=GPUMEM32GB
#SBATCH --gres=gpu:V100:1 

module load gpu
module load mamba

cat llm_benchmark.sh
cat config_llm_benchmark.yaml

source activate Framing_HF_2
nvidia-smi

srun python benchmark_llms.py \
  --config_path_yaml config_llm_benchmark.yaml\
  --model_name EleutherAI/gpt-neo-1.3B\
  --gradient_accumulation_steps 2\
  --max_length_padding 512\
  --minibatch_size 1\
  --n_epochs 1\
  --analysis_unit raw_text
