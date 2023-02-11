#!/usr/bin/env bash
#SBATCH --output=llm_jobs/%2j_gptJ.out
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres gpu:1
module load gpu
module load mamba
cat llm_benchmark.sh
cat config_llm_benchmark.yaml
source activate Framing_HF_2
srun python benchmark_llms.py \
  --config_path_yaml config_llm_benchmark.yaml\
  --model_name EleutherAI/gpt-j-6B\
  --gradient_accumulation_steps 2\
  --max_length_padding 512\
  --minibatch_size 2\
  --n_epochs 1\
  --analysis_unit raw_text
