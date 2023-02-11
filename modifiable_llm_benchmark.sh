#!/usr/bin/env bash
#SBATCH --output=llm_benchmark_jobs/%2j_${model_name}_minib_${minibatch_size}_gradsteps_${gradient_accumulation_steps}_warmup_0.2_data_collator_$analysis_unit.out
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
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
  --model_name $model_name\
  --gradient_accumulation_steps $gradient_accumulation_steps\
  --max_length_padding 512\
  --minibatch_size $minibatch_size\
  --n_epochs $n_epochs\
  --analysis_unit $analysis_unit