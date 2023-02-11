#!/usr/bin/env bash
#SBATCH --output=llm_benchmark_jobs/%2j_$model_name_minib_$minibatch_size_gradsteps_$gradient_accumulation_steps_warmup_0.2_data_collator_$analysis_unit.out
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
<<<<<<< HEAD
  --model_name bert-base-multilingual-cased\
  --gradient_accumulation_steps 1\
  --max_length_padding 16\
  --minibatch_size 4\
  --n_epochs 1\
  --analysis_unit title
=======
  --model_name $model_name\
  --gradient_accumulation_steps $gradient_accumulation_steps\
  --max_length_padding 512\
  --minibatch_size $minibatch_size\
  --n_epochs $n_epochs\
  --analysis_unit $analysis_unit
>>>>>>> 9e0278df52123a72db4130e96da88af7c4d8cb65
