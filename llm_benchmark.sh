#!/usr/bin/env bash
#SBATCH --output=llm_jobs/%2j_test_chunked_prediction.out
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres gpu:1

module load gpu
module load mamba

cat llm_benchmark.sh
cat config_llm_benchmark.yaml

source activate Framing_HF_2
nvidia-smi

srun python benchmark_llms.py \
  --config_path_yaml config_llm_benchmark.yaml\
  --model_name bert-base-multilingual-cased\
  --gradient_accumulation_steps 2\
  --max_length_padding 512\
  --minibatch_size 16\
  --n_epochs 1\
  --analysis_unit raw_text\
  --truncated 0\
  --single_train_test_split_filepath multilingual_train_test_raw_text_max_words_length_500_min_words_length_30_chunk_word_overlap_250.hf\
  --metrics_output_dir tuning_results_llms multilingual test_cluster
