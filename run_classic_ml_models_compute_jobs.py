import os


from utils.constants import LANGUAGES, UNITS_OF_ANALYSES


if __name__ == '__main__':
    truncated = 0

    for model_name, model_params in LLMS.items():
        print(model_name)
        print(model_params)
        for analysis_unit in UNITS_OF_ANALYSES:
            os.environ['analysis_unit'] = analysis_unit
            os.environ['model_name'] = model_name
            os.environ['gradient_accumulation_steps'] = str(model_params['gradient_accumulation_steps'])
            os.environ['minibatch_size'] = str(model_params['minibatch_size'])
            os.environ['n_epochs'] = str(model_params['n_epochs'])
            os.environ['max_length_padding'] = str(model_params['max_length_padding'])

            os.environ['truncated'] = str(truncated)

            if truncated == 0:
                os.environ['single_train_test_split_filepath'] = f'multilingual_train_test_{analysis_unit}_' \
                                                                 f'max_words_length_500_min_words_length_30_chunk_word_overlap_250.hf'
            elif truncated == 1:
                os.environ['single_train_test_split_filepath'] = 'multilingual_train_test_ds.hf'

            else:
                raise RuntimeError('truncated must be 0 or 1')

            print("run slurm job")

            if model_params['run_on'] == 'GPUMEM16GB':
                os.system('sbatch modifiable_llm_benchmark.sh')

            elif model_params['run_on'] == 'GPUMEM32GB':
                os.system('sbatch modifiable_llm_benchmark_V10032GB.sh')

            elif model_params['run_on'] == 'GPUMEM80GB':
                os.system('sbatch modifiable_llm_benchmark_A100.sh')

            else:
                os.system('sbatch modifiable_llm_benchmark.sh')

