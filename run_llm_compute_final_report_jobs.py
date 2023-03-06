import os

from utils.constants import UNITS_OF_ANALYSES


LLMS = {
    'xlm-roberta-large': {
        'gradient_accumulation_steps': 2,
        'minibatch_size': 4,
        'n_epochs': 7,
        'max_length_padding': 512,
        'run_on': 'GPUMEM16GB',
        'units_of_analysis': ['title', 'title_and_first_paragraph', 'title_and_5_sentences', 'title_and_10_sentences', 'title_and_first_sentence_each_paragraph', 'raw_text']
    },

    'bert-base-multilingual-cased': {
        'gradient_accumulation_steps': 2,
        'minibatch_size': 16,
        'n_epochs': 65,
        'max_length_padding': 512,
        'run_on': 'GPUMEM16GB',
        'units_of_analysis': ['title', 'title_and_first_paragraph']
    },

    'distilbert-base-multilingual-cased': {
        'gradient_accumulation_steps': 1,
        'minibatch_size': 64,
        'n_epochs': 75,
        'max_length_padding': 512,
         'run_on': 'GPUMEM16GB',
        'units_of_analysis': ['title_and_first_paragraph', 'title_and_5_sentences', 'title_and_10_sentences', 'title_and_first_sentence_each_paragraph', 'raw_text']
    },

    'gpt2': {
        'gradient_accumulation_steps': 2,
        'minibatch_size': 4,
        'n_epochs': 15,
        'max_length_padding': 512,
        'run_on': 'GPUMEM16GB',
        'units_of_analysis': ['title', 'title_and_first_paragraph', 'title_and_5_sentences']
    },

    'EleutherAI/gpt-neo-125M': {
        'gradient_accumulation_steps': 4,
        'minibatch_size': 2,
        'n_epochs': 13,
        'max_length_padding': 512,
        'run_on': 'GPUMEM16GB',
        'units_of_analysis': ['title', 'title_and_first_paragraph', 'title_and_5_sentences', 'title_and_10_sentences', 'title_and_first_sentence_each_paragraph', 'raw_text']
    },

    'AshtonIsNotHere/xlm-roberta-long-base-4096': {
        'gradient_accumulation_steps': 2,
        'minibatch_size': 4,
        'n_epochs': 20,
        'max_length_padding': 4096,
        'run_on': 'GPUMEM32GB',
        'units_of_analysis': ['title', 'title_and_first_paragraph']
    },

    'facebook/mbart-large-50': {
        'gradient_accumulation_steps': 2,
        'minibatch_size': 4,
        'n_epochs': 7,
        'max_length_padding': 512,
        'run_on': 'GPUMEM32GB',
        'units_of_analysis': ['title', 'title_and_first_paragraph']
    },

    'EleutherAI/gpt-neo-1.3B': {
        'gradient_accumulation_steps': 4,
        'minibatch_size': 2,
        'n_epochs': 8,
        'max_length_padding': 512,
        'run_on': 'GPUMEM80GB',
        'units_of_analysis': ['title', 'title_and_first_paragraph', 'title_and_5_sentences', 'title_and_10_sentences', 'title_and_first_sentence_each_paragraph', 'raw_text']
    },
}

AVOID = ['xlm-roberta-large', 'bert-base-multilingual-cased', 'distilbert-base-multilingual-cased', 'gpt2',
         'EleutherAI/gpt-neo-125M', 'facebook/mbart-large-50', 'AshtonIsNotHere/xlm-roberta-long-base-4096']

if __name__ == '__main__':
    truncated = 0

    for model_name, model_params in LLMS.items():

        if model_name in AVOID:
            continue

        print(model_name)
        print(model_params)
        for analysis_unit in model_params['units_of_analysis']:
            print(f'\t{analysis_unit}')
            os.environ['analysis_unit'] = analysis_unit
            os.environ['model_name'] = model_name
            os.environ['gradient_accumulation_steps'] = str(model_params['gradient_accumulation_steps'])
            os.environ['minibatch_size'] = str(model_params['minibatch_size'])
            os.environ['n_epochs'] = str(model_params['n_epochs'])
            os.environ['max_length_padding'] = str(model_params['max_length_padding'])

            os.environ['truncated'] = str(truncated)

            if truncated == 0:
                os.environ['single_train_test_split_filepath'] = f'multilingual_train_test_80_20_split_{analysis_unit}_' \
                                                                 f'max_words_length_500_min_words_length_30_chunk_word_overlap_250.hf'
            elif truncated == 1:
                os.environ['single_train_test_split_filepath'] = 'multilingual_train_test_ds_80_20_split.hf'

            else:
                raise RuntimeError('truncated must be 0 or 1')

            print("run slurm job")

            if model_params['run_on'] == 'GPUMEM16GB':
                os.system('sbatch evaluate_llm.sh')

            elif model_params['run_on'] == 'GPUMEM32GB':
                os.system('sbatch evaluate_llm_V100.sh')

            elif model_params['run_on'] == 'GPUMEM80GB':
                os.system('sbatch evaluate_llm_A100.sh')

            else:
                assert False
