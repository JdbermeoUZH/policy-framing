import os

LLMS = {
    #'xlm-roberta-large': {
    #    'gradient_accumulation_steps': 2,
    #    'minibatch_size': 4,
    #    'n_epochs': 20,
    #    'max_length_padding': 512
    #},

    #'bert-base-multilingual-cased': {
    #    'gradient_accumulation_steps': 2,
    #    'minibatch_size': 16,
    #    'n_epochs': 100,
    #    'max_length_padding': 512
    #},

    #'distilbert-base-multilingual-cased': {
    #    'gradient_accumulation_steps': 1,
    #    'minibatch_size': 64,
    #    'n_epochs': 100,
    #    'max_length_padding': 512
    #}

    'gpt-2': {
        'gradient_accumulation_steps': 2,
        'minibatch_size': 4,
        'n_epochs': 20,
        'max_length_padding': 512
    },

    #'EleutherAI/gpt-neo-125M': {
    #    'gradient_accumulation_steps': 2,
    #    'minibatch_size': 2,
    #    'n_epochs': 20,
    #    'max_length_padding': 512
    #},

    #'AshtonIsNotHere/xlm-roberta-long-base-4096': {
    #    'gradient_accumulation_steps': 8,
    #    'minibatch_size': 1,
    #    'n_epochs': 20,
    #    'max_length_padding': 4096
    #},
}

UNITS_OF_ANALYSES = ('title', 'title_and_first_paragraph', 'title_and_5_sentences', 'title_and_10_sentences',
                     'title_and_first_sentence_each_paragraph', 'raw_text')


if __name__ == '__main__':
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

            print("run slurm job")
            os.system('sbatch modifiable_llm_benchmark.sh')
