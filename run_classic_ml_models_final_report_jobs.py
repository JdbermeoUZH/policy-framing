import os


from utils.constants import LANGUAGES, UNITS_OF_ANALYSES

map_language_names = {'en': 'english', 'it': 'italian', 'fr': 'french', 'po': 'polish', 'ru': 'russian', 'ge': 'german'}
#unit_of_analysis_groups = (('raw_text', 'title'), ('title_and_first_sentence_each_paragraph', 'title_and_first_paragraph'),
#                           ('title_and_5_sentences', 'title_and_10_sentences'))

unit_of_analysis_groups = (('raw_text', ), )

if __name__ == '__main__':

    for language in ['po']:#LANGUAGES:
        print(f'Launching jobs for language: {map_language_names[language]}')

        for analysis_unit_group in unit_of_analysis_groups:
            print(f'launching job for unit: {analysis_unit_group}')

            language_model_params = f'training.{map_language_names[language]}'

            os.environ['languages'] = language
            os.environ['analysis_unit'] = ' '.join(analysis_unit_group)
            os.environ['model_list'] = 'all'

            # Run the tunning
            os.environ['preprocessing_hyperparam_module'] = f'{language_model_params}.preprocesing_params_config'
            os.environ['use_same_params_across_units'] = str(1)
            os.environ['model_hyperparam_module'] = f'{language_model_params}.hyperparam_space_config'
            os.environ['default_params'] = str(0)
            os.environ['output_dir'] = ' '.join(['..', 'final_evaluation', 'classical_ml_models', 'tunned_default_preproc_params'])
            os.environ['metric_file_prefix'] = 'tunned'

            os.system('sbatch evaluate.sh')

            # Run models with default params
            os.environ['preprocessing_hyperparam_module'] = f'training.default_params.preprocesing_params_config'
            os.environ['use_same_params_across_units'] = str(1)
            os.environ['model_hyperparam_module'] = f'training.default_params.hyperparam_space_config'
            os.environ['default_params'] = str(1)
            os.environ['output_dir'] = ' '.join(['..', 'final_evaluation', 'classical_ml_models', 'default_parameters'])
            os.environ['metric_file_prefix'] = 'default_parameters'

            #os.system('sbatch evaluate.sh')


