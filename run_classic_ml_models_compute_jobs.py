import os


from utils.constants import LANGUAGES, UNITS_OF_ANALYSES

map_language_names = {'en': 'english', 'it': 'italian', 'fr': 'french', 'po': 'polish', 'ru': 'russian', 'ge': 'german'}

if __name__ == '__main__':
    for language in LANGUAGES:
        print(f'Launching jobs for language: {map_language_names[language]}')

        for analysis_unit in UNITS_OF_ANALYSES:
            print(f'launching job for unit: {analysis_unit}')

            language_model_params = f'training.{map_language_names[language]}'

            os.environ['languages'] = language
            os.environ['analysis_unit'] = analysis_unit
            os.environ['preprocessing_hyperparam_module'] = f'{language_model_params}.preprocesing_params_config.py'
            os.environ['model_hyperparam_module'] = f'{language_model_params}.hyperparam_space_config_default'
            os.environ['model_list'] = 'all'
            os.environ['default_params'] = str(0)
            os.environ['n_samples'] = str(0)

            # Launch job with fixed preprocessing params
            os.environ['tune_preprocessing_params'] = str(0)
            os.environ['experiment_base_name'] = 'benchmark_fixed_preproc_params'

            os.system('sbatch modifiable_benchmark_classic_ml_model.sh')

            # Launch 2 job of 15 iterations each to fine-tune the preprocessing parameters
            os.environ['tune_preprocessing_params'] = str(1)
            os.environ['n_samples'] = str(10)

            os.environ['experiment_base_name'] = 'benchmark_tune_preproc_params_1'
            os.system('sbatch modifiable_benchmark_classic_ml_model.sh')

            os.environ['experiment_base_name'] = 'benchmark_tune_preproc_params_2'
            os.system('sbatch modifiable_benchmark_classic_ml_model.sh')
            break
        break
