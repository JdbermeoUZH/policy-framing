import os
from types import ModuleType, GeneratorType

import yaml
import argparse
from typing import Tuple
from importlib import import_module

import spacy
from sklearn.preprocessing import MultiLabelBinarizer

from preprocessing.InputDataset import FramingArticleDataset
from preprocessing.BOWPipeline import BOWPipeline, basic_tokenizing_and_cleaning
from training.Logger import Logger
from training.MultiLabelEstimator import MultiLabelEstimator
from utils.metrics_config import scoring_functions
from utils import helper_fns

LANGUAGES = ('en', 'it', 'fr', 'po', 'ru', 'ge')

SPACY_MODELS = {
    'en': {'small': 'en_core_web_sm', 'large': 'en_core_web_trf'},
    'it': {'small': 'it_core_news_sm', 'large': 'it_core_news_lg'},
    'fr': {'small': 'fr_core_news_sm', 'large': 'fr_dep_news_trf'},
    'po': {'small': 'pl_core_news_sm', 'large': 'pl_core_news_lg'},
    'ru': {'small': 'ru_core_news_sm', 'large': 'ru_core_news_lg'},
    'ge': {'small': 'de_core_news_sm', 'large': 'de_dep_news_trf'}
}

LABELS = ('fairness_and_equality', 'security_and_defense', 'crime_and_punishment', 'morality',
          'policy_prescription_and_evaluation', 'capacity_and_resources', 'economic', 'cultural_identity',
          'health_and_safety', 'quality_of_life', 'legality_constitutionality_and_jurisprudence',
          'political', 'public_opinion', 'external_regulation_and_reputation')

UNITS_OF_ANALYSES = ('title', 'title_and_first_paragraph', 'title_and_5_sentences', 'title_and_10_sentences',
                     'title_and_first_sentence_each_paragraph', 'raw_text')


def parse_arguments_and_load_config_file() -> Tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(description='Subtask-2')
    parser.add_argument('--config_path_yaml', type=str,
                        help='Path to YAML configuration file overall benchmarking parameters')
    arguments = parser.parse_args()

    # Load parameters of configuration file
    with open(arguments.config_path_yaml, "r") as stream:
        try:
            yaml_config_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    return arguments, yaml_config_params


def report_train_test_performance(results_cv, report_metric: str = 'f1_micro'):
    mean_score_train = results_cv[f'train_{report_metric}'].mean()
    std_score_train = results_cv[f'train_{report_metric}'].std()
    mean_score_test = results_cv[f'test_{report_metric}'].mean()
    std_score_test = results_cv[f'test_{report_metric}'].std()
    train_test_mean_diff = mean_score_train - mean_score_test
    train_test_std_diff = (std_score_train ** 2 + std_score_test ** 2) ** (1/2)

    print(f"Perfromance on trainset ({report_metric}): {mean_score_train: 0.3f} (+-{std_score_train: 0.2f})")
    print(f"Perfromance on testset ({report_metric}): {mean_score_test: 0.3f} (+-{std_score_test: 0.2f})")
    print(f"train_score - test_score ({report_metric}): {train_test_mean_diff: 0.3f} (+-{train_test_std_diff: 0.2f})")


if __name__ == "__main__":
    # Load script arguments and configuration file
    args, config = parse_arguments_and_load_config_file()
    dataset_config = config['dataset']
    preprocessing_config = config['preprocessing']
    training_config = config['training']
    estimators_config = import_module(config['training']['model_hyperparam_module'])
    metric_log_config = config['metric_logging']
    run_config = config['run']

    preprocessing_tune_config = None
    if preprocessing_config['tune_preprocessing_params']:
        preprocessing_tune_config = preprocessing_config['param_search']

    if run_config['supress_warnings']:
        helper_fns.supress_all_warnings()

    # Run the experiments for the datasets selected
    for language in dataset_config['languages']:
        language_header = f"Currently running experiments for: {language}"
        print(language_header)
        print("#" * len(language_header) + '\n' + "#" * len(language_header))

        # Load input data and preprocess
        ################################
        nlp = spacy.load(SPACY_MODELS[language][preprocessing_config['spacy_model_size']])

        train_data = FramingArticleDataset(
            data_dir=dataset_config['data_dir'],
            language=language,
            subtask=dataset_config['subtask'],
            train_split=preprocessing_config['split'],
            load_preprocessed_units_of_analysis=preprocessing_config['load_preproc_input_data'],
            units_of_analysis_dir=os.path.join(dataset_config['data_dir'], 'preprocessed')
        )

        # If not extracted already, extract the different units of analysis
        if not preprocessing_config['load_preproc_input_data']:
            train_data.extract_all_units_of_analysis(nlp=nlp)

        # One hot encode the multilabel target
        mlb = MultiLabelBinarizer()
        mlb.fit([LABELS])

        y_train = mlb.transform(train_data.train_df.frames.str.lower().str.split(','))

        # Create additional object that will be used during the experiments
        ###################################################################

        # Define the vectorizing pipeline(s) to use
        bow_pipeline = BOWPipeline(
            tokenizer=lambda string: basic_tokenizing_and_cleaning(string, spacy_nlp_model=nlp),
            use_tfidf=preprocessing_config['use_tfidf'],
            min_df=preprocessing_config['min_df'],
            max_df=preprocessing_config['max_df'],
            max_features=preprocessing_config['max_features'],
            ngram_range=tuple(preprocessing_config['ngram_range']),
            min_var=preprocessing_config['min_var'],
            corr_threshold=preprocessing_config['corr_threshold']
        )

        # Iterate over each family of models in specified in yaml and .py config files
        # Estimate performance on the model using the different units of analysis
        units_of_analysis = UNITS_OF_ANALYSES if preprocessing_config['analysis_unit'] == 'all' \
            else [unit for unit in preprocessing_config['analysis_unit'] if unit in UNITS_OF_ANALYSES]

        #####################
        # Run the experiments
        #####################
        for unit_of_analysis in units_of_analysis:
            notify_current_unit_of_analysis = f"Unit of Analysis: {unit_of_analysis}"
            print(notify_current_unit_of_analysis)
            print("#" * len(notify_current_unit_of_analysis))

            # Whether to generate a list of preprocessing pipelines with multiple parameters or not
            if preprocessing_tune_config:
                vectorizing_pipelines = bow_pipeline.sample_pipelines_from_hypeparameter_space(
                    **preprocessing_tune_config)
            else:
                vectorizing_pipelines = (bow_pipeline.pipeline,)

            # Vectorize the text data
            for i, vectorizing_pipeline_i in enumerate(vectorizing_pipelines):
                if isinstance(vectorizing_pipelines, GeneratorType):
                    pipline_i_notification_str = f"Running experiments for preprocessing pipeline: {i}"
                    print(pipline_i_notification_str)
                    print('-' * len(pipline_i_notification_str))
                    print('-' * len(pipline_i_notification_str) + '\n')

                try:
                    X_train = vectorizing_pipeline_i.fit_transform(train_data.train_df[unit_of_analysis])
                except ValueError as value_e:
                    print(f'ERROR: {value_e}')
                    print('The parameters used where')
                    print(vectorizing_pipeline_i.get_params())
                    print('\n\n')
                    continue

                # Iterate over models types
                for model_name in training_config['model_list']:

                    model_type = estimators_config.MODEL_LIST[model_name]['model_type']
                    model_subtype = estimators_config.MODEL_LIST[model_name]['model_subtype']
                    if 'wrap_mlb_clf' in estimators_config.MODEL_LIST[model_name].keys():
                        wrap_mlb_clf = estimators_config.MODEL_LIST[model_name]['wrap_mlb_clf']
                    else:
                        wrap_mlb_clf = True

                    notify_current_model_str = f"Currently running estimates for model: {model_name}"
                    print(notify_current_model_str)
                    print("-"*len(notify_current_model_str))

                    # Define the object that will log performance with MLFlow
                    if metric_log_config['logging_level'] in ['outer_cv', 'model_wide']:
                        experiment_name = f"{metric_log_config['experiment_base_name']}_{language}_" \
                                          f"{unit_of_analysis}_{model_type}"
                    else:
                        experiment_name = f"{metric_log_config['experiment_base_name']}_{language}_" \
                                          f"{unit_of_analysis}_{model_name}"

                    metric_logger = Logger(
                        logging_dir=metric_log_config['logging_path'],
                        experiment_name=experiment_name,
                        rewrite_experiment=metric_log_config['rewrite_experiment'],
                        logging_level=metric_log_config['logging_level'],
                    )

                    # Define model
                    multilabel_cls = MultiLabelEstimator(
                        model_name=model_name,
                        base_estimator=estimators_config.MODEL_LIST[model_name]['model'],
                        base_estimator_hyperparam_dist=estimators_config.MODEL_LIST[model_name]['hyperparam_space'],
                        treat_labels_as_independent=training_config['mlb_cls_independent'],
                        scoring_functions=scoring_functions,
                        wrap_mlb_clf=wrap_mlb_clf
                    )

                    some_log_params = {
                        'language': language,
                        'unit_of_analysis': unit_of_analysis,
                        'spacy_model_used': SPACY_MODELS[language][preprocessing_config['spacy_model_size']],
                        'preprocessing_pipeline': vectorizing_pipeline_i,
                        'estimator': multilabel_cls,
                        'model_type': model_type,
                        'model_subtype': model_subtype
                    }

                    if training_config['default_params']:
                        results_cv = multilabel_cls.cross_validation(
                            X=X_train, y=y_train,
                            k_outer=training_config['nested_cv']['outer_folds'],
                            return_train_score=training_config['return_train_metrics'],
                            n_jobs=run_config['n_jobs']
                        )
                        metric_logger.log_model_wide_performance(cv_results=results_cv, **some_log_params)

                    else:
                        # Estimate performance with nested cross validation
                        if preprocessing_tune_config:
                            n_search_iter = training_config['nested_cv']['n_search_iter']
                        else:
                            has_n_search_iter = 'n_search_iter' in estimators_config.MODEL_LIST[model_name].keys()
                            n_search_iter = estimators_config.MODEL_LIST[model_name]['n_search_iter'] if has_n_search_iter\
                                else training_config['nested_cv']['n_search_iter']

                        try:
                            results_cv = multilabel_cls.nested_cross_validation(
                                X=X_train, y=y_train,
                                k_outer=training_config['nested_cv']['outer_folds'],
                                hyperparam_samples_per_outer_fold=n_search_iter,
                                k_inner=training_config['nested_cv']['outer_folds'],
                                ranking_score=training_config['nested_cv']['ranking_score'],
                                return_train_score=training_config['return_train_metrics'],
                                n_jobs=run_config['n_jobs']
                            )

                            # Log the results of the experiment
                            hyperparam_distrs_filepath = os.path.join(
                                *config['training']['model_hyperparam_module'].split('.'))
                            hyperparam_distrs_filepath += '.py'

                            metric_logger.log_metrics(
                                cv_results=results_cv,
                                hyperparam_distrs_filepath=hyperparam_distrs_filepath,
                                **some_log_params
                            )

                        except Exception as e:
                            print(f'Error while trying to fit model: {model_name}')
                            print(e)
                            continue

                    # Print model wide train and test error
                    report_train_test_performance(results_cv=results_cv, report_metric=training_config['metric_to_report'])

                    print('\n')
                print('\n')
            print('\n\n')
