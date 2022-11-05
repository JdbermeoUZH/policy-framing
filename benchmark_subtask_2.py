import os
from types import ModuleType

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

SPACY_MODELS = {
    'en': {'small': 'en_core_web_sm', 'large': 'en_core_web_trf'},
    'ru': {'small': 'ru_core_news_sm', 'large': 'ru_core_news_lg'},
    'it': {'small': 'it_core_news_sm', 'large': 'it_core_news_lg'},
    'fr': {'small': 'fr_core_news_sm', 'large': 'fr_dep_news_trf'},
    'po': {'small': 'pl_core_news_sm', 'large': 'pl_core_news_lg'},
    'ge': {'small': 'de_core_news_sm', 'large': 'de_dep_news_trf'}
}

LABELS = ('fairness_and_equality', 'security_and_defense', 'crime_and_punishment', 'morality',
          'policy_prescription_and_evaluation', 'capacity_and_resources', 'economic', 'cultural_identity',
          'health_and_safety', 'quality_of_life', 'legality_constitutionality_and_jurisprudence',
          'political', 'public_opinion', 'external_regulation_and_reputation')

UNITS_OF_ANALYSES = ('title', 'title_and_first_paragraph', 'title_and_5_sentences', 'title_and_10_sentences',
                     'title_and_first_sentence_each_paragraph', 'raw_text')

DEFAULT_SCORING_FUNCTIONS = ('f1_micro', 'f1_macro', 'accuracy', 'precision_micro',
                             'precision_macro', 'recall_micro', 'recall_macro')


def parse_arguments_and_load_config_file() -> Tuple[argparse.Namespace, dict, ModuleType]:
    parser = argparse.ArgumentParser(description='Subtask-2')
    parser.add_argument('--config_path_yaml', type=str,
                        help='Path to YAML configuration file overall benchmarking parameters')
    parser.add_argument('--config_path_py_models', type=str,
                        help="Path to '.py' configuration file with model implementations and "
                             "hyper-parameter distributions to search with RandomSearch")
    arguments = parser.parse_args()

    # Load parameters of configuration file
    with open(arguments.config_path_yaml, "r") as stream:
        try:
            yaml_config_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    # Import '.py' config file with models and their search space
    estimators_config_ = import_module(arguments.config_path_py_models)

    return arguments, yaml_config_params, estimators_config_


if __name__ == "__main__":
    # Load script arguments and configuration file
    args, config, estimators_config = parse_arguments_and_load_config_file()
    dataset_config = config['dataset']
    preprocessing_config = config['preprocessing']
    training_config = config['training']
    metric_log_config = config['metric_logging']

    # Load input data and preprocess
    nlp = spacy.load(SPACY_MODELS[dataset_config['language']][preprocessing_config['spacy_model_size']])

    train_data = FramingArticleDataset(
        data_dir=dataset_config['data_dir'],
        language=dataset_config['language'],
        subtask=dataset_config['subtask'],
        split=preprocessing_config['split'],
        load_preprocessed_units_of_analysis=preprocessing_config['load_preproc_input_data'],
        units_of_analysis_dir=os.path.join(dataset_config['data_dir'], 'preprocessed')
    )

    # Preprocess text data
    if not preprocessing_config['load_preproc_input_data']:
        train_data.extract_all_units_of_analysis(nlp=nlp)

    vectorizing_pipeline = BOWPipeline(
        tokenizer=lambda string: basic_tokenizing_and_cleaning(string, spacy_nlp_model=nlp),
        use_tfidf=preprocessing_config['use_tfidf'],
        min_df=preprocessing_config['min_df'],
        max_df=preprocessing_config['max_df'],
        max_features=preprocessing_config['max_features'],
        ngram_range=preprocessing_config['ngram_range'],
        min_var=preprocessing_config['min_var'],
        corr_threshold=preprocessing_config['corr_threshold']
    )

    mlb = MultiLabelBinarizer()
    mlb.fit([LABELS])

    y_train = mlb.transform(train_data.df.frames.str.lower().str.split(','))

    # Log performance with MLFlow
    metric_logger = Logger(
        logging_dir=metric_log_config['logging_path'],
        experiment_name=metric_log_config['experiment_name'],
        rewrite_experiment=metric_log_config['rewrite_experiment']
    )

    # Iterate over each family of models in specified in yaml and .py config files
    # Estimate performance on the model using the different units of analysis
    units_of_analysis = UNITS_OF_ANALYSES if preprocessing_config['analysis_unit'] == 'all' \
        else [preprocessing_config['analysis_unit']]

    for unit_of_analysis in units_of_analysis:
        X_train = vectorizing_pipeline.pipeline.fit_transform(train_data.df[unit_of_analysis])

        for model_name in training_config['model_list']:
            print(f"Currently running estimates for model: {model_name}")
            print("#"*50)
            # Define model
            multilabel_cls = MultiLabelEstimator(
                base_estimator=estimators_config.MODEL_LIST[model_name]['model'],
                base_estimator_hyperparam_dist=estimators_config.MODEL_LIST[model_name]['hyperparam_space'],
                treat_labels_as_independent=training_config['mlb_cls_independent'],
                scoring_functions=DEFAULT_SCORING_FUNCTIONS
            )

            some_log_params = {
                'language': dataset_config['language'],
                'unit_of_analysis': unit_of_analysis,
                'spacy_model_used': SPACY_MODELS[dataset_config['language']][preprocessing_config['spacy_model_size']],
                'preprocessing_pipeline': vectorizing_pipeline,
                'estimator': multilabel_cls
            }

            if training_config['default_params']:
                cv_results = multilabel_cls.cross_validation(
                    X=X_train, y=y_train,
                    k_outer=training_config['nested_cv']['outer_folds'],
                )
                metric_logger.log_model_wide_performance(cv_results=cv_results, **some_log_params)

            else:
                # Estimate performance with nested cross validation
                model_specific_n_search_iter = estimators_config.MODEL_LIST[model_name]['n_search_iter']
                n_search_iter = model_specific_n_search_iter if model_specific_n_search_iter is not None \
                    else training_config['nested_cv']['n_search_iter']

                nested_cv_results = multilabel_cls.nested_cross_validation(
                    X=X_train, y=y_train,
                    k_outer=training_config['nested_cv']['outer_folds'],
                    hyperparam_samples_per_outer_fold=n_search_iter,
                    k_inner=training_config['nested_cv']['outer_folds'],
                    ranking_score=training_config['nested_cv']['ranking_score']
                )

                # Log the results of the experiment
                metric_logger.log_model_wide_performance(cv_results=nested_cv_results, **some_log_params)
                metric_logger.log_hyper_param_performance_outer_fold(cv_results=nested_cv_results, **some_log_params)
                metric_logger.log_hyper_param_performance_inner_fold(cv_results=nested_cv_results, **some_log_params)
