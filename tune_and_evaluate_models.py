import os
import yaml
import argparse
from typing import Tuple
from importlib import import_module

import pandas as pd
import spacy
from joblib import dump
from sklearn.preprocessing import MultiLabelBinarizer

from training.MultiLabelEstimator import MultiLabelEstimator
from preprocessing.InputDataset import FramingArticleDataset
from preprocessing.BOWPipeline import BOWPipeline, basic_tokenizing_and_cleaning
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

LABELS = ('Economic', 'Capacity_and_resources', 'Morality', 'Fairness_and_equality',
          'Legality_Constitutionality_and_jurisprudence', 'Policy_prescription_and_evaluation', 'Crime_and_punishment',
          'Security_and_defense', 'Health_and_safety', 'Quality_of_life', 'Cultural_identity', 'Public_opinion',
          'Political', 'External_regulation_and_reputation')


def parse_arguments_and_load_config_file() -> Tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(description='Subtask-2')
    parser.add_argument('--config_path_yaml', type=str,
                        help='Path to YAML configuration file overall benchmarking parameters')
    parser.add_argument('--language', type=str, default=None,
                        help='Path to YAML configuration file overall benchmarking parameters')
    parser.add_argument('--train_set', type=str, default=None,
                        help='Path to YAML configuration file overall benchmarking parameters')
    parser.add_argument('--eval_set', type=str, default=None,
                        help='Path to YAML configuration file overall benchmarking parameters')
    arguments = parser.parse_args()

    # Load parameters of configuration file
    with open(arguments.config_path_yaml, "r") as stream:
        try:
            yaml_config_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    if arguments.language is not None:
        yaml_config_params['dataset']['language'] = [arguments.language]

    if arguments.train_set is not None:
        yaml_config_params['evaluate']['train_set'] = arguments.train_set

    if arguments.eval_set is not None:
        yaml_config_params['evaluate']['eval_set'] = arguments.eval_set

    return arguments, yaml_config_params


if __name__ == "__main__":
    # Load script arguments and configuration file
    args, config = parse_arguments_and_load_config_file()
    dataset_config = config['dataset']
    preprocessing_config = config['preprocessing']
    run_config = config['run']
    evaluate_config = config['evaluate']
    models_config = evaluate_config['models']
    estimators_config = import_module(models_config['hyperparam_module'])

    os.makedirs(evaluate_config['output_dir'], exist_ok=True)
    best_model_dir_path = os.path.join(evaluate_config['output_dir'], 'best_models')
    os.makedirs(best_model_dir_path, exist_ok=True)

    predicted_instances_dir_path = os.path.join(evaluate_config['output_dir'], 'predicted_instances')
    os.makedirs(predicted_instances_dir_path, exist_ok=True)

    predicted_instances_train_dir_path = os.path.join(
        predicted_instances_dir_path, f"on_trainset_{evaluate_config['train_set']}")
    predicted_instances_test_dir_path = os.path.join(
        predicted_instances_dir_path, f"on_evalset_{evaluate_config['eval_set']}")
    os.makedirs(predicted_instances_train_dir_path, exist_ok=True)
    os.makedirs(predicted_instances_test_dir_path, exist_ok=True)

    if run_config['supress_warnings']:
        helper_fns.supress_all_warnings()

    # Preprocess the data to train the models
    for language in dataset_config['languages']:
        best_model_dir_path = os.path.join(best_model_dir_path, language)
        os.makedirs(best_model_dir_path, exist_ok=True)
        predicted_instances_train_dir_path = os.path.join(predicted_instances_train_dir_path, language)
        os.makedirs(predicted_instances_train_dir_path, exist_ok=True)
        predicted_instances_test_dir_path = os.path.join(predicted_instances_test_dir_path, language)
        os.makedirs(predicted_instances_test_dir_path, exist_ok=True)

        nlp = spacy.load(SPACY_MODELS[language][preprocessing_config['spacy_model_size']])
        print(evaluate_config['train_set'])
        print(evaluate_config['eval_set'])
        dataset = FramingArticleDataset(    
            data_dir=dataset_config['data_dir'],
            language=language,
            subtask=dataset_config['subtask'],
            train_split=evaluate_config['train_set'],
            eval_split=evaluate_config['eval_set'],
            load_preprocessed_units_of_analysis=preprocessing_config['load_preproc_input_data'],
            units_of_analysis_dir=os.path.join(dataset_config['data_dir'], 'preprocessed')
        )

        # If not extracted already, extract the different units of analysis
        if not preprocessing_config['load_preproc_input_data']:
            dataset.extract_all_units_of_analysis(nlp=nlp)

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

        # Vectorize input data
        unit_of_analysis = preprocessing_config['analysis_unit']
        X_train = bow_pipeline.pipeline.fit_transform(dataset.train_df[unit_of_analysis])
        X_test = bow_pipeline.pipeline.transform(dataset.eval_df[unit_of_analysis])

        # One hot encode the multilabel target
        mlb = MultiLabelBinarizer()
        mlb.fit([LABELS])
        y_train = mlb.transform(dataset.train_df.frames.str.split(','))

        # Tune models and get their predictions on the eval set
        for model_name in models_config['model_list']:
            notify_current_model_str = f"Currently tunning: {model_name}"
            print(notify_current_model_str)
            print("-" * len(notify_current_model_str))

            # Define model
            multilabel_cls = MultiLabelEstimator(
                model_name=model_name,
                base_estimator=estimators_config.MODEL_LIST[model_name]['model'],
                base_estimator_hyperparam_dist=estimators_config.MODEL_LIST[model_name]['hyperparam_space'],
                treat_labels_as_independent=models_config['mlb_cls_independent'],
                scoring_functions=scoring_functions
            )

            # Tune the model
            try:
                search_results = multilabel_cls.tune_model(
                    X=X_train,
                    y=y_train,
                    n_iterations=estimators_config.MODEL_LIST[model_name]['n_search_iter'],
                    n_folds=models_config['n_folds'],
                    ranking_score=models_config['ranking_score'],
                )

                # Evaluate the model on the eval set and store predictions
                best_model = search_results.best_estimator_
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Persist the best model
                dump(best_model, os.path.join(best_model_dir_path, f'{model_name}.joblib'))

                # Store the predictions
                pd.DataFrame(y_train_pred, columns=LABELS, index=dataset.train_df.index).to_csv(os.path.join(
                    predicted_instances_train_dir_path, f"{language}_{model_name}_y_{evaluate_config['train_set']}.csv"))

                pd.DataFrame(y_test_pred, columns=LABELS, index=dataset.eval_df.index).to_csv(os.path.join(
                    predicted_instances_test_dir_path, f"{language}_{model_name}_y_{evaluate_config['eval_set']}.csv"))

            except Exception as e:
                print(f'Error while trying to fit model: {model_name}')
                print(e)
                continue
