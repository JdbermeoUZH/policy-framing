import os
import yaml
import pprint
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
from utils.constants import LABELS, UNITS_OF_ANALYSES, SPACY_MODELS, LANGUAGES
from utils.metrics_config import compute_multi_label_metrics
compute_multi_label_metrics
def parse_arguments_and_load_config_file() -> Tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(description='Subtask-2')
    parser.add_argument('--config_path_yaml', type=str)
    parser.add_argument('--languages', type=str, default=None, nargs="*")
    parser.add_argument('--analysis_unit', type=str, default=None, nargs="*")
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--model_list', type=str, default=None, nargs="*")
    parser.add_argument('--mlb_cls_independent', type=int, default=None)
    parser.add_argument('--default_params', type=int, default=None)

    #tune_preprocessing_params
    #n_samples
    arguments = parser.parse_args()

    # Load parameters of configuration file
    with open(arguments.config_path_yaml, "r") as stream:
        try:
            yaml_config_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    if arguments.languages is not None:
        yaml_config_params['dataset']['languages'] = arguments.languages

    if arguments.analysis_unit is not None:
        yaml_config_params['preprocessing']['analysis_unit'] = arguments.analysis_unit

    if arguments.n_samples is not None:
        yaml_config_params['preprocessing']['param_search']['n_samples'] = arguments.n_samples

    if arguments.model_list is not None:
        yaml_config_params['training']['model_list'] = arguments.model_list

    if arguments.mlb_cls_independent is not None:
        yaml_config_params['training']['mlb_cls_independent'] = arguments.mlb_cls_independent == 1

    if arguments.mlb_cls_independent is not None:
        yaml_config_params['training']['default_params'] = arguments.default_params == 1

    print('command line args:')
    pprint.pprint(arguments)
    print('\n\n')

    print('config args: ')
    pprint.pprint(yaml_config_params)
    print('\n\n')

    return arguments, yaml_config_params


if __name__ == "__main__":
    # Load script arguments and configuration file
    args, config = parse_arguments_and_load_config_file()

    run_config = config['run']

    dataset_config = config['dataset']

    preprocessing_config = config['preprocessing']
    preprocessing_params_config = import_module(preprocessing_config['preprocessing_hyperparam_module']).PREPROCESSING

    train_config = config['training']
    estimators_config = import_module(train_config['model_hyperparam_module'])
    model_list = train_config['model_list'] if train_config['model_list'] not in ['all', ['all']] \
        else list(estimators_config.MODEL_LIST.keys())

    output_config = config['output']

    output_dir = os.path.join(*output_config['output_dir'])

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'performance'), exist_ok=True)

    predicted_instances_dir_path = os.path.join(output_dir, 'predicted_instances')
    os.makedirs(predicted_instances_dir_path, exist_ok=True)

    if run_config['supress_warnings']:
        helper_fns.supress_all_warnings()

    # Iterate over the datasets specified
    for language in dataset_config['languages']:
        # Load the train and test splits
        data_dir = os.path.join(*dataset_config['data_dir'])

        train_df = pd.read_csv(os.path.join(data_dir, f'final_evaluation_train_{language}.csv'), index_col='id')
        test_df = pd.read_csv(os.path.join(data_dir, f'final_evaluation_test_{language}.csv'), index_col='id')

        # Tune and measure performance for each unit of analysis listed
        units_of_analysis = UNITS_OF_ANALYSES if preprocessing_config['analysis_unit'][0] == 'all' \
            else [unit for unit in preprocessing_config['analysis_unit'] if unit in UNITS_OF_ANALYSES]

        for unit_of_analysis in units_of_analysis:
            metrics = []

            # Create dirs where outputs will be stored
            if output_config['store_best_models']:
                best_model_dir_path = os.path.join(output_dir, 'best_models', language, unit_of_analysis)
                os.makedirs(best_model_dir_path, exist_ok=True)

            if output_config['store_predictions']:
                predicted_instances_train_dir_path = os.path.join(predicted_instances_dir_path, "train", language, unit_of_analysis)
                os.makedirs(predicted_instances_train_dir_path, exist_ok=True)

                predicted_instances_test_dir_path = os.path.join(predicted_instances_dir_path, "test", language, unit_of_analysis)
                os.makedirs(predicted_instances_test_dir_path, exist_ok=True)

            # Define the vectorizing pipeline(s) to use
            if preprocessing_config['use_same_params_across_units']:
                preprocessing_params = preprocessing_params_config['all']['fixed_params']
                preprocessing_params_search_space = preprocessing_params_config['all']['param_search']

            else:
                preprocessing_params = preprocessing_params_config[unit_of_analysis]['fixed_params']
                preprocessing_params_search_space = preprocessing_params_config[unit_of_analysis]['param_search']

            nlp = spacy.load(SPACY_MODELS[language][preprocessing_config['spacy_model_size']])

            bow_pipeline = BOWPipeline(
                tokenizer=lambda string: basic_tokenizing_and_cleaning(string, spacy_nlp_model=nlp),
                use_tfidf=preprocessing_params['use_tfidf'],
                min_df=preprocessing_params['min_df'],
                max_df=preprocessing_params['max_df'],
                max_features=preprocessing_params['max_features'],
                ngram_range=tuple(preprocessing_params['ngram_range']),
                min_var=preprocessing_params['min_var'],
                corr_threshold=preprocessing_params['corr_threshold']
            )

            # Vectorize input data
            X_train = bow_pipeline.pipeline.fit_transform(train_df[unit_of_analysis])
            X_test = bow_pipeline.pipeline.transform(test_df[unit_of_analysis])

            # One hot encode the multilabel target
            mlb = MultiLabelBinarizer()
            mlb.fit([LABELS])
            y_train = mlb.transform(train_df.frames.str.split(','))
            y_test = mlb.transform(test_df.frames.str.split(','))

            # Tune models and get their predictions on the eval set
            for model_name in model_list:
                notify_current_model_str = f"Currently tunning: {model_name}"
                print(notify_current_model_str)
                print("-" * len(notify_current_model_str))

                # Define model
                multilabel_cls = MultiLabelEstimator(
                    model_name=model_name,
                    base_estimator=estimators_config.MODEL_LIST[model_name]['model'],
                    base_estimator_hyperparam_dist=estimators_config.MODEL_LIST[model_name]['hyperparam_space'],
                    treat_labels_as_independent=train_config['mlb_cls_independent'],
                    scoring_functions=scoring_functions
                )

                # Tune the model
                try:
                    search_results = multilabel_cls.tune_model(
                        X=X_train,
                        y=y_train,
                        n_iterations=estimators_config.MODEL_LIST[model_name]['n_search_iter'],
                        n_folds=train_config['n_folds_tunning_eval'],
                        ranking_score=train_config['ranking_metric'],
                    )
                except Exception as e:
                    print(f'Error while trying to fit model: {model_name}')
                    print(e)
                    continue

                # Evaluate the model on the eval set and store predictions
                best_model = search_results.best_estimator_
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                if output_config['store_best_models']:
                    # Persist the best model
                    dump(best_model, os.path.join(best_model_dir_path, f'{model_name}.joblib'))

                if output_config['store_predictions']:
                    # Store the predictions
                    pd.DataFrame(y_train_pred, columns=LABELS, index=train_df.index)\
                        .to_csv(os.path.join(predicted_instances_train_dir_path, f"{language}_{model_name}_y_train.csv"))

                    pd.DataFrame(y_test_pred, columns=LABELS, index=test_df.index).to_csv(os.path.join(
                        predicted_instances_test_dir_path, f"{language}_{model_name}_y_test.csv"))

                # Calculate metrics of the model
                metrics_train = compute_multi_label_metrics(y_train, y_train_pred, prefix='train')
                metrics_test = compute_multi_label_metrics(y_test, y_test_pred, prefix='test')

                # Add metrics of the model
                metrics.append(
                    {
                        'language': language, 'unit_of_analysis': unit_of_analysis,
                        'model_type': estimators_config.MODEL_LIST[model_name]['model_type'],
                        'model_subtype': estimators_config.MODEL_LIST[model_name]['model_subtype'],
                        'model_name': model_name, **metrics_train, **metrics_test
                    }
                )

            # Save performance metrics
            metrics_filename = f'{output_config["metric_file_prefix"]}_{language}_{unit_of_analysis}_metrics.csv'
            pd.DataFrame(metrics)\
                .set_index(['language', 'unit_of_analysis', 'model_type', 'model_subtype']).sort_index()\
                .to_csv(os.path.join(output_dir, 'performance', metrics_filename))
