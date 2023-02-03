import os
import shutil
from typing import Optional

import mlflow
import pandas as pd
from mlflow import log_metric, log_param, log_artifact, set_tracking_uri, create_experiment, get_experiment_by_name
from sklearn.pipeline import Pipeline

from preprocessing.BOWPipeline import BOWPipeline
from training import MultiLabelEstimator


class Logger:
    def __init__(
            self,
            logging_dir: str,
            experiment_name: str = "test_run",
            rewrite_experiment: bool = False,
            logging_level: str = 'outer_cv'
    ):
        self.logging_dir = os.path.abspath(logging_dir)
        self.experiment_name = experiment_name
        self.logging_level = logging_level
        set_tracking_uri(self.logging_dir)

        past_test_exp = get_experiment_by_name(name=experiment_name)

        if rewrite_experiment and past_test_exp is not None:
            # Delete experiment if it has been ran before
            past_test_exp_id = past_test_exp.experiment_id
            mlflow.delete_experiment(past_test_exp_id)
            past_test_exp = None
            exp_trash_dir = os.path.join(logging_dir, '.trash', str(past_test_exp_id))

            if os.path.exists(exp_trash_dir):
                shutil.rmtree(exp_trash_dir)

        # Try to
        if past_test_exp is None:
            self.experiment_id = create_experiment(self.experiment_name)
        else:
            self.experiment_id = past_test_exp.experiment_id

        mlflow.set_experiment(experiment_id=self.experiment_id)

    def _log_preprocessing_params(
            self,
            unit_of_analysis: str,
            spacy_model_used: str,
            preprocessing_pipeline: Pipeline
    ) -> None:
        log_param('n_features', len(preprocessing_pipeline.get_feature_names_out()))
        log_param('unit_of_analysis', unit_of_analysis)
        log_param('spacy_model_used', spacy_model_used)

        is_count_vectorizer = type(preprocessing_pipeline['vectorizer']).__name__ == 'CountVectorizer'
        log_param('is_count_vectorizer', is_count_vectorizer)
        log_param('min_df', preprocessing_pipeline['vectorizer'].min_df)
        log_param('max_df', preprocessing_pipeline['vectorizer'].max_df)
        log_param('vectorizer_max_features', preprocessing_pipeline['vectorizer'].max_features)
        log_param('n_gram_range_start', preprocessing_pipeline['vectorizer'].ngram_range[0])
        log_param('n_gram_range_end', preprocessing_pipeline['vectorizer'].ngram_range[1])

        if not is_count_vectorizer:
            log_param('tfidf', preprocessing_pipeline['vectorizer'].use_idf)

        log_param('corr_threshold', preprocessing_pipeline['corr_filter'].corr_threshold)

        log_param('min_var', preprocessing_pipeline['low_var_filter'].threshold)

    def _log_base_estimator_info(
            self,
            estimator: MultiLabelEstimator,
            model_type: str,
            model_subtype: str
    ) -> None:
        log_param('model_name', estimator.get_model_name())
        log_param('model_type', model_type)
        log_param('model_subtype', model_subtype)
        log_param('base_estimator_name', estimator.get_base_estimator_name())
        log_param('multilabel_type', estimator.get_multilabel_model_type())

    def log_model_wide_performance(
            self,
            language: str,
            unit_of_analysis: str,
            spacy_model_used: str,
            preprocessing_pipeline: Pipeline,
            estimator: MultiLabelEstimator,
            model_type: str,
            model_subtype: str,
            cv_results: dict,
            hyperparam_distrs_filepath: Optional[str] = None
    ) -> None:
        with mlflow.start_run(experiment_id=self.experiment_id):
            # Log preprocessing params
            log_param('language', language)
            self._log_preprocessing_params(unit_of_analysis=unit_of_analysis, spacy_model_used=spacy_model_used,
                                           preprocessing_pipeline=preprocessing_pipeline)
            # Log model information
            self._log_base_estimator_info(estimator=estimator, model_type=model_type, model_subtype=model_subtype)

            # Log model wide performance
            log_param('analysis_level', 'model_wide')
            for metric in [key for key in cv_results.keys() if any(x in key for x in ['train', 'test'])]:
                log_metric(f'{metric}_mean', cv_results[metric].mean())
                log_metric(f'{metric}_std', cv_results[metric].std())

            # For nested cross validation, log the hyperparameters space used
            if hyperparam_distrs_filepath:
                log_artifact(hyperparam_distrs_filepath)
        mlflow.end_run()

    def log_hyper_param_performance_outer_fold(
            self,
            language: str,
            unit_of_analysis: str,
            spacy_model_used: str,
            preprocessing_pipeline: Pipeline,
            estimator: MultiLabelEstimator,
            cv_results: dict,
            model_type: str,
            model_subtype: str
    ) -> None:
        n_outer_fold = len(cv_results['estimator'])

        for outer_fold_i in range(n_outer_fold):

            with mlflow.start_run(experiment_id=self.experiment_id):
                # Log preprocessing params
                log_param('language', language)
                self._log_preprocessing_params(unit_of_analysis=unit_of_analysis, spacy_model_used=spacy_model_used,
                                               preprocessing_pipeline=preprocessing_pipeline)
                # Log model information
                self._log_base_estimator_info(estimator=estimator, model_type=model_type, model_subtype=model_subtype)

                log_param('analysis_level', 'outer_cv')

                # Log the best hyper-params found for the outer fold_i values
                for key, value in cv_results['estimator'][outer_fold_i].best_params_.items():
                    log_param(key.split('__')[-1], value)

                # Log the metrics
                for metric in [key for key in cv_results.keys() if any(x in key for x in ['train', 'test'])]:
                    log_metric(metric, cv_results[metric][outer_fold_i])

            mlflow.end_run()

    def log_hyper_param_performance_inner_fold(
            self,
            language: str,
            unit_of_analysis: str,
            spacy_model_used: str,
            preprocessing_pipeline: Pipeline,
            estimator: MultiLabelEstimator,
            cv_results: dict,
            model_type: str,
            model_subtype: str
    ) -> None:

        # Average performance across inner folds of the different hyper-param samples
        outer_fold_hyperparam_searches = cv_results['estimator']

        for hyperparam_search in outer_fold_hyperparam_searches:
            results_df = pd.DataFrame(hyperparam_search.cv_results_)
            metrics_mean_cols = [col for col in results_df.columns
                                 if any(x in col for x in ['mean_train', 'mean_test'])]
            metrics_std_cols = [col for col in results_df.columns
                                if any(x in col for x in ['std_train', 'std_test'])]

            for sample_i, hyperparam_sample_results in results_df.iterrows():
                with mlflow.start_run(experiment_id=self.experiment_id, nested=True):
                    log_param('language', language)
                    # Log preprocessing params
                    self._log_preprocessing_params(unit_of_analysis=unit_of_analysis, spacy_model_used=spacy_model_used,
                                                   preprocessing_pipeline=preprocessing_pipeline)
                    # Log model information
                    self._log_base_estimator_info(estimator=estimator, model_type=model_type, model_subtype=model_subtype)

                    log_param('analysis_level', 'inner_cv')

                    # Log the hyperparameters
                    for param, param_value in hyperparam_sample_results.params.items():
                        log_param(param.split('__')[-1], param_value)

                    # Log the metrics
                    for mean_metric, std_metric in zip(metrics_mean_cols, metrics_std_cols):
                        log_metric(mean_metric, hyperparam_sample_results[mean_metric])
                        log_metric(std_metric, hyperparam_sample_results[std_metric])

                mlflow.end_run()

    def log_metrics(
            self,
            cv_results: dict,
            hyperparam_distrs_filepath: str,
            **kwargs
            ):

        # Log the results of the experiment
        if self.logging_level in ['outer_cv', 'inner_cv', 'model_wide']:
            self.log_model_wide_performance(
                cv_results=cv_results,
                hyperparam_distrs_filepath=hyperparam_distrs_filepath,
                **kwargs
            )
        if self.logging_level in ['outer_cv', 'inner_cv']:
            self.log_hyper_param_performance_outer_fold(cv_results=cv_results, **kwargs)

        if self.logging_level in ['inner_cv']:
            self.log_hyper_param_performance_inner_fold(cv_results=cv_results, **kwargs)


if __name__ == '__main__':
    logging_path = os.path.join('..', 'mlruns')

    # Run the experiment
    multilabel_nested_cv_objects = MultiLabelEstimator.main()

    # Log the results of the experiment
    metric_logger = Logger(logging_path, experiment_name='test_run', rewrite_experiment=False)

    # Log the results of the experiment
    metric_logger.log_model_wide_performance(**multilabel_nested_cv_objects)
    metric_logger.log_hyper_param_performance_outer_fold(**multilabel_nested_cv_objects)
    metric_logger.log_hyper_param_performance_inner_fold(**multilabel_nested_cv_objects)
