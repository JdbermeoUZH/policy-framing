import os
import glob
from typing import Tuple, Optional, List, Union

import spacy
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV, cross_validate, PredefinedSplit
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from preprocessing.InputDataset import FramingArticleDataset
from preprocessing.BOWPipeline import BOWPipeline, basic_tokenizing_and_cleaning

DEFAULT_SCORING_FUNCTIONS = ('f1_micro', 'f1_macro', 'accuracy', 'precision_micro',
                             'precision_macro', 'recall_micro', 'recall_macro')


def _recode_param_nams(hyper_param_dist: dict, estimator_type: str) -> dict:
    if estimator_type == 'chain':
        new_dict = {f'base_estimator__{k}': v for k, v in hyper_param_dist.items()}
    elif estimator_type == 'independent':
        new_dict = {f'estimator__{k}': v for k, v in hyper_param_dist.items()}
    else:
        new_dict = hyper_param_dist

    return new_dict


class MultiLabelEstimator:
    def __init__(
            self,
            model_name: str,
            base_estimator: ClassifierMixin,
            base_estimator_hyperparam_dist: dict,
            treat_labels_as_independent: bool = True,
            scoring_functions: Tuple[str, ...] = DEFAULT_SCORING_FUNCTIONS,
            random_seed: str = 123,
            wrap_mlb_clf: bool = True
    ):
        self.model_name = model_name
        self.base_estimator_name = base_estimator.__str__()

        if wrap_mlb_clf:
            self.estimator_type = 'independent' if treat_labels_as_independent else 'chain'
            self.multi_label_estimator = \
                MultiOutputClassifier(base_estimator) if treat_labels_as_independent else ClassifierChain(base_estimator)
        else:
            self.estimator_type = 'mutilabel'
            self.multi_label_estimator = base_estimator

        self.estimator_hyperparam_dists = _recode_param_nams(
            base_estimator_hyperparam_dist, self.estimator_type)

        self.scoring_functions = scoring_functions
        self.random_seed = random_seed

    def nested_cross_validation(
            self,
            X, y,
            k_outer: int = 10,
            hyperparam_samples_per_outer_fold: int = 10,
            k_inner: int = 5,
            ranking_score: str = 'f1_micro',
            scoring_functions: Optional[Union[Tuple[str], List[str]]] = None,
            shuffle_folds: bool = True,
            return_train_score: bool = False,
            n_jobs: int = -1,
            precomputed_outer_fold_dir: str = None,
            X_index = None
    ) -> dict:
        # Verify dimensions match
        assert X.shape[0] == y.shape[0]

        print(f"Running Nested Cross Validation for {self.base_estimator_name}")
        print(f"Total number of training runs is: "
              f"{k_outer * (hyperparam_samples_per_outer_fold * k_inner + 1)}")
        print(f"Hyper-param search in inner loop over: {self.estimator_hyperparam_dists}")
        scoring_functions = self.scoring_functions if not scoring_functions else scoring_functions

        # Define inner loop random search routine
        search_routine = RandomizedSearchCV(
            estimator=self.multi_label_estimator,
            param_distributions=self.estimator_hyperparam_dists,
            n_iter=hyperparam_samples_per_outer_fold,
            cv=MultilabelStratifiedKFold(n_splits=k_inner, shuffle=shuffle_folds, random_state=self.random_seed),
            scoring=scoring_functions,
            refit=ranking_score,
            return_train_score=return_train_score,
            n_jobs=n_jobs
        )

        outer_fold_cv_splitter = self._get_outer_fold_cv_splitter(
            X_index, k_outer, precomputed_outer_fold_dir, shuffle_folds)

        # Define Outer-loop routine and execute it
        nested_cv_results = cross_validate(
            search_routine,
            X, y,
            scoring=scoring_functions,
            cv=outer_fold_cv_splitter,
            return_estimator=True,
            return_train_score=return_train_score,
            n_jobs=n_jobs
        )

        return nested_cv_results

    def _get_outer_fold_cv_splitter(self, X_index, k_outer: int, precomputed_outer_fold_dir: str, shuffle_folds):
        if precomputed_outer_fold_dir is not None:
            idx_fold_membership = np.full(X_index.shape, -1)

            for test_fold_filepath in glob.glob(os.path.join(precomputed_outer_fold_dir, '*')):
                # Load the datasets for each fold and create the id for each fold group
                fold_i = int(test_fold_filepath.split('_')[-2])

                # Fold numbering started at 1
                fold_i -= 1

                # Load indexes
                fold_i_index = pd.read_csv(test_fold_filepath)[['id']].set_index('id').index

                idx_fold_membership = np.where(X_index.isin(fold_i_index), fold_i, idx_fold_membership)

            outer_fold_cv_splitter = PredefinedSplit(idx_fold_membership)
        else:
            outer_fold_cv_splitter = MultilabelStratifiedKFold(n_splits=k_outer, shuffle=shuffle_folds,
                                                               random_state=self.random_seed)
        return outer_fold_cv_splitter

    def cross_validation(
            self,
            X, y,
            k_outer: int = 10,
            scoring_functions: Optional[Union[Tuple[str], List[str]]] = None,
            shuffle_folds: bool = True,
            return_train_score: bool = False,
            n_jobs: int = -1,
            precomputed_outer_fold_dir: str = None,
            X_index=None
    ) -> dict:
        # If no soring functions are specified, use default ones of the object
        scoring_functions = self.scoring_functions if not scoring_functions else scoring_functions

        cv_splitter = self._get_outer_fold_cv_splitter(
            X_index, k_outer, precomputed_outer_fold_dir, shuffle_folds)

        cv_results = cross_validate(
            self.multi_label_estimator,
            X, y,
            scoring=scoring_functions,
            cv=cv_splitter,
            return_estimator=False,
            return_train_score=return_train_score,
            n_jobs=n_jobs
        )

        return cv_results

    def tune_model(
            self,
            X, y,
            n_iterations: int = 10,
            n_folds: int = 3,
            ranking_score: str = 'f1_micro',
            scoring_functions: Optional[Union[Tuple[str], List[str]]] = None,
            shuffle_folds: bool = True,
            return_train_score: bool = False,
            n_jobs: int = -1
    ):
        # Define inner loop random search routine
        rs_cv = RandomizedSearchCV(
            estimator=self.multi_label_estimator,
            param_distributions=self.estimator_hyperparam_dists,
            n_iter=n_iterations,
            cv=MultilabelStratifiedKFold(n_splits=n_folds, shuffle=shuffle_folds, random_state=self.random_seed),
            scoring=scoring_functions,
            refit=ranking_score,
            return_train_score=return_train_score,
            n_jobs=n_jobs
        )

        search_results = rs_cv.fit(X, y)

        return search_results

    def get_stratified_splits(
            self,
            X, y,
            folds: int = 3,
            shuffle=True
    ):
        mskf = MultilabelStratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=self.random_seed)
        return mskf.split(X, y)

    def get_base_estimator_name(self) -> str:
        if 'XGB' in self.base_estimator_name:
            return self.base_estimator_name[:15]
        else:
            return self.base_estimator_name

    def get_model_name(self) -> str:
        return self.model_name

    def get_multilabel_model_type(self) -> str:
        return self.estimator_type


def main() -> dict:
    from sklearn.svm import SVC
    from sklearn.preprocessing import MultiLabelBinarizer
    DATA_DIR = os.path.join('..', 'data')

    # Test with dataset in english of subtask 2

    # Load the data
    en_train = FramingArticleDataset(data_dir=DATA_DIR, language='ge', subtask=2, train_split='train',
                                     load_preprocessed_units_of_analysis=True,
                                     units_of_analysis_dir=os.path.join(DATA_DIR, 'preprocessed'))

    # Preprocess text data
    en_nlp = spacy.load('en_core_web_sm')
    vectorizing_pipeline = BOWPipeline(
        tokenizer=lambda string: basic_tokenizing_and_cleaning(string, spacy_nlp_model=en_nlp),
        use_tfidf=True,
        min_df=0.05,
        max_df=0.95,
        ngram_range=(1, 1),
        max_features=None,
        min_var=1e-3,
        corr_threshold=.9
    )

    X_train = vectorizing_pipeline.pipeline.fit_transform(en_train.train_df.title_and_first_paragraph)

    # Preprocess labels
    labels = ('fairness_and_equality', 'security_and_defense', 'crime_and_punishment', 'morality',
              'policy_prescription_and_evaluation',
              'capacity_and_resources', 'economic', 'cultural_identity', 'health_and_safety', 'quality_of_life',
              'legality_constitutionality_and_jurisprudence',
              'political', 'public_opinion', 'external_regulation_and_reputation')
    mlb = MultiLabelBinarizer()
    mlb.fit([labels])
    y_train = mlb.transform(en_train.train_df.frames.str.lower().str.split(','))

    # Run the nested cross validation estimator for SVM with RBF kernel
    svm = SVC()
    hyperparam_space = {
        'estimator__C': loguniform(1e-2, 1e3),
        'estimator__gamma': loguniform(1e-4, 1e-1)
    }

    multilabel_cls = MultiLabelEstimator(
        base_estimator=svm,
        base_estimator_hyperparam_dist=hyperparam_space,
        treat_labels_as_independent=True
    )

    results_nested_cv = multilabel_cls.nested_cross_validation(
        X=X_train, y=y_train,
        k_outer=10,
        hyperparam_samples_per_outer_fold=3,
        k_inner=5,
        ranking_score='f1_micro',
        shuffle_folds=True
    )

    default_model_results_cv = multilabel_cls.cross_validation(
        X=X_train, y=y_train,
        k_outer=3,
        shuffle_folds=True
    )

    main_objects_params = {
        'language': 'en',
        'unit_of_analysis': 'title_and_first_paragraph',
        'spacy_model_used': f'{en_nlp.meta["lang"]}_{en_nlp.meta["name"]}',
        'preprocessing_pipeline': vectorizing_pipeline,
        'estimator': multilabel_cls,
        'nested_cv_results': results_nested_cv
    }

    return main_objects_params


if __name__ == '__main__':
    main_results_dict = main()
    results_nested_cv_ = main_results_dict['nested_cv_results']
