import os
from typing import Tuple, Optional, List, Union

import pandas as pd
import spacy
from scipy.stats import loguniform
from sklearn.base import ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain

from preprocessing.InputDataset import FramingArticleDataset
from preprocessing.BOWPipeline import BOWPipeline, basic_tokenizing_and_cleaning

DEFAULT_SCORING_FUNCTIONS = ('f1_micro', 'f1_macro', 'accuracy', 'precision_micro',
                             'precision_macro', 'recall_micro', 'recall_macro')


class Estimator:

    def __init__(
            self,
            base_estimator: ClassifierMixin,
            base_estimator_hyperparam_dist: dict,
            treat_labels_as_independent: bool = True,
            scoring_functions: Tuple[str] = DEFAULT_SCORING_FUNCTIONS
    ):
        self.estimator_type = 'independent' if treat_labels_as_independent else 'chain'
        self.base_estimator_name = type(base_estimator).__name__
        self.multi_label_estimator = \
            MultiOutputClassifier(base_estimator) if treat_labels_as_independent else ClassifierChain(base_estimator)
        self.estimator_hyperparam_dists = base_estimator_hyperparam_dist
        self.scoring_functions = scoring_functions

    def nested_cross_validation(
            self,
            X, y,
            k_outer: int = 10,
            hyperparam_samples_per_outer_fold: int = 10,
            k_inner: int = 5,
            ranking_score: str = 'f1_micro',
            scoring_functions: Optional[Union[Tuple[str], List[str]]] = None
    ) -> dict:
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
            cv=k_inner,
            scoring=scoring_functions,
            refit=ranking_score
        )

        # Define Outer-loop routine and execute it
        nested_cv_results = cross_validate(
            search_routine,
            X, y,
            scoring=scoring_functions,
            cv=k_outer,
            return_estimator=True
        )

        return nested_cv_results


if __name__ == '__main__':
    DATA_DIR = os.path.join('..', 'data')
    print('a')
    # Test with dataset in english of subtask 2
    en_train = FramingArticleDataset(data_dir=DATA_DIR, language='en', subtask=2, split='train',
                                     load_preprocessed_units_of_analysis=True,
                                     units_of_analysis_dir=os.path.join(DATA_DIR, 'preprocessed'))
    print('b')
    vectorizing_pipeline = BOWPipeline(
        tokenizer=lambda string: basic_tokenizing_and_cleaning(string, spacy_nlp_model=spacy.load('en_core_web_sm')),
        use_tfidf=True,
        min_df=0.05,
        max_df=0.95,
        ngram_range=(1, 1),
        max_features=1000,
        min_var=1e-3,
        corr_threshold=0.9
    )

    print('c')
    X_train = vectorizing_pipeline.pipeline.fit_transform(en_train.df.title_and_first_paragraph)
    print('d')
    y_train = FramingArticleDataset.vectorize_multilabels()
