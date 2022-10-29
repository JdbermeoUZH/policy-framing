import os
from typing import Tuple, Optional, List, Union

import spacy
from scipy.stats import loguniform
from sklearn.base import ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain

from preprocessing.InputDataset import FramingArticleDataset
from preprocessing.BOWPipeline import BOWPipeline, basic_tokenizing_and_cleaning

DEFAULT_SCORING_FUNCTIONS = ('f1_micro', 'f1_macro', 'accuracy', 'precision_micro',
                             'precision_macro', 'recall_micro', 'recall_macro')


class MultiLabelEstimator:
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

    def get_model_name(self) -> str:
        return self.base_estimator_name

    def get_multilabel_model_type(self) -> str:
        return self.estimator_type


def main() -> dict:
    from sklearn.svm import SVC
    from sklearn.preprocessing import MultiLabelBinarizer
    DATA_DIR = os.path.join('..', 'data')

    # Test with dataset in english of subtask 2

    # Load the data
    en_train = FramingArticleDataset(data_dir=DATA_DIR, language='en', subtask=2, split='train',
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

    X_train = vectorizing_pipeline.pipeline.fit_transform(en_train.df.title_and_first_paragraph)

    # Preprocess labels
    labels = ('fairness_and_equality', 'security_and_defense', 'crime_and_punishment', 'morality',
              'policy_prescription_and_evaluation',
              'capacity_and_resources', 'economic', 'cultural_identity', 'health_and_safety', 'quality_of_life',
              'legality_constitutionality_and_jurisprudence',
              'political', 'public_opinion', 'external_regulation_and_reputation')
    mlb = MultiLabelBinarizer()
    mlb.fit([labels])
    y_tran = y_train = mlb.transform(en_train.df.frames.str.lower().str.split(','))

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
        k_outer=3,
        hyperparam_samples_per_outer_fold=3,
        k_inner=2,
        ranking_score='f1_micro'
    )

    main_objects_params = {
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
