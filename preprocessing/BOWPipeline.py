import os
from typing import Tuple, List, Optional

import pandas as pd
import spacy
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from skopt.space import Space
from skopt.sampler import Lhs

from preprocessing.CorrelationFilter import CorrelationFilter


def map_hyperparams(hyperparam_basename: str, hyperparam_list: list[str]):
    for hyperparam in hyperparam_list:
        if hyperparam_basename in hyperparam:
            return hyperparam

    return None


def basic_tokenizing_and_cleaning(text: str, spacy_nlp_model: spacy.Language) -> List[str]:
    """
    Lemmatize, remove punctutation, and stopwords of a string
    :return:
    """
    return [token.lemma_ for token in spacy_nlp_model(text) if not token.is_punct and not token.is_stop]


class BOWPipeline:
    def __init__(self,
                 tokenizer: callable,
                 use_tfidf: bool = True,
                 min_df: float = 0.05,
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 1),
                 max_features: Optional[int] = 10000,
                 min_var: Optional[float] = 1e-4,
                 corr_threshold: Optional[float] = 0.9
                 ):
        self.use_tfidf = use_tfidf
        self.tokenizer = tokenizer
        self.vectorizer = TfidfVectorizer() if use_tfidf else CountVectorizer()
        self.vectorizer.set_params(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            max_features=max_features,
            tokenizer=tokenizer
        )

        self.pipeline = Pipeline([('vectorizer', self.vectorizer)])

        if min_var:
            self.add_low_var_threshold(min_var=min_var)
        else:
            self.min_var = None

        if corr_threshold:
            self.add_corr_filter(corr_threshold=corr_threshold)
        else:
            self.corr_threshold = None

    def add_low_var_threshold(self, min_var: float = 1e-4):
        self.min_var = min_var
        self.pipeline.steps.append(('low_var_filter', VarianceThreshold(min_var)))

    def add_corr_filter(self, corr_threshold: float = 0.9):
        self.corr_threshold = corr_threshold
        self.pipeline.steps.append(('corr_filter', CorrelationFilter(corr_threshold)))

    def sample_pipelines_from_hypeparameter_space(
            self,
            n_samples: int,
            min_df_range: Optional[Tuple[float, float]] = (0, 0.3),
            max_df_range: Optional[Tuple[float, float]] = (0.6, 1),
            max_features_range: Optional[Tuple[int, int]] = (200, 1000),
            **kwargs
    ):
        """
        Explore the space of hyperparameters using form of sampling that ensures we explore a large portion of the
        hyperparameter space.

        Note: Specify the exact name of the paramereter followed by _range: <parameter_name>_range and a tuple for it's range

        :param min_df_range:
        :param max_df_range:
        :param max_features_range:
        :return: iterator of pipelines over the sampled hyperparameters
        """
        # Merge list of hyperparameter ranges
        hyperparams_ranges_dict = {**kwargs, **{'min_df_range': min_df_range, 'max_df_range': max_df_range ,
                                         'max_features_range': max_features_range}}

        # Define hyperpameters that apply
        pipeline_params = [param for param in self.pipeline.get_params().keys() if len(param.split('__')) > 1]

        hyperparams_ranges_dict = {
            map_hyperparams(key.split('_range')[0], pipeline_params): value
            for key, value in hyperparams_ranges_dict.items()
            if map_hyperparams(key.split('_range')[0], pipeline_params) in pipeline_params and (
                    isinstance(value, tuple) or isinstance(value, list))
        }

        # Get the list of the ranges to sample it with LHS
        hyperparam_vars = [hyperparam_var for hyperparam_var in hyperparams_ranges_dict.values()]
        lhs = Lhs(criterion="maximin", iterations=10000)
        hyperparam_samples = lhs.generate(hyperparam_vars, n_samples)

        for hyperparams in hyperparam_samples:
            yield self.pipeline.set_params(
                **{hyperparam: hyperparam_value
                   for hyperparam, hyperparam_value in zip(hyperparams_ranges_dict.keys(), hyperparams)}
            )


if __name__ == "__main__":
    # Load example dataset
    DATA_DIR = os.path.join('data', 'preprocessed')
    en_train_df = pd.read_csv(os.path.join(DATA_DIR, 'input_en_train.csv'), index_col='id')

    # Use the pipeline on a dataset
    en_nlp = spacy.load('en_core_web_sm')
    preproc_pipeline = BOWPipeline(
        tokenizer=lambda string: basic_tokenizing_and_cleaning(string, spacy_nlp_model=en_nlp),
        use_tfidf=True,
        min_df=0.05,
        max_df=0.95,
        ngram_range=(1, 1),
        max_features=None,
        min_var=1e-3,
        corr_threshold=.9
    )

    pipeline_generator = preproc_pipeline.sample_pipelines_from_hypeparameter_space(
        3, corr_threshold_range=(0.8, 1), some_non_valid_hyperparam='sdglk')

    for i, pipeline_i in enumerate(pipeline_generator):
        print(f'pipeline_{i}')
        print(pipeline_i)

    train_df = preproc_pipeline.pipeline.fit_transform(en_train_df.title_and_first_paragraph)

