import os
from typing import Tuple, List, Optional

import pandas as pd
import spacy
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from preprocessing.CorrelationFilter import CorrelationFilter


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
                 min_var: float = 1e-4,
                 corr_threshold: float = 0.9
                 ):
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
        max_features=None
    )

    preproc_pipeline.add_low_var_threshold(min_var=1e-3)
    preproc_pipeline.add_corr_filter(corr_threshold=0.9)
    train_df = preproc_pipeline.pipeline.fit_transform(en_train_df.title_and_first_paragraph)
