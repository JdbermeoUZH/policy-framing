import os
from typing import Optional

import pandas as pd
import spacy
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer


"""
Helper Functions
"""


def extract_n_sentences(text: str, nlp: spacy.Language, n_sentences: int, break_pattern: str = '\n') -> str:
    sentence_list = list(nlp(text).sents)
    sentence_list = sentence_list[: min(len(sentence_list), n_sentences)]

    # Add an additional pattern to split up the textract_n_sentences
    sent_list = []
    for sentence in sentence_list:
        sent_list += [sentence for sentence in sentence.text.split(break_pattern) if
                      not sentence.isspace() and len(sentence) != 0]

    return ' '.join(sent_list[: min(len(sent_list), n_sentences)])


"""
Classes to represent task 3 datasets
"""


def _create_article_dataframe(article_dir: str) -> pd.DataFrame:
    text = []

    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(article_dir))):
        id_, txt = fil[7:].split('.')[0], \
                   open(os.path.join(article_dir, fil), 'r', encoding='utf-8').read()
        text.append((id_, txt))

    return pd.DataFrame(text, columns=['id', 'raw_text']) \
        .astype({'id': 'int', 'raw_text': 'str'}) \
        .set_index('id')


def _separate_title_content(article_df: pd.DataFrame, remove_raw_data_col: bool = False) -> pd.DataFrame:
    """
       1) Remove trailing whitespaces and breaklines
       2) Separate on single breakline "\n". No titles had single breaklines inside the title (before the '/n/n')
       3) Remove trailing whitespaces and breaklines for title and content again
    :param remove_raw_data_col:
    :return:
    """

    article_df[['title', 'content']] = article_df.pipe(lambda df: df.raw_text.str.strip()) \
                                           .pipe(lambda str_series: str_series.str.split('\n', 1, expand=True)) \
                                           .pipe(lambda df: df.apply(lambda series: series.str.strip()))

    if remove_raw_data_col:
        article_df.drop('raw_text', axis=1, inplace=True)

    return article_df


class BaseArticleDataset:
    def __init__(self, data_dir: str = 'data', language: str = 'en', subtask: int = 2, train_split: str = 'train',
                 eval_split: str = 'test', load_preprocessed: bool = False):
        self.language = language
        self.subtask = subtask
        self.split = train_split
        self.train_articles_dir = os.path.join(data_dir, language, f'{train_split}-articles-subtask-{subtask}')
        self.train_label_filepath = os.path.join(data_dir, language, f'{train_split}-labels-subtask-{subtask}.txt')
        self.eval_articles_dir = os.path.join(data_dir, language, f'{eval_split}-articles-subtask-{subtask}')
        self.eval_label_filepath = None

        if eval_split in ['train', 'dev']:
            self.eval_label_filepath = os.path.join(data_dir, language, f'{eval_split}-labels-subtask-{subtask}.txt')

        if not load_preprocessed:
            self.train_df = _create_article_dataframe(self.train_articles_dir)
            self.eval_df = _create_article_dataframe(self.eval_articles_dir)

    def __str__(self):
        return self.train_df.__str__()

    def __repr__(self):
        return self.train_df.__repr__()


def _extract_title_and_first_n_sentences(df: pd.DataFrame, nlp: spacy.Language, n_sentences: int) -> pd.DataFrame:
    if 'title' not in df.columns:
        _separate_title_content()

    df[f'title_and_{n_sentences}_sentences'] = \
        df.title + '\n' + \
        df.content.map(lambda content: extract_n_sentences(
            content, nlp, n_sentences=n_sentences, break_pattern='\n'))

    return df


def _extract_title_and_first_paragraph(
        df: pd.DataFrame, nlp: spacy.Language,
        min_token_paragraph: int = 40,  min_sentences_paragraph: int = 1) -> pd.DataFrame:

    first_paragraph = df.content.str.split('\n').str[0]
    second_paragraph = df.content.str.split('\n').str[1]

    # If there is only one sentence in the first paragraph or its length is under 40 tokens,
    # merge it with the second paragraph
    merge_first_two_paragraphs = \
        [len(processed_str) <= min_token_paragraph or len(list(processed_str.sents)) <= min_sentences_paragraph
         for processed_str in nlp.pipe(first_paragraph.tolist())]

    first_paragraph = (first_paragraph + ' ' + second_paragraph).where(merge_first_two_paragraphs, first_paragraph)

    df[f'title_and_first_paragraph'] = \
        df.title + '\n' + first_paragraph

    return df


def _extract_title_and_first_sentence_each_paragraph(df: pd.DataFrame, nlp: spacy.Language) -> pd.DataFrame:
    paragraphs_per_doc = df.content.str.split('\n')

    df[f'title_and_first_sentence_each_paragraph'] = paragraphs_per_doc.map(lambda paragraphs: ' '.join(
        [next(nlp(paragraph).sents).text for paragraph in paragraphs if paragraph.strip() != '']))

    return df


class FramingArticleDataset(BaseArticleDataset):
    labels = ('fairness_and_equality', 'security_and_defense', 'crime_and_punishment', 'morality',
              'policy_prescription_and_evaluation', 'capacity_and_resources', 'economic', 'cultural_identity',
              'health_and_safety', 'quality_of_life', 'legality_constitutionality_and_jurisprudence',
              'political', 'public_opinion', 'external_regulation_and_reputation')
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit([labels])

    def __init__(self, data_dir: str = 'data', language: str = 'en', subtask: int = 2, train_split: str = 'train',
                 eval_split: str = 'test', load_preprocessed_units_of_analysis: bool = False,
                 units_of_analysis_dir: str = os.path.join('data', 'preprocessed'),
                 units_of_analysis_format: str = 'csv',
                 remove_duplicates: bool = True):

        if load_preprocessed_units_of_analysis:
            super().__init__(data_dir=data_dir, language=language, subtask=subtask, train_split=train_split,
                             load_preprocessed=load_preprocessed_units_of_analysis)
            self.train_df = pd.read_csv(
                os.path.join(units_of_analysis_dir, f'input_{language}_{train_split}.{units_of_analysis_format}'),
                index_col='id'
            )

            self.eval_df = pd.read_csv(
                os.path.join(units_of_analysis_dir, f'input_{language}_{eval_split}.{units_of_analysis_format}'),
                index_col='id'
            )
        else:
            if train_split in ['train', 'dev']:
                super().__init__(data_dir=data_dir, language=language, subtask=subtask, train_split=train_split,
                                 eval_split=eval_split, load_preprocessed=load_preprocessed_units_of_analysis)
                self._add_labels()
                if remove_duplicates:
                    self._remove_duplicates()

            elif train_split == 'train_and_dev':
                # Load train data
                super().__init__(data_dir=data_dir, language=language, subtask=subtask, train_split='train',
                                 eval_split=eval_split, load_preprocessed=load_preprocessed_units_of_analysis)
                self._add_labels()
                if remove_duplicates:
                    self._remove_duplicates()
                train_df = self.train_df.copy()

                # Load dev data
                super().__init__(data_dir=data_dir, language=language, subtask=subtask, train_split='dev',
                                 eval_split=eval_split, load_preprocessed=load_preprocessed_units_of_analysis)
                self._add_labels()
                self.train_df = pd.concat([train_df, self.train_df])

            self.separate_title_content()

    def separate_title_content(self):
        self.train_df = _separate_title_content(self.train_df)
        self.eval_df = _separate_title_content(self.eval_df)

    def _add_labels(self) -> None:
        # MAKE LABEL DATAFRAME
        train_labels = pd.read_csv(self.train_label_filepath, sep='\t', header=None, names=['id', 'frames'],
                             dtype={'id': 'int', 'frames': 'str'}) \
            .set_index('id')
        self.train_df = train_labels.join(self.train_df)

        if self.eval_label_filepath is not None:
            eval_labels = pd.read_csv(self.eval_label_filepath, sep='\t', header=None, names=['id', 'frames'],
                        dtype={'id': 'int', 'frames': 'str'}) \
                .set_index('id')
            self.eval_df = eval_labels.join(self.eval_df)

    def _remove_duplicates(self):
        """
        Manually identified duplicate documents in the trainset
        :return:
        """
        if self.language == 'en' and self.subtask == 2 and self.split == 'train':
            self.train_df.drop([999000878, 833032367], axis=0, inplace=True)

        if self.language == 'ge' and self.subtask == 2 and self.split == 'train':
            self.train_df.drop([224], axis=0, inplace=True)

    def extract_title_and_first_n_sentences(self, nlp: spacy.Language, n_sentences: int) -> None:
        self.train_df = _extract_title_and_first_n_sentences(self.train_df, nlp, n_sentences)
        self.eval_df = _extract_title_and_first_n_sentences(self.eval_df, nlp, n_sentences)

    def extract_title_and_first_paragraph(self, nlp: spacy.Language,
                                          min_token_paragraph: int = 40, min_sentences_paragraph: int = 1) -> None:
        self.train_df = _extract_title_and_first_paragraph(
            self.train_df, nlp, min_token_paragraph, min_sentences_paragraph)

        self.eval_df = _extract_title_and_first_paragraph(
            self.eval_df, nlp, min_token_paragraph, min_sentences_paragraph)

    def extract_title_and_first_sentence_each_paragraph(self, nlp: spacy.Language) -> None:
        self.train_df = _extract_title_and_first_sentence_each_paragraph(self.train_df, nlp)
        self.eval_df = _extract_title_and_first_sentence_each_paragraph(self.eval_df, nlp)

    def extract_all_units_of_analysis(self, nlp: spacy.Language) -> None:
        self.extract_title_and_first_n_sentences(nlp=nlp, n_sentences=5)
        self.extract_title_and_first_n_sentences(nlp=nlp, n_sentences=10)
        self.extract_title_and_first_paragraph(nlp=nlp)
        self.extract_title_and_first_sentence_each_paragraph(nlp=nlp)

    def vectorize_multilabels(self, y=None):
        if not y:
            y = self.train_df.frames

        return self.multilabel_binarizer.transform(y.str.lower().str.split(','))

def main(input_data_dir: str, subtask: int, output_path_dir: str, train_split: str, eval_split: str):
    languages = ('en', 'ru', 'it', 'fr', 'po', 'ge')


    SPACY_MODELS = {
        'en': {'small': 'en_core_web_sm', 'large': 'en_core_web_trf'},
        'ru': {'small': 'ru_core_news_sm', 'large': 'ru_core_news_lg'},
        'it': {'small': 'it_core_news_sm', 'large': 'it_core_news_lg'},
        'fr': {'small': 'fr_core_news_sm', 'large': 'fr_dep_news_trf'},
        'po': {'small': 'pl_core_news_sm', 'large': 'pl_core_news_lg'},
        'ge': {'small': 'de_core_news_sm', 'large': 'de_dep_news_trf'}
    }

    for language in languages:
        nlp = spacy.load(SPACY_MODELS[language]['small'])
        print(f'Processing: {language}')

        dataset = FramingArticleDataset(
            data_dir=input_data_dir,
            language=language, subtask=subtask,
            train_split=train_split,
            eval_split=eval_split
        )

        dataset.extract_all_units_of_analysis(nlp=nlp)
        dataset.train_df.to_csv(os.path.join(output_path_dir, f'input_{language}_{train_split}.csv'))
        dataset.eval_df.to_csv(os.path.join(output_path_dir, f'input_{language}_{eval_split}.csv'))


if __name__ == "__main__":
    output_path_ = os.path.join('..', 'data', 'preprocessed')
    os.makedirs(output_path_, exist_ok=True)
    # Extract units of analyses for all languages
    main(input_data_dir='../data/data', subtask=2, output_path_dir=output_path_, train_split='train_and_dev', eval_split='test')
