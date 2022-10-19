import os

import pandas as pd
import spacy
from tqdm import tqdm

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
                      not sentence.isspace() and not len(sentence) == 0]

    return ' '.join(sent_list[: min(len(sent_list), n_sentences)])


"""
Classes to represent task 3 datasets
"""


class BaseArticleDataset:
    def __init__(self, data_dir: str = 'data', language: str = 'en', subtask: int = 2, split: str = 'train'):
        self.language = language
        self.subtask = subtask
        self.split = split
        self.label_file_path = os.path.join(data_dir, language, f'{split}-labels-subtask-{subtask}.txt')
        self.article_dir_path = os.path.join(data_dir, language, f'{split}-articles-subtask-{subtask}')
        self.df = self._create_article_dataframe()

    def _create_article_dataframe(self) -> pd.DataFrame:
        text = []

        for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(self.article_dir_path))):
            id_, txt = fil[7:].split('.')[0], \
                       open(os.path.join(self.article_dir_path, fil), 'r', encoding='utf-8').read()
            text.append((id_, txt))

        return pd.DataFrame(text, columns=['id', 'raw_text']) \
            .astype({'id': 'int', 'raw_text': 'str'}) \
            .set_index('id')

    def separate_title_content(self, remove_raw_data_col: bool = False) -> None:
        """
           1) Remove trailing whitespaces and breaklines
           2) Separate on single breakline "\n"
           3) Remove trailing whitespaces and breaklines for title and content again
        :param remove_raw_data_col:
        :return:
        """

        self.df[['title', 'content']] = self.df.pipe(lambda df: df.raw_text.str.strip()) \
                                               .pipe(lambda str_series: str_series.str.split('\n', 1, expand=True)) \
                                               .pipe(lambda df: df.apply(lambda series: series.str.strip()))

        if remove_raw_data_col:
            self.df.drop('raw_text', axis=1, inplace=True)

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.df.__repr__()


class FramingArticleDataset(BaseArticleDataset):
    def __init__(self, data_dir: str = 'data', language: str = 'en', subtask: int = 2, split: str = 'train'):
        super().__init__(data_dir=data_dir, language=language, subtask=subtask, split=split)
        if split == 'train':
            self._add_labels()
            self._remove_duplicates()

        self.separate_title_content()

    def _add_labels(self) -> None:
        # MAKE LABEL DATAFRAME
        labels = pd.read_csv(self.label_file_path, sep='\t', header=None, names=['id', 'frames'],
                             dtype={'id': 'int', 'frames': 'str'}) \
            .set_index('id')

        # JOIN
        self.df = labels.join(self.df)[['raw_text', 'frames']]

    def _remove_duplicates(self):
        """
        Manually identified duplicate documents in the trainset
        :return:
        """
        if self.language == 'en' and self.subtask == 2 and self.split == 'train':
            self.df.drop([999000878, 833032367], axis=0, inplace=True)

    def extract_title_and_first_n_sentences(self, nlp: spacy.Language, n_sentences: int) -> None:
        if 'title' not in self.df.columns:
            self.separate_title_content()

        self.df[f'title_and_{n_sentences}_sentences'] = \
            self.df.title + '\n' + \
            self.df.content.map(lambda content: extract_n_sentences(
                content, nlp, n_sentences=n_sentences, break_pattern='\n'))

    def extract_title_and_first_paragraph(self, nlp: spacy.Language,
                                          min_token_paragraph: int = 40, min_sentences_paragraph: int = 1) -> None:
        first_paragraph = self.df.content.str.split('\n').str[0]
        second_paragraph = self.df.content.str.split('\n').str[1]

        # If there is only one sentence in the first paragraph or its length is under 40 tokens,
        # merge it with the second paragraph
        merge_first_two_paragraphs = \
            [len(processed_str) <= min_token_paragraph or len(list(processed_str.sents)) <= min_sentences_paragraph
             for processed_str in nlp.pipe(first_paragraph.tolist())]

        first_paragraph = (first_paragraph + ' ' + second_paragraph).where(merge_first_two_paragraphs, first_paragraph)

        self.df[f'title_and_first_paragraph'] = \
            self.df.title + '\n' + first_paragraph

    def extract_title_and_first_sentence_each_paragraph(self, nlp: spacy.Language) -> None:
        paragraphs_per_doc = self.df.content.str.split('\n')

        self.df[f'title_and_first_sentence_each_paragraph'] = paragraphs_per_doc.map(lambda paragraphs: ' '.join(
            [next(nlp(paragraph).sents).text for paragraph in paragraphs if paragraph.strip() != '']))

    def extract_all_units_of_analysis(self, nlp: spacy.Language) -> None:
        self.extract_title_and_first_n_sentences(nlp=nlp, n_sentences=5)
        self.extract_title_and_first_n_sentences(nlp=nlp, n_sentences=10)
        self.extract_title_and_first_paragraph(nlp=nlp)
        self.extract_title_and_first_sentence_each_paragraph(nlp=nlp)

    def save_dataset(self, output_path: str) -> None:
        if output_path.split('.')[-1] == 'csv':
            self.df.to_csv(output_path)

        if output_path.split('.')[-1] == 'pkl':
            self.df.to_pickle(output_path)


def main(language: str, input_data_dir: str, subtask: int, output_path_dir: str):
    spacy_models = {'en': 'en_core_web_sm', 'ru': 'ru_core_news_sm', 'it': 'it_core_news_sm'}
    nlp = spacy.load(spacy_models[language])

    train_data = FramingArticleDataset(data_dir=input_data_dir, language=language, subtask=subtask, split='train')
    train_data.extract_all_units_of_analysis(nlp=nlp)
    train_data.save_dataset(os.path.join(output_path_dir, f'input_{language}_train.csv'))

    test_data = FramingArticleDataset(data_dir=input_data_dir, language=language, subtask=subtask, split='dev')
    test_data.extract_all_units_of_analysis(nlp=nlp)
    train_data.save_dataset(os.path.join(output_path_dir, f'input_{language}_dev.csv'))


if __name__ == "__main__":
    output_path_ = os.path.join('../data', 'preprocessed')
    os.makedirs(output_path_, exist_ok=True)
    main(language='en', input_data_dir='../data', subtask=2, output_path_dir=output_path_)
    main(language='ru', input_data_dir='../data', subtask=2, output_path_dir=output_path_)
    main(language='it', input_data_dir='../data', subtask=2, output_path_dir=output_path_)
