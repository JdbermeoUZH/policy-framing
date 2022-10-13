import os

import pandas as pd
import spacy
from tqdm import tqdm

"""
Helper Functions
"""


def extract_n_sentences(text, nlp, n_sentences, break_pattern='\n'):
    sentence_list = list(nlp(text).sents)
    sentence_list = sentence_list[: min(len(sentence_list), n_sentences)]

    # Add an additional pattern to split up the textract_n_sentences
    sent_list = []
    for sentence in sentence_list:
        sent_list += [sentence for sentence in sentence.text.split(break_pattern) if
                      not sentence.isspace() and not len(sentence) == 0]

    return sent_list[: min(len(sent_list), n_sentences)]


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
            id, txt = fil[7:].split('.')[0], \
                      open(os.path.join(self.article_dir_path, fil), 'r', encoding='utf-8').read()
            text.append((id, txt))

        return pd.DataFrame(text, columns=['id', 'raw_text']) \
            .astype({'id': 'int', 'raw_text': 'str'}) \
            .set_index('id')

    def separate_title_content(self, remove_raw_data_col: bool = False) -> None:
        # TODO: Improve behavior by:
        #   1) Remove trailing whitespaces and breaklines
        #   2) Separate on single breakline "\n"
        #   3) Remove trailing whitespaces and breaklines for title and content again
        self.df[['title', 'content']] = self.df.raw_text.str.split('\n\n', 1, expand=True)

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

    def _add_labels(self) -> None:
        # MAKE LABEL DATAFRAME
        labels = pd.read_csv(self.label_file_path, sep='\t', header=None, names=['id', 'frames'],
                             dtype={'id': 'int', 'frames': 'str'}) \
            .set_index('id')

        # JOIN
        self.df = labels.join(self.df)[['raw_text', 'frames']]

    # TODO: Add the units of analyses proposed by Meysam
    def extract_title_and_first_n_sentences(self, nlp: spacy.Language, n_sentences: int) -> None:
        if 'title' not in self.df.columns:
            self.separate_title_content()

        self.df['title_and_first_sentence'] = \
            self.df.title + '\n' + \
            extract_n_sentences(self.df.content, nlp,
                                n_sentences=n_sentences, break_pattern='\n')


if __name__ == "__main__":
    # Load data into a dataframe and separate both title and content
    semeval_data = FramingArticleDataset(data_dir='data', language='en', subtask=2, split='train')
    semeval_data.separate_title_content()
    print(semeval_data)
