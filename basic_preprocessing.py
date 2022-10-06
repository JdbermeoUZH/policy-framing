import os

import pandas as pd
from tqdm import tqdm


class BaseArticleDataset:
    def __init__(self, data_dir: str = 'data', language: str = 'en', subtask: int = 2, split: str = 'train'):
        self.label_file_path = os.path.join(data_dir, language, f'{split}-labels-subtask-{subtask}.txt')
        self.article_dir_path = os.path.join(data_dir, language, f'{split}-articles-subtask-{subtask}')
        self.df = self._create_article_dataframe()

    def _create_article_dataframe(self) -> pd.DataFrame:
        text = []

        for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(self.article_dir_path))):
            id, txt = fil[7:].split('.')[0], \
                      open(os.path.join(self.article_dir_path, fil), 'r', encoding='utf-8').read()
            text.append((id, txt))

        return pd.DataFrame(text, columns=['id', 'raw_text'])\
            .astype({'id': 'int', 'raw_text': 'str'})\
            .set_index('id')

    def separate_title_content(self, remove_raw_data_col: bool = True) -> None:
        self.df[['title', 'content']] = self.df.raw_text.str.split('\n\n', 1, expand=True)

        if remove_raw_data_col:
            self.df.drop('raw_text', axis =1, inplace=True)


class FramingArticleDataset(BaseArticleDataset):
    def __init__(self, data_dir: str = 'data', language: str = 'en', subtask: int = 2, split: str = 'train'):
        super().__init__(data_dir=data_dir, language=language, subtask=subtask, split=split)
        self._add_labels()

    def _add_labels(self) -> None:
        # MAKE LABEL DATAFRAME
        labels = pd.read_csv(self.label_file_path, sep='\t', header=None, names=['id', 'frames'],
                             dtype={'id': 'int', 'frames': 'str'})\
            .set_index('id')

        # JOIN
        self.df = labels.join(self.df)[['raw_text', 'frames']]


if __name__ == "__main__":
    semeval_data = FramingArticleDataset(data_dir='data', language='en', subtask=2, split='train')
    semeval_data.separate_title_content()

