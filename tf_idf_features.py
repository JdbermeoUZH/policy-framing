import argparse
from typing import List

import yaml
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


def basic_cleaning(text: str, spacy_nlp_model: spacy.Language) -> List[str]:
    """
    Lemmatize, remove punctutation, and stopwords of a string
    :return:
    """
    return [token.lemma_ for token in spacy_nlp_model(text) if not token.is_punct and not token.is_stop]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subtask-2')
    parser.add_argument('config_path', type=str, nargs=1,
                        help='Path to configuration file with parameters to use')
    args = parser.parse_args()

    # Load parameters of configuration file
    with open(args.config_path, "r") as stream:
        try:
            config_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Load data
    sem_eval_train = FramingArticleDataset(data_dir='data', language='en', subtask=2, split='train')
    sem_eval_train.separate_title_content(remove_raw_data_col=False)



