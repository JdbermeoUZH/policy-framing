import os
from types import ModuleType, GeneratorType

import yaml
import argparse
from typing import Tuple
from importlib import import_module

import spacy
from sklearn.preprocessing import MultiLabelBinarizer

from preprocessing.InputDataset import FramingArticleDataset
from preprocessing.BOWPipeline import BOWPipeline, basic_tokenizing_and_cleaning
from training.Logger import Logger
from training.MultiLabelEstimator import MultiLabelEstimator
from utils.metrics_config import scoring_functions
from utils import helper_fns
from benchmark_subtask_2 import parse_arguments_and_load_config_file

LANGUAGES = ('en', 'it', 'fr', 'po', 'ru', 'ge')

SPACY_MODELS = {
    'en': {'small': 'en_core_web_sm', 'large': 'en_core_web_trf'},
    'it': {'small': 'it_core_news_sm', 'large': 'it_core_news_lg'},
    'fr': {'small': 'fr_core_news_sm', 'large': 'fr_dep_news_trf'},
    'po': {'small': 'pl_core_news_sm', 'large': 'pl_core_news_lg'},
    'ru': {'small': 'ru_core_news_sm', 'large': 'ru_core_news_lg'},
    'ge': {'small': 'de_core_news_sm', 'large': 'de_dep_news_trf'}
}

LABELS = ('fairness_and_equality', 'security_and_defense', 'crime_and_punishment', 'morality',
          'policy_prescription_and_evaluation', 'capacity_and_resources', 'economic', 'cultural_identity',
          'health_and_safety', 'quality_of_life', 'legality_constitutionality_and_jurisprudence',
          'political', 'public_opinion', 'external_regulation_and_reputation')

UNITS_OF_ANALYSES = ('title', 'title_and_first_paragraph', 'title_and_5_sentences', 'title_and_10_sentences',
                     'title_and_first_sentence_each_paragraph', 'raw_text')


if __name__ == "__main__":
    # Load script arguments and configuration file
    args, config = parse_arguments_and_load_config_file()
    dataset_config = config['dataset']
    preprocessing_config = config['preprocessing']
    training_config = config['training']
    estimators_config = import_module(config['training']['model_hyperparam_module'])
    metric_log_config = config['metric_logging']
    run_config = config['run']

