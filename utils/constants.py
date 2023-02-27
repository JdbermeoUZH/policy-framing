
LANGUAGES = ('en', 'it', 'fr', 'po', 'ru', 'ge')

SPACY_MODELS = {
    'en': {'small': 'en_core_web_sm', 'large': 'en_core_web_trf'},
    'it': {'small': 'it_core_news_sm', 'large': 'it_core_news_lg'},
    'fr': {'small': 'fr_core_news_sm', 'large': 'fr_dep_news_trf'},
    'po': {'small': 'pl_core_news_sm', 'large': 'pl_core_news_lg'},
    'ru': {'small': 'ru_core_news_sm', 'large': 'ru_core_news_lg'},
    'ge': {'small': 'de_core_news_sm', 'large': 'de_dep_news_trf'}
}

LABELS = ('Economic', 'Capacity_and_resources', 'Morality', 'Fairness_and_equality',
          'Legality_Constitutionality_and_jurisprudence', 'Policy_prescription_and_evaluation', 'Crime_and_punishment',
          'Security_and_defense', 'Health_and_safety', 'Quality_of_life', 'Cultural_identity', 'Public_opinion',
          'Political', 'External_regulation_and_reputation')


UNITS_OF_ANALYSES = ('title', 'title_and_first_paragraph', 'title_and_5_sentences', 'title_and_10_sentences',
                     'title_and_first_sentence_each_paragraph', 'raw_text')
