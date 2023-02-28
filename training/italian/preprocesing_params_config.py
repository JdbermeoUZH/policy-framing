

PREPROCESSING = {
    'all': {
        'fixed_params': {
            'use_tfidf': True,
            'min_df': 1,
            'max_df': 1.0,
            'max_features': 10000,
            'ngram_range': [1, 3],
            'min_var': 0.0,
            'corr_threshold': 1.0,
        },

        'param_search': {
            'min_df_range': [0.008, 0.012],
            'max_df_range': [0.78, 0.8],
            'min_var_range': [0, 0.01],
            'corr_threshold_range': [0.987, 1]
        }
    },

    'raw_text': {
        'fixed_params': {
            'use_tfidf': True,
            'min_df': 0.01,
            'max_df': 0.8,
            'max_features': 10000,
            'ngram_range': [1, 3],
            'min_var': 0.0,
            'corr_threshold': 1.0,
        },

        'param_search': {
            'min_df_range': [0.0, 0.04],
            'max_df_range': [0.4, 1.0],
            'min_var_range': [0.0, 0.01],
            'corr_threshold_range': [0.98, 1.0]
        }
    },

    'title_and_first_sentence_each_paragraph': {
        'fixed_params': {
            'use_tfidf': True,
            'min_df': 0.0015,
            'max_df': 0.8,
            'max_features': 10000,
            'ngram_range': [1, 3],
            'min_var': 0.0,
            'corr_threshold': 1.0,
        },

        'param_search': {
            'min_df_range': [0.0, 0.04],
            'max_df_range': [0.4, 0.9],
        }
    },

    'title_and_10_sentences': {
        'fixed_params': {
            'use_tfidf': True,
            'min_df': 0.01,
            'max_df': 0.675,
            'max_features': 10000,
            'ngram_range': [1, 3],
            'min_var': 0.0,
            'corr_threshold': 1.0,
        },

        'param_search': {
            'min_df_range': [0.0, 0.03],
            'max_df_range': [0.4, 0.9],
            'min_var_range': [0, 0.001],
            'corr_threshold_range': [0.97, 1]
        }
    },

    'title_and_5_sentences': {
        'fixed_params': {
            'use_tfidf': True,
            'min_df': 0.01,
            'max_df': 0.675,
            'max_features': 10000,
            'ngram_range': [1, 3],
            'min_var': 0.0,
            'corr_threshold': 1.0,
        },

        'param_search': {
            'min_df_range': [0.000, 0.015],
            'max_df_range': [0.4, 0.85],
            'min_var_range': [0, 0.001],
            'corr_threshold_range': [0.92, 1.0]
        }
    },

    'title_and_first_paragraph': {
        'fixed_params': {
            'use_tfidf': True,
            'min_df': 0.003,
            'max_df': 0.625,
            'max_features': 10000,
            'ngram_range': [1, 3],
            'min_var': 0.0,
            'corr_threshold': 1.0,
        },

        'param_search': {
            'min_df_range': [0.0, 0.015],
            'max_df_range': [0.4, 0.8],
        }
    },

    'title': {
        'fixed_params': {
            'use_tfidf': True,
            'min_df': 0.003,
            'max_df': 0.625,
            'max_features': 10000,
            'ngram_range': [1, 3],
            'min_var': 0.0,
            'corr_threshold': 1.0,
        },

        'param_search': {
            'min_df_range': [0.0, 0.015],
            'max_df_range': [0.4, 0.8],
        }
    }
}