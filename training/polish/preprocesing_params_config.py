

PREPROCESSING = {
    'all': {
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
            'min_df_range': [0.008, 0.012],
            'max_df_range': [0.78, 0.8],
            'min_var_range': [0, 0.01],
            'corr_threshold_range': [0.987, 1]
        }
    },

    'raw_text': {
        'fixed_params': {
            'use_tfidf': True,
            'min_df': 0.017,
            'max_df': 0.82,
            'max_features': 10000,
            'ngram_range': [1, 3],
            'min_var': 0.0,
            'corr_threshold': 0.99,
        },

        'param_search': {
            'min_df_range': [0.01, 0.04],
            'max_df_range': [0.45, 0.87],
            'min_var_range': [0.0, 0.01],
            'corr_threshold_range': [0.98, 1.0]
        }
    },

    'title_and_first_sentence_each_paragraph': {
        'fixed_params': {
            'use_tfidf': True,
            'min_df': 0.0015,
            'max_df': 0.77,
            'max_features': 10000,
            'ngram_range': [1, 3],
            'min_var': 0.0,
            'corr_threshold': 1.0,
        },

        'param_search': {
            'min_df_range': [0.0, 0.04],
            'max_df_range': [0.76, 0.78],
        }
    },

    'title_and_10_sentences': {
        'fixed_params': {
            'use_tfidf': True,
            'min_df': 0.025,
            'max_df': 0.55,
            'max_features': 10000,
            'ngram_range': [1, 3],
            'min_var': 0.0,
            'corr_threshold': 1.0,
        },

        'param_search': {
            'min_df_range': [0.02, 0.03],
            'max_df_range': [0.48, 0.62],
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
            'min_df_range': [0.008, 0.015],
            'max_df_range': [0.64, 0.84],
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
            'max_df_range': [0.4, 0.65],
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
            'max_df_range': [0.4, 0.65],
        }
    }
}