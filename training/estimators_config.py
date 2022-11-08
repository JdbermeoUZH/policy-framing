from scipy.stats import loguniform, randint
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

MODEL_LIST = \
    {
        'DummyProbSampling': {
            'model': DummyClassifier(strategy='stratified'),
            'n_search_iter': 1,
            'hyperparam_space': {
                'estimator__strategy': ['stratified'],
            }
        },

        'DummyUniformSampling': {
            'model': DummyClassifier(strategy='uniform'),
            'n_search_iter': 1,
            'hyperparam_space': {
                'estimator__strategy': ['uniform'],
            }
        },

        'DummyMostFrequent': {
            'model': DummyClassifier(strategy='prior'),
            'n_search_iter': 1,
            'hyperparam_space': {
                'estimator__strategy': ['prior'],
            }
        },

        'LogisticRegression': {
            'model': LogisticRegression(penalty='none'),
            'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__dual': [True, False]
            }
        },

        'LogisticRegressionRidge': {
            'model': LogisticRegression(penalty='l2'),
            'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__dual': [True, False]
            }
        },

        'LogisticRegressionLasso': {
            'model': LogisticRegression(penalty='l1', solver='liblinear'),
            'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__dual': [True, False]
            }
        },

        'LogisticRegressionElasticNet': {
            'model': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5),
            'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__dual': [True, False]
            }
        },

        'RidgeClassifier': {
            'model': RidgeClassifier(),
            'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1e-2, 1e3),
                'estimator__kernel': ['rbf', 'poly', 'sigmoid']
            }
        },

        'SVM': {
            'model': SVC(),
            'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__gamma': loguniform(1e-4, 1e-1),
                'estimator__kernel': ['rbf', 'poly', 'sigmoid']
            }
        },

        'LinearSVM': {
            'model': LinearSVC(),
            'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__penalty': ['l2', 'l1'],
                'estimator__dual': [True, False],
                'estimator__class_weight': ['balance', None],
            }
        },


        'RandomForest': {
            'model': RandomForestClassifier(),
            'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__max_features': ['sqrt', 'log2'],
                'estimator__n_estimators': [50, 100, 200, 400],
                'estimator__max_depth': randint(2, 50),
                'estimator__min_samples_leaf': randint(1, 10),
                'estimator__class_weight': ["balanced", "balanced_subsample", None]
            }
        },

        'XGBoost': {
            'model': XGBClassifier(),
            'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__max_features': ['sqrt', 'log2'],
                'estimator__gamma': loguniform(1e-2, 1e3),
                'estimator__max_depth': randint(6, 30),
                'estimator__min_child_weight': randint(1, 10),
                'estimator__max_delta_step': [0, 1, 5],
            }
        },

        'RandomForestV2': {
            'model': XGBRFClassifier(),
            'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__n_estimators': [50, 100, 200, 400],
                'estimator__max_depth': randint(2, 50),
            }
        }

    }

