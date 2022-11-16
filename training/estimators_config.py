from scipy.stats import loguniform, randint
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB

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
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__class_weight': ['balanced', None],
                'estimator__max_iter': randint(1000, 100000)
            }
        },

        'LogisticRegressionRidge': {
            'model': LogisticRegression(penalty='l2'),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__solver': ['liblinear', 'lbfgs'],
                'estimator__class_weight': ['balanced', None],
                'estimator__max_iter': randint(1000, 100000)
            }
        },

        'LogisticRegressionRidgeDual': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', dual=True),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__class_weight': ['balanced', None],
                'estimator__max_iter': randint(1000, 100000)
            }
        },

        'LogisticRegressionLasso': {
            'model': LogisticRegression(penalty='l1', solver='liblinear'),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__class_weight': ['balanced', None],
                'estimator__max_iter': randint(1000, 100000)
            }
        },

        'LogisticRegressionElasticNet': {
            'model': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__class_weight': ['balanced', None],
                'estimator__max_iter': randint(30000, 100000)
            }
        },

        'RidgeClassifier': {
            'model': RidgeClassifier(),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1e-2, 1e3),
                'estimator__class_weight': ['balanced', None],
                'estimator__max_iter': randint(1000, 100000)
            }
        },

        'SVM': {
            'model': SVC(),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__gamma': loguniform(1e-4, 1e-1),
                'estimator__kernel': ['rbf', 'poly', 'sigmoid']
            }
        },

        'LinearSVM': {
            'model': LinearSVC(dual=False),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__penalty': ['l2', 'l1'],
                'estimator__class_weight': ['balanced', None],
                'estimator__max_iter': randint(2000, 20000)
            }
        },

        'LinearSVMDual': {
            'model': LinearSVC(dual=True, penalty='l2'),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 1e3),
                'estimator__class_weight': ['balanced', None],
                'estimator__max_iter': randint(2000, 20000)
            }
        },

        'RandomForest': {
            'model': RandomForestClassifier(),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__max_features': ['sqrt', 'log2'],
                'estimator__n_estimators': [50, 100, 200, 400],
                'estimator__max_depth': randint(2, 50),
                'estimator__min_samples_leaf': randint(1, 10),
                'estimator__class_weight': ["balanced", "balanced_subsample", None]
            }
        },

        'XGBoost': {
            'model': XGBClassifier(verbosity=0, silent=True),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__max_features': ['sqrt', 'log2'],
                'estimator__gamma': loguniform(1e-2, 1e3),
                'estimator__max_depth': randint(6, 30),
                'estimator__min_child_weight': randint(1, 10),
                'estimator__max_delta_step': [0, 1, 5],
            }
        },

        'RandomForestV2': {
            'model': XGBRFClassifier(verbosity=0, silent=True),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__n_estimators': [50, 100, 200, 400],
                'estimator__max_depth': randint(2, 50),
            }
        },

        'ComplementNaiveBayes': {
            'model': ComplementNB(),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1e-2, 1e3),
                'estimator__norm': [True, False]
            }
        },

        'NaiveBayes': {
            'model': MultinomialNB(),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1e-2, 1e3),
                'estimator__fit_prior': [True, False]
            }
        }

    }

