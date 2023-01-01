from scipy.stats import loguniform, randint
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB

MODEL_LIST = \
    {
        'LogisticRegression': {
            'model': LogisticRegression(penalty='none', class_weight='balanced', max_iter=100000),
            'n_search_iter': 5,
            'hyperparam_space': {
                'estimator__random_state': randint(0, 1000)
            }
        },

        'LogisticRegressionRidge': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', max_iter=100000),
            'n_search_iter': 60,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-6, 0.5)
            }
        },

        'LogisticRegressionRidgeDual': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                        max_iter=100000),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-4, 0.05),
            }
        },

        'LogisticRegressionLasso': {
            'model': LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000, class_weight='balanced'),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__C': loguniform(1, 100),
            }
        },


        'LogisticRegressionElasticNet': {
            'model': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                        class_weight='balanced'),
            'n_search_iter': 30,
            'hyperparam_space': {
                'estimator__C': loguniform(0.6, 7)
            }
        },

        'RidgeClassifierV1': {
            'model': RidgeClassifier(max_iter=100000, class_weight='balanced'),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1, 2e3)
            }
        },

        'RidgeClassifierV2': {
            'model': RidgeClassifier(max_iter=100000, class_weight='balanced'),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1, 50)
            }
        },

        'SVM_rbf': {
            'model': SVC(kernel='rbf', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__C': loguniform(30, 800),
                'estimator__gamma': loguniform(5e-4, 0.04),
            }
        },


        'SVM_sigmoid_narrow_gamma': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 600),
                'estimator__gamma': loguniform(1e-2, 500),
            }
        },

        'SVM_sigmoid_broader_gamma': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-2, 600),
                'estimator__gamma': loguniform(1e-4, 500),
            }
        },

        'LinearSVM': {
            'model': LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=100000),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-6, 5e-3),
            }
        },

        'LinearSVMDual': {
            'model': LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=100000),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-6, 1e-2),
            }
        },

        'RandomForestSK_V1': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__max_features': ['sqrt', 'log2'],
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': randint(2, 25),
                'estimator__min_samples_leaf': randint(5, 35),
                'estimator__class_weight': ["balanced_subsample"]
            }
        },

        'RandomForestSK_V2': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 30,
            'hyperparam_space': {
                'estimator__max_features': randint(1, 50),
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': randint(2, 25),
                'estimator__min_samples_leaf': randint(5, 35),
            }
        },

        'RandomForestSK_V3': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 70,
            'hyperparam_space': {
                'estimator__criterion': ['gini', 'entropy', 'log_loss'],
                'estimator__max_features': randint(1, 50),
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': randint(2, 100),
                'estimator__min_samples_leaf': randint(5, 35),
                'estimator__ccp_alpha': loguniform(1e-6, 5e-1),

            }
        },

        'XGBoost': {
            'model': XGBClassifier(verbosity=0, silent=True),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__max_features': ['sqrt', 'log2'],
                'estimator__gamma': loguniform(1e-5, 1e-2),
                'estimator__max_depth': randint(6, 30),
                'estimator__min_child_weight': randint(1, 6),
                'estimator__max_delta_step': [0, 1, 5],
            }
        },

        'XGBoostV2': {
            'model': XGBClassifier(verbosity=0, silent=True, use_label_encoder=False),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__max_features': randint(1, 20),
                'estimator__gamma': loguniform(1e-5, 1e-2),
                'estimator__max_depth': randint(6, 35),
                'estimator__min_child_weight': randint(1, 6),
                'estimator__max_delta_step': randint(0, 6),
            }
        },

        'ComplementNaiveBayes': {
            'model': ComplementNB(),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1e-3, 2),
                'estimator__norm': [True, False]
            }
        },

        'NaiveBayes': {
            'model': MultinomialNB(fit_prior=False),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__alpha': loguniform(0.07, 1),
            }
        }

    }


if __name__ == '__main__':
    print(MODEL_LIST.keys())
