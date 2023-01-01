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
            'n_search_iter': 20,
            'hyperparam_space': {
                'estimator__C': loguniform(5e-3, 5e-2),
                'estimator__random_state': randint(0, 1000)
            }
        },

        'LogisticRegressionRidgeDual': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced'),
            'n_search_iter': 20,
            'hyperparam_space': {
                'estimator__C': loguniform(0.018, 0.5),
                'estimator__class_weight': ['balanced'],
                'estimator__max_iter': randint(20000, 100000),
                'estimator__random_state': randint(0, 1000)
            }
        },

        'LogisticRegressionLasso': {
            'model': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=100000),
            'n_search_iter': 20,
            'hyperparam_space': {
                'estimator__C': loguniform(1.5, 3.5),
                'estimator__random_state': randint(0, 1000)
            }
        },

        'LogisticRegressionElasticNet': {
            'model': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=250000,
                                        class_weight='balanced'),
            'n_search_iter': 20,
            'hyperparam_space': {
                'estimator__C': loguniform(1, 7),
            }
        },

        'RidgeClassifier': {
            'model': RidgeClassifier(max_iter=100000, class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__alpha': loguniform(10, 2e3)
            }
        },

        'SVM_rbf': {
            'model': SVC(kernel='rbf', class_weight='balanced'),
            'n_search_iter': 30,
            'hyperparam_space': {
                'estimator__C': loguniform(700, 1e3),
                'estimator__gamma': loguniform(5e-4, 1e-3),
            }
        },


        'SVM_sigmoid': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            'n_search_iter': 20,
            'hyperparam_space': {
                'estimator__C': loguniform(0.2, 1),
                'estimator__gamma': loguniform(2, 8),
            }
        },


        'LinearSVM': {
            'model': LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=1e5),
            'n_search_iter': 80,
            'hyperparam_space': {
                'estimator__C': loguniform(5e-4, 6e-3),
            }
        },

        'LinearSVMDual': {
            'model': LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-5, 3e-3),
            }
        },

        'RandomForest_V1': {
            'model': RandomForestClassifier(class_weight="balanced_subsample", bootstrap=False),
            'n_search_iter': 100,
            'hyperparam_space': {
                'estimator__max_features': randint(2, 50),
                'estimator__criterion': ['gini', 'entropy', 'log_loss'],
                'estimator__n_estimators': [50, 75, 100],
                'estimator__max_depth': randint(2, 100),
                'estimator__ccp_alpha': loguniform(1e-6, 5e-3),
                'estimator__min_samples_leaf': randint(5, 20),
                'estimator__max_leaf_nodes': randint(2, 200)
            }
        },

        'RandomForest_V2': {
            'model': RandomForestClassifier(class_weight="balanced_subsample", bootstrap=False),
            'n_search_iter': 100,
            'hyperparam_space': {
                'estimator__max_features': randint(14, 50),
                'estimator__criterion': ['gini', 'entropy', 'log_loss'],
                'estimator__n_estimators': [50, 75, 100],
                'estimator__max_depth': randint(20, 40),
                'estimator__ccp_alpha': loguniform(1e-6, 5e-3),
                'estimator__min_samples_leaf': randint(12, 20),
                'estimator__max_leaf_nodes': randint(40, 100)
            }
        },

        'XGBoost': {
            'model': XGBClassifier(verbosity=0, silent=True, use_label_encoder=False),
            #'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__max_features': ['sqrt', 'log2'],
                'estimator__gamma': loguniform(1e-5, 1e-2),
                'estimator__max_depth': randint(6, 30),
                'estimator__min_child_weight': randint(1, 6),
                'estimator__max_delta_step': [0, 1, 5],
            }
        },

        'XGBoostV2': {
            'model': XGBClassifier(verbosity=0, silent=True),
            # 'n_search_iter': 10,
            'hyperparam_space': {
                'estimator__max_features': randint(1, 20),
                'estimator__gamma': loguniform(1e-5, 1e-2),
                'estimator__max_depth': randint(6, 30),
                'estimator__min_child_weight': randint(1, 6),
                'estimator__max_delta_step': [0, 1, 5],
            }
        },


        'ComplementNaiveBayes': {
            'model': ComplementNB(),
            'n_search_iter': 30,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1e-1, 0.5),
                'estimator__norm': [True, False]
            }
        },

        'NaiveBayes': {
            'model': MultinomialNB(fit_prior=False),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__alpha': loguniform(0.005, 0.4),
            }
        },

    }


if __name__ == '__main__':
    print(MODEL_LIST.keys())
