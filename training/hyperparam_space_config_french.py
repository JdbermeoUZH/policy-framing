from scipy.stats import loguniform, randint, uniform
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

        'XGBoost_narrow': {
            'model': XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False, booster='gbtree', gamma=0), #silent=True,
            #'n_search_iter': 80,
            'hyperparam_space': {
                'n_estimators': randint(150, 300),
                'reg_lambda': uniform(loc=30, scale=70),
                'colsample_bytree': uniform(loc=0.9, scale=0.1),
                'max_features': randint(10, 20),
                'learning_rate': uniform(loc=0.05, scale=0.95),
                'max_depth': randint(8, 10),
                'min_child_weight': uniform(loc=0.15, scale=0.25),
                'max_delta_step': [10],
                'scale_pos_weight': uniform(loc=10.5, scale=7.7),
                'subsample': uniform(loc=0.8, scale=0.2),
                'colsample_bynode': uniform(loc=0.8, scale=0.2)

            }
        },

        'XGBoost_broad': {
            'model': XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False, booster='gbtree', gamma=0), #silent=True,
            #'n_search_iter': 80,
            'hyperparam_space': {
                'n_estimators': randint(150, 500),
                'reg_lambda': loguniform(0.1, 1000),
                'colsample_bytree': uniform(loc=0.6, scale=0.4),
                'max_features': randint(2, 50),
                'learning_rate': loguniform(1e-2, 0.2),
                'max_depth': randint(3, 50),
                'min_child_weight': loguniform(0.01, 35),
                'max_delta_step': uniform(loc=1, scale=14),
                'scale_pos_weight': uniform(loc=1, scale=24),
                'subsample': uniform(loc=0.6, scale=0.4),
                'colsample_bynode': uniform(loc=0.6, scale=0.4)
            }
        },

        'XGBoost_narrow_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False,
                                         booster='gbtree', gamma=0))]),

            # silent=True,
            #'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 300),
                'model__reg_lambda': uniform(loc=30, scale=70),
                'model__colsample_bytree': uniform(loc=0.9, scale=0.1),
                'model__max_features': randint(10, 20),
                'model__learning_rate': uniform(loc=0.05, scale=0.95),
                'model__max_depth': randint(8, 10),
                'model__min_child_weight': uniform(loc=0.15, scale=0.25),
                'model__max_delta_step': [10],
                'model__scale_pos_weight': uniform(loc=10.5, scale=7.7),
                'model__subsample': uniform(loc=0.8, scale=0.2),
                'model__colsample_bynode': uniform(loc=0.8, scale=0.2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'XGBoost_broad_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False,
                                         booster='gbtree', gamma=0))]),

            # silent=True,
            #'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 500),
                'model__reg_lambda': loguniform(0.1, 1000),
                'model__colsample_bytree': uniform(loc=0.6, scale=0.4),
                'model__max_features': randint(2, 50),
                'model__learning_rate': loguniform(1e-2, 0.2),
                'model__max_depth': randint(3, 50),
                'model__min_child_weight': loguniform(0.01, 35),
                'model__max_delta_step': uniform(loc=1, scale=14),
                'model__scale_pos_weight': uniform(loc=1, scale=24),
                'model__subsample': uniform(loc=0.6, scale=0.4),
                'model__colsample_bynode': uniform(loc=0.6, scale=0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'XGBoost_narrow_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False,
                                         booster='gbtree', gamma=0))]),

            # silent=True,
            #'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 300),
                'model__reg_lambda': uniform(loc=30, scale=70),
                'model__colsample_bytree': uniform(loc=0.9, scale=0.1),
                'model__max_features': randint(10, 20),
                'model__learning_rate': uniform(loc=0.05, scale=0.95),
                'model__max_depth': randint(8, 10),
                'model__min_child_weight': uniform(loc=0.15, scale=0.25),
                'model__max_delta_step': [10],
                'model__scale_pos_weight': uniform(loc=10.5, scale=7.7),
                'model__subsample': uniform(loc=0.8, scale=0.2),
                'model__colsample_bynode': uniform(loc=0.8, scale=0.2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'XGBoost_broad_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False,
                                         booster='gbtree', gamma=0))]),

            # silent=True,
            #'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 500),
                'model__reg_lambda': loguniform(0.1, 1000),
                'model__colsample_bytree': uniform(loc=0.6, scale=0.4),
                'model__max_features': randint(2, 50),
                'model__learning_rate': loguniform(1e-2, 0.2),
                'model__max_depth': randint(3, 50),
                'model__min_child_weight': loguniform(0.01, 35),
                'model__max_delta_step': uniform(loc=1, scale=14),
                'model__scale_pos_weight': uniform(loc=1, scale=24),
                'model__subsample': uniform(loc=0.6, scale=0.4),
                'model__colsample_bynode': uniform(loc=0.6, scale=0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'XGBoost_narrow_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False,
                                         booster='gbtree', gamma=0))]),
            # silent=True,
            #'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 300),
                'model__reg_lambda': uniform(loc=30, scale=70),
                'model__colsample_bytree': uniform(loc=0.9, scale=0.1),
                'model__max_features': randint(10, 20),
                'model__learning_rate': uniform(loc=0.05, scale=0.95),
                'model__max_depth': randint(8, 10),
                'model__min_child_weight': uniform(loc=0.15, scale=0.25),
                'model__max_delta_step': [10],
                'model__scale_pos_weight': uniform(loc=10.5, scale=7.7),
                'model__subsample': uniform(loc=0.8, scale=0.2),
                'model__colsample_bynode': uniform(loc=0.8, scale=0.2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'XGBoost_broad_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False,
                                         booster='gbtree', gamma=0))]),

            # silent=True,
            #'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 500),
                'model__reg_lambda': loguniform(0.1, 1000),
                'model__colsample_bytree': uniform(loc=0.6, scale=0.4),
                'model__max_features': randint(2, 50),
                'model__learning_rate': loguniform(1e-2, 0.2),
                'model__max_depth': randint(3, 50),
                'model__min_child_weight': loguniform(0.01, 35),
                'model__max_delta_step': uniform(loc=1, scale=14),
                'model__scale_pos_weight': uniform(loc=1, scale=24),
                'model__subsample': uniform(loc=0.6, scale=0.4),
                'model__colsample_bynode': uniform(loc=0.6, scale=0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'XGBoost_narrow_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False,
                                         booster='gbtree', gamma=0))]),

            # silent=True,
            #'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 300),
                'model__reg_lambda': uniform(loc=30, scale=70),
                'model__colsample_bytree': uniform(loc=0.9, scale=0.1),
                'model__max_features': randint(10, 20),
                'model__learning_rate': uniform(loc=0.05, scale=0.95),
                'model__max_depth': randint(8, 10),
                'model__min_child_weight': uniform(loc=0.15, scale=0.25),
                'model__max_delta_step': [10],
                'model__scale_pos_weight': uniform(loc=10.5, scale=7.7),
                'model__subsample': uniform(loc=0.8, scale=0.2),
                'model__colsample_bynode': uniform(loc=0.8, scale=0.2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
            }
        },

        'XGBoost_broad_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False,
                                         booster='gbtree', gamma=0))]),

            # silent=True,
            #'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 500),
                'model__reg_lambda': loguniform(0.1, 1000),
                'model__colsample_bytree': uniform(loc=0.6, scale=0.4),
                'model__max_features': randint(2, 50),
                'model__learning_rate': loguniform(1e-2, 0.2),
                'model__max_depth': randint(3, 50),
                'model__min_child_weight': loguniform(0.01, 35),
                'model__max_delta_step': uniform(loc=1, scale=14),
                'model__scale_pos_weight': uniform(loc=1, scale=24),
                'model__subsample': uniform(loc=0.6, scale=0.4),
                'model__colsample_bynode': uniform(loc=0.6, scale=0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
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
