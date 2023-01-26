from scipy.stats import loguniform, randint
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline

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
                'estimator__C': loguniform(1e-6, 1e-1)
            }
        },

        'LogisticRegressionRidge_V2': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', max_iter=100000),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__C': loguniform(5e-3, 5e-2),
                'estimator__random_state': randint(0, 1000)
            }
        },

        'LogisticRegressionRidgeDual': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                        max_iter=100000),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-4, 1e-1),
            }
        },

        'LogisticRegressionRidgeDualV2': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                        max_iter=100000),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__C': loguniform(0.018, 0.5),
            }
        },

        'LogisticRegressionLassoV1': {
            'model': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=100000),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__C': loguniform(1, 20),
            }
        },

        'LogisticRegressionLassoV2': {
            'model': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=100000),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__C': loguniform(1, 10),
            }
        },

        'LogisticRegressionLassoV3': {
            'model': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=100000),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__C': loguniform(1.5, 3.5),
            }
        },

        'LogisticRegressionElasticNet': {
            'model': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=250000,
                                        class_weight='balanced'),
            'n_search_iter': 40,
            'hyperparam_space': {
                'estimator__C': loguniform(1, 7),
            }
        },

        'RidgeClassifier': {
            'model': RidgeClassifier(max_iter=100000, class_weight='balanced'),
            'n_search_iter': 60,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1, 2e3)
            }
        },

        'SVM_rbf': {
            'model': SVC(kernel='rbf', class_weight='balanced'),
            'n_search_iter': 60,
            'hyperparam_space': {
                'estimator__C': loguniform(10, 1e3),
                'estimator__gamma': loguniform(1e-4, 1e-2),
            }
        },

        'SVM_rbf_v2': {
            'model': SVC(kernel='rbf', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__C': loguniform(1, 20),
                'estimator__gamma': loguniform(1e-2, 0.2),
            }
        },

        'SVM_sigmoid': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__C': loguniform(0.05, 8),
                'estimator__gamma': loguniform(1e-1, 8),
            }
        },

        'SVM_sigmoidV1': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__C': loguniform(0.2, 1),
                'estimator__gamma': loguniform(2, 8),
            }
        },

        'SVM_sigmoidV2': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__C': loguniform(0.05, 6),
                'estimator__gamma': loguniform(1e-1, 8),
            }
        },

        'LinearSVM': {
            'model': LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=1e5),
            'n_search_iter': 80,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-6, 5e-3),
            }
        },

        'LinearSVM_broad_C': {
            'model': LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=1e5),
            'n_search_iter': 80,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-6, 1),
            }
        },

        'LinearSVMDual': {
            'model': LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-6, 3e-3),
            }
        },

        'LinearSVMDual_broad_C': {
            'model': LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__C': loguniform(1e-7, 1e-2),
            }
        },

        'RandomForest_v1': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 60,
            'hyperparam_space': {
                'estimator__max_features': ['sqrt', 'log2'],
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': randint(2, 50),
                'estimator__min_samples_leaf': randint(5, 25),
            }
        },

        'RandomForest_v2': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 60,
            'hyperparam_space': {
                'estimator__max_features': ['sqrt', 'log2'],
                'estimator__criterion': ['gini', 'entropy', 'log_loss'],
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': randint(2, 100),
                'estimator__min_samples_leaf': randint(5, 25),
            }
        },

        'RandomForest_v3': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 60,
            'hyperparam_space': {
                'estimator__max_features': randint(2, 100),
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': randint(2, 50),
                'estimator__min_samples_leaf': randint(5, 25),
            }
        },

        'RandomForest_v4': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 80,
            'hyperparam_space': {
                'estimator__max_features': randint(2, 100),
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': randint(2, 50),
                'estimator__ccp_alpha': loguniform(1e-6, 0.1),
                'estimator__min_samples_leaf': randint(5, 25),
                'estimator__bootstrap': [True, False],
                'estimator__max_leaf_nodes': randint(1, 100)
            }
        },

        'XGBoost_extra_narrow': {
            'model': XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False, booster='gbtree', gamma=0), #silent=True,
            'n_search_iter': 80,
            'hyperparam_space': {
                'n_estimators': [150],
                'reg_lambda': uniform(30, 70),
                'colsample_bytree': uniform(0.9, 1.0),
                'max_features': [20],
                'learning_rate': [0.1],
                'max_depth': [5],
                'min_child_weight': uniform(0.15, 0.4),
                'max_delta_step': [10],
                'scale_pos_weight': uniform(10.5, 12),
                'subsample': [1.0],
                'colsample_bynode': [1.0]

            }
        },

        'XGBoost_narrow': {
            'model': XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False, booster='gbtree', gamma=0), #silent=True,
            'n_search_iter': 80,
            'hyperparam_space': {
                'n_estimators': randint(150, 300),
                'reg_lambda': uniform(30, 100),
                'colsample_bytree': uniform(0.9, 1.0),
                'max_features': randint(10, 20),
                'learning_rate': uniform(0.05, 0.1),
                'max_depth': randint(8, 10),
                'min_child_weight': uniform(0.15, 0.4),
                'max_delta_step': [10],
                'scale_pos_weight': uniform(10.5, 18.2),
                'subsample': uniform(0.8, 1),
                'colsample_bynode': uniform(0.8, 1)

            }
        },

        'XGBoost_broad': {
            'model': XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False, booster='gbtree', gamma=0), #silent=True,
            'n_search_iter': 80,
            'hyperparam_space': {
                'n_estimators': randint(150, 500),
                'reg_lambda': uniform(30, 1000),
                'colsample_bytree': uniform(0.8, 1.0),
                'max_features': randint(10, 50),
                'learning_rate': loguniform(1e-2, 0.2),
                'max_depth': randint(3, 20),
                'min_child_weight': uniform(0.15, 0.85),
                'max_delta_step': uniform(8, 12),
                'scale_pos_weight': uniform(7.5, 18.2),
                'subsample': uniform(0.6, 1),
                'colsample_bynode': uniform(0.6, 1)

            }
        },


        'XGBoost_narrow_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False,
                                         booster='gbtree', gamma=0))]),

            # silent=True,
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__model__n_estimators': randint(150, 300),
                'model__reg_lambda': uniform(30, 100),
                'model__colsample_bytree': uniform(0.9, 1.0),
                'model__max_features': randint(10, 20),
                'model__learning_rate': uniform(0.05, 0.1),
                'model__max_depth': randint(8, 10),
                'model__min_child_weight': uniform(0.15, 0.4),
                'model__max_delta_step': [10],
                'model__scale_pos_weight': uniform(10.5, 18.2),
                'model__subsample': uniform(0.8, 1),
                'model__colsample_bynode': uniform(0.8, 1),
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
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 500),
                'model__reg_lambda': uniform(30, 1000),
                'model__colsample_bytree': uniform(0.8, 1.0),
                'model__max_features': randint(10, 50),
                'model__learning_rate': loguniform(1e-2, 0.2),
                'model__max_depth': randint(3, 20),
                'model__min_child_weight': uniform(0.15, 0.85),
                'model__max_delta_step': uniform(8, 12),
                'model__scale_pos_weight': uniform(7.5, 18.2),
                'model__subsample': uniform(0.6, 1),
                'model__colsample_bynode': uniform(0.6, 1),
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
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__model__n_estimators': randint(150, 300),
                'model__reg_lambda': uniform(30, 100),
                'model__colsample_bytree': uniform(0.9, 1.0),
                'model__max_features': randint(10, 20),
                'model__learning_rate': uniform(0.05, 0.1),
                'model__max_depth': randint(8, 10),
                'model__min_child_weight': uniform(0.15, 0.4),
                'model__max_delta_step': [10],
                'model__scale_pos_weight': uniform(10.5, 18.2),
                'model__subsample': uniform(0.8, 1),
                'model__colsample_bynode': uniform(0.8, 1),
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
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 500),
                'model__reg_lambda': uniform(30, 1000),
                'model__colsample_bytree': uniform(0.8, 1.0),
                'model__max_features': randint(10, 50),
                'model__learning_rate': loguniform(1e-2, 0.2),
                'model__max_depth': randint(3, 20),
                'model__min_child_weight': uniform(0.15, 0.85),
                'model__max_delta_step': uniform(8, 12),
                'model__scale_pos_weight': uniform(7.5, 18.2),
                'model__subsample': uniform(0.6, 1),
                'model__colsample_bynode': uniform(0.6, 1),
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
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__model__n_estimators': randint(150, 300),
                'model__reg_lambda': uniform(30, 100),
                'model__colsample_bytree': uniform(0.9, 1.0),
                'model__max_features': randint(10, 20),
                'model__learning_rate': uniform(0.05, 0.1),
                'model__max_depth': randint(8, 10),
                'model__min_child_weight': uniform(0.15, 0.4),
                'model__max_delta_step': [10],
                'model__scale_pos_weight': uniform(10.5, 18.2),
                'model__subsample': uniform(0.8, 1),
                'model__colsample_bynode': uniform(0.8, 1),
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
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 500),
                'model__reg_lambda': uniform(30, 1000),
                'model__colsample_bytree': uniform(0.8, 1.0),
                'model__max_features': randint(10, 50),
                'model__learning_rate': loguniform(1e-2, 0.2),
                'model__max_depth': randint(3, 20),
                'model__min_child_weight': uniform(0.15, 0.85),
                'model__max_delta_step': uniform(8, 12),
                'model__scale_pos_weight': uniform(7.5, 18.2),
                'model__subsample': uniform(0.6, 1),
                'model__colsample_bynode': uniform(0.6, 1),
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
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__model__n_estimators': randint(150, 300),
                'model__reg_lambda': uniform(30, 100),
                'model__colsample_bytree': uniform(0.9, 1.0),
                'model__max_features': randint(10, 20),
                'model__learning_rate': uniform(0.05, 0.1),
                'model__max_depth': randint(8, 10),
                'model__min_child_weight': uniform(0.15, 0.4),
                'model__max_delta_step': [10],
                'model__scale_pos_weight': uniform(10.5, 18.2),
                'model__subsample': uniform(0.8, 1),
                'model__colsample_bynode': uniform(0.8, 1),
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
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(150, 500),
                'model__reg_lambda': uniform(30, 1000),
                'model__colsample_bytree': uniform(0.8, 1.0),
                'model__max_features': randint(10, 50),
                'model__learning_rate': loguniform(1e-2, 0.2),
                'model__max_depth': randint(3, 20),
                'model__min_child_weight': uniform(0.15, 0.85),
                'model__max_delta_step': uniform(8, 12),
                'model__scale_pos_weight': uniform(7.5, 18.2),
                'model__subsample': uniform(0.6, 1),
                'model__colsample_bynode': uniform(0.6, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },


        'ComplementNaiveBayes_broad': {
            'model': ComplementNB(),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1e-4, 0.5),
                'estimator__norm': [True, False]
            }
        },

        'ComplementNaiveBayes_narrow': {
            'model': ComplementNB(),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1e-1, 2),
                'estimator__norm': [True, False]
            }
        },

        'ComplementNaiveBayes_narrowV2': {
            'model': ComplementNB(),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1e-1, 0.5),
                'estimator__norm': [True, False]
            }
        },

        'ComplementNaiveBayes_Broad_v2': {
            'model': ComplementNB(),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__alpha': loguniform(1e-3, 2),
                'estimator__norm': [True, False]
            }
        },

        'NaiveBayes_broad': {
            'model': MultinomialNB(fit_prior=False),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__alpha': loguniform(0.005, 0.4),
            }
        },

        'NaiveBayes_narrow': {
            'model': MultinomialNB(fit_prior=False),
            'n_search_iter': 30,
            'hyperparam_space': {
                'estimator__alpha': loguniform(0.1, 0.4),
            }
        },

        'NaiveBayes_narrow_v2': {
            'model': MultinomialNB(fit_prior=False),
            'n_search_iter': 50,
            'hyperparam_space': {
                'estimator__alpha': loguniform(0.2, 2),
            }
        }
    }


if __name__ == '__main__':
    print(MODEL_LIST.keys())
