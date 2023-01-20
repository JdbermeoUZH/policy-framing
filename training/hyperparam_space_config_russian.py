from scipy.stats import loguniform, randint, uniform
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline


MODEL_LIST = \
    {
        'LogisticRegression': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty=None, max_iter=int(1e6)))]
            ),
            'n_search_iter': 4,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
            }
        },

        'LogisticRegression_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty=None, max_iter=int(1e6)))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-6, 1e4)
            }
        },

        'LogisticRegression_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty=None, max_iter=int(1e6)))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LogisticRegression_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty=None, max_iter=int(1e6)))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LogisticRegression_SMOTE': {
            'model': Pipeline([
                ('up', SMOTE(k_neighbors=3)),
                ('preproc', StandardScaler(with_mean=False)),
                ('model', LogisticRegression(penalty=None, max_iter=int(1e6)))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'up__k_neighbors': [2, 3],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
            }
        },

        'LogisticRegression_SMOTE_v2': {
            'model': Pipeline([
                ('up', SMOTE(k_neighbors=3)),
                ('preproc', StandardScaler(with_mean=False)),
                ('model', LogisticRegression(penalty=None, max_iter=int(1e6)))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'up__k_neighbors': [2, 3],
                'up__sampling_strategy': uniform(0.7, 0.3),
            }
        },

        'LogisticRegression_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE(k_neighbors=3)),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty=None, max_iter=int(1e6)))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': randint(2, 10),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegression_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE(k_neighbors=3)),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty=None, max_iter=int(1e6)))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': randint(2, 10),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegression_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE(k_neighbors=3)),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty=None, max_iter=int(1e6)))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': randint(2, 10),
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'LogisticRegression_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE(k_neighbors=3)),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty=None, max_iter=int(1e6)))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': randint(2, 10),
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'LogisticRegressionRidgeV1': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', max_iter=100000),
            'n_search_iter': 80,
            'hyperparam_space': {
               'C': loguniform(1e-6, 0.3)
            }
        },

        'LogisticRegressionRidgeV2': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', max_iter=100000),
            'n_search_iter': 80,
            'hyperparam_space': {
               'C': loguniform(1e-6, 1.2)
            }
        },

        'LogisticRegressionRidge_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-6, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LogisticRegressionRidge_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-6, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionRidge_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-6, 1),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionRidge_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-6, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionRidge_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
               'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-6, 10),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionRidge_SMOTE_v3': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
               'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-6, 10),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionRidge_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
               'model__C': loguniform(1e-6, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'LogisticRegressionRidge_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
               'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-6, 10),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegressionRidge_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
               'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-6, 10),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        # Might need to comment these again (these tended to always fail with few search iterations).
        'LogisticRegressionRidge_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced',
                                              max_iter=int(1e6)))]),
            'n_search_iter': 50,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-6, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
            }
        },

        'LogisticRegressionRidge_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-6, 10),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'LogisticRegressionRidge_SVMSMOTE_v3': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-6, 10),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'LogisticRegressionRidgeDual': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                        max_iter=100000),
            'n_search_iter': 60,
            'hyperparam_space': {
               'C': loguniform(1e-4, 0.4),
            }
        },

        'LogisticRegressionRidgeDual_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-4, 0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        # Takes waaaay too long
        'LogisticRegressionRidgeDual_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 10),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionRidgeDual_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 10),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionRidgeDual_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-4, 0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionRidgeDual_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionRidgeDual_SMOTE_v3': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionRidgeDual_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-4, 0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'LogisticRegressionRidgeDual_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegressionRidgeDual_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        # Might need to comment these again (these tended to always fail with few search iterations).
        'LogisticRegressionRidgeDual_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-4, 0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3]
            }
        },

        'LogisticRegressionRidgeDual_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'LogisticRegressionRidgeDual_SVMSMOTE_v3': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'LogisticRegressionLassoV1': {
            'model': LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000, class_weight='balanced'),
            'n_search_iter': 40,
            'hyperparam_space': {
                'C': loguniform(1, 10),
            }
        },

        'LogisticRegressionLassoV2': {
            'model': LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000, class_weight='balanced'),
            'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1, 1e4),
            }
        },

        'LogisticRegressionLasso_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1, 10),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LogisticRegressionLasso_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-1, 1e4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionLasso_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-1, 1e4),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionLasso_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1, 10),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionLasso_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-1, 1e4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionLasso_SMOTE_v3': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-1, 1e4),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionLasso_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1, 10),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'LogisticRegressionLasso_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-1, 1e4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegressionLasso_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-1, 1e4),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegressionLasso_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1, 10),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
            }
        },

        'LogisticRegressionLasso_BorderlineSVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-1, 1e4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'LogisticRegressionLasso_BorderlineSVMSMOTE_v3': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-1, 1e4),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'LogisticRegressionElasticNetV1': {
            'model':  Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                        class_weight='balanced'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-2, 5.5e3)
            }
        },

        'LogisticRegressionElasticNetV2': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                              class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-2, 800)
            }
        },

        'LogisticRegressionElasticNetV3': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                              class_weight='balanced'))]),
            'n_search_iter': 40,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1, 20)
            }
        },

        'LogisticRegressionElasticNet_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                              class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1, 20),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LogisticRegressionElasticNet_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 5.5e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionElasticNet_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 5.5e3),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionElasticNet_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                              class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1, 20),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionElasticNet_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 5.5e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionElasticNet_SMOTE_v3': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 5.5e3),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'LogisticRegressionElasticNet_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                              class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1, 20),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'LogisticRegressionElasticNet_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 5.5e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegressionElasticNet_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 5.5e3),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegressionElasticNet_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                              class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1, 20),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
            }
        },

        'LogisticRegressionElasticNet_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 5.5e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'LogisticRegressionElasticNet_SVMSMOTE_v3': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 5.5e3),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },


        ######################################################################################################

        'RidgeClassifierV1': {
            'model': RidgeClassifier(max_iter=100000, class_weight='balanced'),
            'n_search_iter': 60,
            'hyperparam_space': {
                'alpha': loguniform(1, 150)
            }
        },

        'RidgeClassifierV2': {
            'model': RidgeClassifier(max_iter=100000, class_weight='balanced'),
            'n_search_iter': 60,
            'hyperparam_space': {
                'alpha': loguniform(1, 5000)
            }
        },

        'RidgeClassifier_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1, 150),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'RidgeClassifier_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__alpha': loguniform(1e-1, 5e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'RidgeClassifier_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__alpha': loguniform(1e-1, 5e3),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'RidgeClassifier_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1, 150),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'RidgeClassifier_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__alpha': loguniform(1e-1, 5e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'RidgeClassifier_SMOTE_v3': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__alpha': loguniform(1e-1, 5e3),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'RidgeClassifier_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1, 150),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'RidgeClassifier_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__alpha': loguniform(1e-1, 5e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'RidgeClassifier_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__alpha': loguniform(1e-1, 5e3),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'RidgeClassifier_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1, 150),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
            }
        },

        'RidgeClassifier_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None],
                'model__alpha': loguniform(1e-1, 5e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'RidgeClassifier_SVMSMOTE_v3': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__alpha': loguniform(1e-1, 5e3),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'SVM_rbf_V1': {
            'model': SVC(kernel='rbf', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'C': loguniform(30, 800),
                'gamma': loguniform(5e-4, 0.04),
            }
        },

        'SVM_rbf_V2': {
            'model': SVC(kernel='rbf', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'C': loguniform(1e2, 1e4),
                'gamma': loguniform(1e-6, 1e-2),
            }
        },

        'SVM_rbf_V3': {
            'model': SVC(kernel='rbf', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'C': loguniform(1e2, 1e4),
                'gamma': loguniform(1e-4, 1e-2),
            }
        },

        'SVM_rbf_V4': {
            'model': SVC(kernel='rbf', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'C': loguniform(1, 2e3),
                'gamma': loguniform(1e-2, 1e-1),
            }
        },

        'SVM_rbf_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf', class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(30, 800),
                'model__gamma': loguniform(5e-4, 0.04),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'SVM_rbf_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(0.1, 1e5),
                'model__gamma': loguniform(1e-6, 1e-2),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'SVM_rbf_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.1, 1e5),
                'model__gamma': loguniform(1e-6, 1e-2),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'SVM_rbf_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf', class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(30, 800),
                'model__gamma': loguniform(5e-4, 0.04),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'SVM_rbf_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1, 1e4),
                'model__gamma': loguniform(1e-6, 1e-2),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'SVM_rbf_SMOTE_v3': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1, 1e4),
                'model__gamma': loguniform(1e-6, 1e-2),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'SVM_rbf_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf', class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(30, 800),
                'model__gamma': loguniform(5e-4, 0.04),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'SVM_rbf_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1, 1e4),
                'model__gamma': loguniform(1e-6, 1e-2),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'SVM_rbf_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1, 1e4),
                'model__gamma': loguniform(1e-6, 1e-2),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'SVM_rbf_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf', class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(30, 800),
                'model__gamma': loguniform(5e-4, 0.04),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
            }
        },

        'SVM_rbf_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1, 1e4),
                'model__gamma': loguniform(1e-6, 1e-2),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'SVM_rbf_SVMSMOTE_v3': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1, 1e4),
                'model__gamma': loguniform(1e-6, 1e-2),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'SVM_sigmoid_narrow_gamma': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'C': loguniform(1e-2, 600),
                'gamma': loguniform(1e-2, 500),
            }
        },

        'SVM_sigmoid_broader_gamma': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'C': loguniform(1e-2, 600),
                'gamma': loguniform(1e-4, 500),
            }
        },

        'SVM_sigmoid_V3': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            'n_search_iter': 50,
            'hyperparam_space': {
                'C': loguniform(150, 2000),
                'gamma': loguniform(1e-2, 0.1),
            }
        },

        'SVM_sigmoid_V4': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(100, 1e4),
                'gamma': loguniform(1e-2, 0.1),
            }
        },

        'SVM_sigmoid_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-2, 600),
                'model__gamma': loguniform(1e-2, 500),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'SVM_sigmoid_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(100, 1e4),
                'model__gamma': loguniform(1e-4, 500),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'SVM_sigmoid_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(100, 1e4),
                'model__gamma': loguniform(1e-4, 500),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'SVM_sigmoid_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-2, 600),
                'model__gamma': loguniform(1e-2, 500),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'SVM_sigmoid_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(100, 1e4),
                'model__gamma': loguniform(1e-4, 500),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

         'SVM_sigmoid_SMOTE_v3': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(100, 1e4),
                'model__gamma': loguniform(1e-4, 500),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'SVM_sigmoid_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-2, 600),
                'model__gamma': loguniform(1e-2, 500),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'SVM_sigmoid_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(100, 1e4),
                'model__gamma': loguniform(1e-4, 500),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'SVM_sigmoid_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(100, 1e4),
                'model__gamma': loguniform(1e-4, 500),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'SVM_sigmoid_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-2, 600),
                'model__gamma': loguniform(1e-2, 500),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3]}
        },

        'SVM_sigmoid_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(100, 1e4),
                'model__gamma': loguniform(1e-4, 500),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

         'SVM_sigmoid_SVMSMOTE_v3': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(100, 1e4),
                'model__gamma': loguniform(1e-4, 500),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'LinearSVM_V1': {
            'model': LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=100000),
            'n_search_iter': 40,
            'hyperparam_space': {
               'C': loguniform(1e-6, 5e-3),
            }
        },

        'LinearSVM_V2': {
            'model': LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=100000),
            'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1e-4, 0.5),
            }
        },

        'LinearSVM_V3': {
            'model': LinearSVC(dual=False, penalty='l2', max_iter=100000),
            'n_search_iter': 80,
            'hyperparam_space': {
                'class_weight': ['balanced', None],
                'C': loguniform(1e-4, 3e3),
            }
        },

        'LinearSVM_V4': {
            'model': LinearSVC(dual=False, class_weight='balanced', max_iter=100000),
            'n_search_iter': 100,
            'hyperparam_space': {
                'penalty': ['l1', 'l2'],
                'class_weight': ['balanced', None],
                'C': loguniform(1e-4, 3e3),
            }
        },

        'LinearSVM_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-5, 0.5),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LinearSVM_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 3e3)
            }
        },

        'LinearSVM_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 3e3)
            }
        },

        'LinearSVM_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-5, 0.5),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'LinearSVM_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-4, 3e3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'LinearSVM_SMOTE_v3': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-4, 3e3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'LinearSVM_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-5, 0.5),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'LinearSVM_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-4, 3e3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LinearSVM_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-4, 3e3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LinearSVM_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-5, 0.5),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
            }
        },

        'LinearSVM_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-4, 3e3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'LinearSVM_SVMSMOTE_v3': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=False, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-4, 3e3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'LinearSVMDual': {
            'model': LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=100000),
            'n_search_iter': 60,
            'hyperparam_space': {
                'C': loguniform(1e-6, 0.5),
            }
        },

        'LinearSVMDual_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-5, 3e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LinearSVMDual_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LinearSVMDual_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LinearSVMDual_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-5, 3e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'LinearSVMDual_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

         'LinearSVMDual_SMOTE_v3': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'LinearSVMDual_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-5, 3e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'LinearSVMDual_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LinearSVMDual_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LinearSVMDual_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-5, 3e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
            }
        },

        'LinearSVMDual_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'LinearSVMDual_SVMSMOTE_v3': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'RandomForestSK_V1': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 50,
            'hyperparam_space': {
                'max_features': ['sqrt', 'log2'],
                'n_estimators': [50, 100, 200],
                'max_depth': randint(2, 25),
                'min_samples_leaf': randint(5, 35),
            }
        },

        'RandomForestSK_V2': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 30,
            'hyperparam_space': {
                'max_features': randint(1, 50),
                'n_estimators': [50, 100, 200],
                'max_depth': randint(2, 25),
                'min_samples_leaf': randint(5, 35),
            }
        },

        'RandomForestSK_V3': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 70,
            'hyperparam_space': {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': randint(1, 50),
                'n_estimators': [50, 100, 200],
                'max_depth': randint(2, 100),
                'min_samples_leaf': randint(5, 35),
                'ccp_alpha': loguniform(1e-6, 5e-1),

            }
        },

        'RandomForest_ROS_v0': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__max_features': ['sqrt', 'log2'],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 25),
                'model__min_samples_leaf': randint(5, 35),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'RandomForest_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 25),
                'model__min_samples_leaf': randint(5, 35),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'RandomForest_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 100),
                'model__min_samples_leaf': randint(5, 35),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'RandomForest_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 100),
                'model__min_samples_leaf': randint(5, 35),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'RandomForest_SMOTE_v0': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__max_features': ['sqrt', 'log2'],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 25),
                'model__min_samples_leaf': randint(5, 35),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'RandomForest_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 25),
                'model__min_samples_leaf': randint(5, 35),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'RandomForest_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 100),
                'model__min_samples_leaf': randint(5, 35),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

         'RandomForest_SMOTE_v3': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 100),
                'model__min_samples_leaf': randint(5, 35),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'RandomForest_BorderlineSMOTE_v0': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__max_features': ['sqrt', 'log2'],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 25),
                'model__min_samples_leaf': randint(5, 35),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'RandomForest_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 25),
                'model__min_samples_leaf': randint(5, 35),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'RandomForest_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 100),
                'model__min_samples_leaf': randint(5, 35),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'RandomForest_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 100),
                'model__min_samples_leaf': randint(5, 35),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'RandomForest_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 25),
                'model__min_samples_leaf': randint(5, 35),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
            }
        },

        'RandomForest_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 100),
                'model__min_samples_leaf': randint(5, 35),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

         'RandomForest_SVMSMOTE_v3': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__max_features': randint(1, 50),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 100),
                'model__min_samples_leaf': randint(5, 35),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'XGBoost_narrow': {
            'model': XGBClassifier(verbosity=0, tree_method='hist', use_label_encoder=False, booster='gbtree', gamma=0), #silent=True,
            'n_search_iter': 80,
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
            'n_search_iter': 80,
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
            'n_search_iter': 80,
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
            'n_search_iter': 80,
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
                'up__sampling_strategy': uniform(0.7, 0.3),
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
            'n_search_iter': 80,
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
                'up__sampling_strategy': uniform(0.7, 0.3),
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
            'n_search_iter': 80,
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
                'up__sampling_strategy': uniform(0.7, 0.3),
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
            'n_search_iter': 80,
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
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'ComplementNaiveBayes': {
            'model': ComplementNB(),
            'n_search_iter': 80,
            'hyperparam_space': {
                'alpha': loguniform(1e-5, 1),
                'norm': [True, False]
            }
        },

        'ComplementNaiveBayes_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'ComplementNaiveBayes_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },


        'ComplementNaiveBayes_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'ComplementNaiveBayes_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'ComplementNaiveBayes_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'ComplementNaiveBayes_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'ComplementNaiveBayes_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'ComplementNaiveBayes_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3]
            }
        },

        'ComplementNaiveBayes_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'NaiveBayes': {
            'model': MultinomialNB(fit_prior=False),
            'n_search_iter': 80,
            'hyperparam_space': {
                'alpha': loguniform(1e-5, 2),
            }
        },

        'NaiveBayes_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB(fit_prior=False))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'NaiveBayes_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'NaiveBayes_ROS_v3': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'NaiveBayes_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB(fit_prior=False))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'NaiveBayes_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'NaiveBayes_SMOTE_v3': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3]
            }
        },

        'NaiveBayes_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB(fit_prior=False))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'NaiveBayes_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

         'NaiveBayes_BorderlineSMOTE_v3': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'NaiveBayes_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB(fit_prior=False))]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7]
            }
        },

        'NaiveBayes_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False], 
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'NaiveBayes_SVMSMOTE_v3': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', MultinomialNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        }

    }

if __name__ == '__main__':
    print(MODEL_LIST.keys())
