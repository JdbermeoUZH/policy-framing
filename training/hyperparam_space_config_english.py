from scipy.stats import loguniform, randint
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline


MODEL_LIST = \
    {
        'LogisticRegression': {
            'model': LogisticRegression(penalty='none', class_weight='balanced', max_iter=100000),
            #'n_search_iter': 5,
            'hyperparam_space': {
                'random_state': randint(0, 1000)
            }
        },

        'LogisticRegression_ROS_v1': {
            'model': Pipeline([('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                               ('model', LogisticRegression(penalty='none', max_iter=100000))]),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-6, 1e4)
            }
        },

        'LogisticRegression_ROS_v2': {
            'model': Pipeline([('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                               ('model', LogisticRegression(penalty='none', max_iter=100000))]),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LogisticRegression_SMOTE': {
            'model': Pipeline([('up', SMOTE()), ('model', LogisticRegression(penalty='none', max_iter=100000))]),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LogisticRegression_BorderlineSMOTE': {
            'model': Pipeline([('up', BorderlineSMOTE()), ('model', LogisticRegression(penalty='none', max_iter=100000))]),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegression_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('model', LogisticRegression(penalty='none', max_iter=100000))]),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },


        ######################################################################################################
        'LogisticRegressionRidge': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', max_iter=100000),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1e-4, 1)
            }
        },

        'LogisticRegressionRidge_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LogisticRegressionRidge_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            #'n_search_iter': 60,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionRidge_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LogisticRegressionRidge_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            #'n_search_iter': 60,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LogisticRegressionRidge_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'LogisticRegressionRidge_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            #'n_search_iter': 60,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegressionRidge_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
            }
        },

        'LogisticRegressionRidge_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', max_iter=100000))]),
            #'n_search_iter': 60,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'LogisticRegressionRidgeDual': {
            'model': LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                        max_iter=100000),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1e-4, 1)
            }
        },

        'LogisticRegressionRidgeDual_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                              max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LogisticRegressionRidgeDual_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=1e6))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionRidgeDual_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                              max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LogisticRegressionRidgeDual_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LogisticRegressionRidgeDual_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                              max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'LogisticRegressionRidgeDual_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegressionRidgeDual_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                              max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20)
            }
        },

        'LogisticRegressionRidgeDual_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'LogisticRegressionLassoV1': {
            'model': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=100000),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1, 20),
            }
        },

        'LogisticRegressionLassoV2': {
            'model': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=100000),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1, 10),
            }
        },

        'LogisticRegressionLassoV3': {
            'model': LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=100000),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1e-2, 100),
            }
        },

        'LogisticRegressionLasso_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', 
                                              max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-2, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LogisticRegressionLasso_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', dual=True, max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionLasso_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-2, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LogisticRegressionLasso_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', dual=True, max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LogisticRegressionLasso_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-2, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'LogisticRegressionLasso_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', dual=True, max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegressionLasso_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-2, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
            }
        },

        'LogisticRegressionLasso_BorderlineSVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', dual=True, max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },



        ######################################################################################################

        'LogisticRegressionElasticNetV1': {
            'model': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                        class_weight='balanced'),
            #'n_search_iter': 20,
            'hyperparam_space': {
                'C': loguniform(1, 7),
            }
        },

        'LogisticRegressionElasticNetV2': {
            'model': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                        class_weight='balanced'),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1e-2, 100),
            }
        },

        'LogisticRegressionElasticNetV3': {
            'model': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                        class_weight='balanced'),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1e-3, 1000),
            }
        },
        
        'LogisticRegressionElasticNet_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000, 
                                              class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1, 7),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LogisticRegressionElasticNet_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-3, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionElasticNet_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                              class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1, 7),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LogisticRegressionElasticNet_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-3, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LogisticRegressionElasticNet_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                              class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1, 7),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'LogisticRegressionElasticNet_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-3, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LogisticRegressionElasticNet_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000,
                                              class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1, 7),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
            }
        },

        'LogisticRegressionElasticNet_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-3, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },



        ######################################################################################################
        'RidgeClassifierV1': {
            'model': RidgeClassifier(max_iter=100000, class_weight='balanced'),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'alpha': loguniform(1, 2e3)
            }
        },

        'RidgeClassifierV2': {
            'model': RidgeClassifier(max_iter=100000, class_weight='balanced'),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'alpha': loguniform(1e-2, 1e5)
            }
        },
        
        'RidgeClassifier_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1, 2e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'RidgeClassifier_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', RidgeClassifier(max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 1e5),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'RidgeClassifier_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1, 2e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'RidgeClassifier_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', RidgeClassifier(max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 1e5),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'RidgeClassifier_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1, 2e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'RidgeClassifier_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', RidgeClassifier(max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 1e5),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'RidgeClassifier_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1, 2e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
            }
        },

        'RidgeClassifier_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', RidgeClassifier(max_iter=100000))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 1e5),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################
        'SVM_rbfV1': {
            'model': SVC(kernel='rbf', class_weight='balanced'),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(700, 1e3),
                'gamma': loguniform(5e-4, 1e-3),
            }
        },

        'SVM_rbfV2': {
            'model': SVC(kernel='rbf', class_weight='balanced'),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1, 1e3),
                'gamma': loguniform(1e-5, 1e-3),
            }
        },

        'SVM_rbfV3': {
            'model': SVC(kernel='rbf', class_weight='balanced'),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(0.1, 1e5),
                'gamma': loguniform(1e-5, 1e-3),
            }
        },

        'SVM_rbf_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', SVC(kernel='rbf', class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(700, 1e3),
                'model__gamma': loguniform(5e-4, 1e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'SVM_rbf_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', SVC(kernel='rbf'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.1, 1e5),
                'model__gamma': loguniform(1e-5, 1e-3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'SVM_rbf_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', SVC(kernel='rbf', class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(700, 1e3),
                'model__gamma': loguniform(5e-4, 1e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'SVM_rbf_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', SVC(kernel='rbf'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.1, 1e5),
                'model__gamma': loguniform(1e-5, 1e-3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'SVM_rbf_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', SVC(kernel='rbf', class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(700, 1e3),
                'model__gamma': loguniform(5e-4, 1e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'SVM_rbf_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', SVC(kernel='rbf'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.1, 1e5),
                'model__gamma': loguniform(1e-5, 1e-3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'SVM_rbf_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', SVC(kernel='rbf', class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(700, 1e3),
                'model__gamma': loguniform(5e-4, 1e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
            }
        },

        'SVM_rbf_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', SVC(kernel='rbf'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.1, 1e5),
                'model__gamma': loguniform(1e-5, 1e-3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },


        ######################################################################################################
        'SVM_sigmoidV1': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            #'n_search_iter': 30,
            'hyperparam_space': {
                'C': loguniform(0.05, 8),
                'gamma': loguniform(1e-1, 8),
            }
        },

        'SVM_sigmoidV2': {
            'model': SVC(kernel='sigmoid', class_weight='balanced'),
            #'n_search_iter': 100,
            'hyperparam_space': {
                'C': loguniform(0.05, 100),
                'gamma': loguniform(1e-1, 100),
            }
        },
        
        'SVM_sigmoid_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.05, 8),
                'model__gamma': loguniform(1e-1, 8),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'SVM_sigmoid_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', SVC(kernel='sigmoid'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.05, 100),
                'model__gamma': loguniform(1e-1, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'SVM_sigmoid_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.05, 8),
                'model__gamma': loguniform(1e-1, 8),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'SVM_sigmoid_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.05, 100),
                'model__gamma': loguniform(1e-1, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'SVM_sigmoid_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.05, 8),
                'model__gamma': loguniform(1e-1, 8),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'SVM_sigmoid_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.05, 100),
                'model__gamma': loguniform(1e-1, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'SVM_sigmoid_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.05, 8),
                'model__gamma': loguniform(1e-1, 8),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20)            }
        },

        'SVM_sigmoid_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(0.05, 100),
                'model__gamma': loguniform(1e-1, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },


        ######################################################################################################
        'LinearSVM_V1': {
            'model': LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=1e5),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1e-6, 5e-3),
            }
        },

        'LinearSVM_V2': {
            'model': LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=1e5),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1e-6, 1),
            }
        },

        'LinearSVM_V3': {
            'model': LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=1e5),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'C': loguniform(1e-4, 1e2),
            }
        },

        'LinearSVM_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=1e5))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-6, 5e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LinearSVM_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', LinearSVC(dual=False, max_iter=1e5))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1e2),
                'model__penalty': ['l1', 'l2'],
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LinearSVM_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=1e5))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-6, 5e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LinearSVM_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LinearSVC(dual=False, max_iter=1e5))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1e2),
                'model__penalty': ['l1', 'l2'],
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LinearSVM_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=1e5))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-6, 5e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LinearSVM_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LinearSVC(dual=False, max_iter=1e5))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1e2),
                'model__penalty': ['l1', 'l2'],
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LinearSVM_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LinearSVC(dual=False, class_weight='balanced', penalty='l2', max_iter=1e5))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-6, 5e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20)
            }
        },

        'LinearSVM_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LinearSVC(dual=False, max_iter=1e5))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-4, 1e2),
                'model__penalty': ['l1', 'l2'],
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################

        'LinearSVMDualV1': {
            'model': LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'C': loguniform(1e-5, 3e-3),
            }
        },

        'LinearSVMDualV2': {
            'model': LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'C': loguniform(1e-5, 1),
            }
        },

        'LinearSVMDualV3': {
            'model': LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'C': loguniform(1e-5, 100),
            }
        },

        'LinearSVMDual_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-5, 3e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': [None, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            }
        },

        'LinearSVMDual_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-5, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LinearSVMDual_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-5, 3e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LinearSVMDual_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-5, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LinearSVMDual_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-5, 3e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'LinearSVMDual_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-5, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LinearSVMDual_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', class_weight='balanced', max_iter=5e4))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-5, 3e-3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
            }
        },

        'LinearSVMDual_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__C': loguniform(1e-5, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },


        ######################################################################################################
        'RandomForest_v1': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'max_features': ['sqrt', 'log2'],
                'n_estimators': [50, 100, 200],
                'max_depth': randint(2, 50),
                'min_samples_leaf': randint(5, 25),
            }
        },

        'RandomForest_v2': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'max_features': ['sqrt', 'log2'],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'n_estimators': [50, 100, 200],
                'max_depth': randint(2, 100),
                'min_samples_leaf': randint(5, 25),
            }
        },

        'RandomForest_v3': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            #'n_search_iter': 60,
            'hyperparam_space': {
                'max_features': randint(2, 100),
                'n_estimators': [50, 100, 200],
                'max_depth': randint(2, 50),
                'min_samples_leaf': randint(5, 25),
            }
        },

        'RandomForest_v4': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'max_features': randint(2, 100),
                'n_estimators': [50, 100, 200],
                'max_depth': randint(2, 50),
                'ccp_alpha': loguniform(1e-6, 0.1),
                'min_samples_leaf': randint(5, 25),
                'bootstrap': [True, False],
                'max_leaf_nodes': randint(1, 100)
            }
        },

        'RandomForest_ROS_v0': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', RandomForestClassifier())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__max_features': ['sqrt', 'log2'],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 100),
                'model__min_samples_leaf': randint(5, 25),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'RandomForest_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', RandomForestClassifier())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__max_features': randint(2, 100),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 50),
                'model__min_samples_leaf': randint(5, 25),
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'RandomForest_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', RandomForestClassifier())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__max_features': randint(2, 100),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 50),
                'model__ccp_alpha': loguniform(1e-6, 0.1),
                'model__min_samples_leaf': randint(5, 25),
                'model__bootstrap': [True, False],
                'max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'RandomForest_SMOTE_v0': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', RandomForestClassifier())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__max_features': ['sqrt', 'log2'],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 100),
                'min_samples_leaf': randint(5, 25),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'RandomForest_SMOTE_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', RandomForestClassifier())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__max_features': randint(2, 100),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 50),
                'model__min_samples_leaf': randint(5, 25),
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'RandomForest_SMOTE_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', RandomForestClassifier())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__max_features': randint(2, 100),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 50),
                'model__ccp_alpha': loguniform(1e-6, 0.1),
                'model__min_samples_leaf': randint(5, 25),
                'model__bootstrap': [True, False],
                'max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'RandomForest_BorderlineSMOTE_v0': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', RandomForestClassifier())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__max_features': ['sqrt', 'log2'],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 100),
                'min_samples_leaf': randint(5, 25),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'RandomForest_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', RandomForestClassifier())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__max_features': randint(2, 100),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 50),
                'model__min_samples_leaf': randint(5, 25),
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'RandomForest_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', RandomForestClassifier())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__max_features': randint(2, 100),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 50),
                'model__ccp_alpha': loguniform(1e-6, 0.1),
                'model__min_samples_leaf': randint(5, 25),
                'model__bootstrap': [True, False],
                'max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'RandomForest_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', RandomForestClassifier())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__max_features': randint(2, 100),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 50),
                'model__min_samples_leaf': randint(5, 25),
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
            }
        },

        'RandomForest_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', RandomForestClassifier())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__class_weight': ['balanced', None, "balanced_subsample"],
                'model__max_features': randint(2, 100),
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 50),
                'model__ccp_alpha': loguniform(1e-6, 0.1),
                'model__min_samples_leaf': randint(5, 25),
                'model__bootstrap': [True, False],
                'max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },


        ######################################################################################################
        'XGBoost': {
            'model': XGBClassifier(verbosity=0, silent=True, use_label_encoder=False),
            ##'n_search_iter': 10,
            'hyperparam_space': {
                'max_features': ['sqrt', 'log2'],
                'gamma': loguniform(1e-5, 1e-2),
                'max_depth': randint(6, 30),
                'min_child_weight': randint(1, 6),
                'max_delta_step': [0, 1, 5],
            }
        },

        'XGBoostV2': {
            'model': XGBClassifier(verbosity=0, silent=True),
            # #'n_search_iter': 10,
            'hyperparam_space': {
                'max_features': randint(1, 20),
                'gamma': loguniform(1e-5, 1e-2),
                'max_depth': randint(6, 30),
                'min_child_weight': randint(1, 6),
                'max_delta_step': [0, 1, 5],
            }
        },

        ######################################################################################################
        'ComplementNaiveBayes_broad': {
            'model': ComplementNB(),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'alpha': loguniform(1e-4, 0.5),
                'norm': [True, False]
            }
        },

        'ComplementNaiveBayes_broadV2': {
            'model': ComplementNB(),
            #'n_search_iter': 100,
            'hyperparam_space': {
                'alpha': loguniform(1e-4, 100),
                'norm': [True, False]
            }
        },

        'ComplementNaiveBayes_narrow': {
            'model': ComplementNB(),
            #'n_search_iter': 30,
            'hyperparam_space': {
                'alpha': loguniform(1e-1, 0.5),
                'norm': [True, False]
            }
        },

        'ComplementNaiveBayes_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', ComplementNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-1, 0.5),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'ComplementNaiveBayes_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', ComplementNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-4, 100),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'ComplementNaiveBayes_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', ComplementNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-1, 0.5),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'ComplementNaiveBayes_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', ComplementNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-4, 100),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'ComplementNaiveBayes_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', ComplementNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-1, 0.5),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'ComplementNaiveBayes_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', ComplementNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-4, 100),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'ComplementNaiveBayes_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', ComplementNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-1, 0.5),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20)
            }
        },

        'ComplementNaiveBayes_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', ComplementNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-4, 100),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        },


        ######################################################################################################

        'NaiveBayes_broad': {
            'model': MultinomialNB(fit_prior=False),
            #'n_search_iter': 50,
            'hyperparam_space': {
                'alpha': loguniform(0.005, 0.4),
            }
        },

        'NaiveBayes_broadV2': {
            'model': MultinomialNB(fit_prior=False),
            #'n_search_iter': 100,
            'hyperparam_space': {
                'alpha': loguniform(1e-2, 100),
            }
        },

        'NaiveBayes_broadV3': {
            'model': MultinomialNB(fit_prior=False),
            #'n_search_iter': 100,
            'hyperparam_space': {
                'alpha': loguniform(1e-2, 100),
                'fit_prior': [True, False],
            }
        },

        'NaiveBayes_narrow': {
            'model': MultinomialNB(fit_prior=False),
            #'n_search_iter': 30,
            'hyperparam_space': {
                'alpha': loguniform(0.1, 0.4),
            }
        },

        'NaiveBayes_ROS_v1': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', MultinomialNB(fit_prior=False))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(0.1, 0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'NaiveBayes_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', MultinomialNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-2, 100),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'NaiveBayes_SMOTE_v1': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', MultinomialNB(fit_prior=False))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(0.1, 0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'NaiveBayes_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', MultinomialNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-2, 100),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'NaiveBayes_BorderlineSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', MultinomialNB(fit_prior=False))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(0.1, 0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1']
            }
        },

        'NaiveBayes_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', MultinomialNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-2, 100),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'NaiveBayes_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', MultinomialNB(fit_prior=False))]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(0.1, 0.4),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20)
            }
        },

        'NaiveBayes_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', MultinomialNB())]),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-2, 100),
                'model__fit_prior': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__out_step': loguniform(1e-6, 1)
            }
        }
    }

if __name__ == '__main__':
    print(MODEL_LIST.keys())
