from scipy.stats import loguniform, randint, uniform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from skmultilearn.adapt import MLARAM, MLkNN, BRkNNbClassifier, BRkNNaClassifier
from skmultilearn.ensemble import RakelD
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline

min_samples_min_class = 3
up_sampling_prop_list = [0.99, 0.9925, 0.995, 0.999, 0.9, 0.95, 0.85]

MODEL_LIST = \
    {
        'LogisticRegression': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty=None, class_weight='balanced', max_iter=100000))]
            ),
            'n_search_iter': 60,
            'model_type': 'LogisticRegression',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced']
            }
        },

        'LogisticRegression_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty=None, class_weight='balanced', max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegression',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-6, 1)
            }
        },

        'LogisticRegression_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', LogisticRegression(penalty=None, class_weight='balanced', max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegression',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 7), min_samples_min_class)
            }
        },

        'LogisticRegression_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty=None, class_weight='balanced', max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegression',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class),
                'up__m_neighbors': randint(14, 30),
                'up__kind': ['borderline-2']
            }
        },

        'LogisticRegression_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty=None, class_weight='balanced', max_iter=100000))]),
            'n_search_iter': 150,
            'model_type': 'LogisticRegression',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class),
                'up__m_neighbors': randint(14, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        ######################################################################################################
        'LogisticRegressionRidgeDual': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', dual=True,
                                              max_iter=100000))]
            ),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-6, 0.4),
                'model__class_weight': ['balanced']
            }
        },

        'LogisticRegressionRidgeDual_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-6, 0.4),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-6, 1)
            }
        },

        'LogisticRegressionRidgeDual_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__C': loguniform(1e-6, 0.4),
                'up__sampling_strategy': ['minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 7), min_samples_min_class) 
            }
        },

        'LogisticRegressionRidgeDual_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__C': loguniform(1e-6, 0.4),
                'up__sampling_strategy': ['minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class),
                'up__m_neighbors': randint(14, 30),
                'up__kind': ['borderline-2']
            }
        },

        'LogisticRegressionRidgeDual_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 150,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-6, 0.4),
                'up__sampling_strategy': ['minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class),
                'up__m_neighbors': randint(14, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        'RakelD_LogisticRegression': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', RakelD())
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'RakelD Partitioning of labels',
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__base_classifier': [LogisticRegression(penalty='l2', solver='liblinear', dual=True,
                                                              class_weight='balanced', max_iter=100000)],
                'model__base_classifier__C': loguniform(1e-6, 0.4),
                'model__base_classifier_require_dense': [False],
                'model__labelset_size': range(1, 6)
            }
        },

        ######################################################################################################
        'LogisticRegressionLasso': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]
            ),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionLasso',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__C': loguniform(1e-2, 100),
            }
        },

        'LogisticRegressionLasso_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', 
                                              max_iter=100000))]),
            'model_type': 'LogisticRegressionLasso',
            'model_subtype': 'Random Oversampling',
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-2, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-6, 1)
            }
        },

        'LogisticRegressionLasso_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionLasso',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__C': loguniform(1e-2, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 7), min_samples_min_class) 
            }
        },

        'LogisticRegressionLasso_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionLasso',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__C': loguniform(1e-2, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class),
                'up__m_neighbors': randint(14, 30),
                'up__kind': ['borderline-2']
            }
        },

        'LogisticRegressionLasso_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                                              max_iter=100000))]),
            'n_search_iter': 150,
            'model_type': 'LogisticRegressionLasso',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-2, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class),
                'up__m_neighbors': randint(14, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },


        ######################################################################################################
        'LogisticRegressionElasticNet': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000,
                                              class_weight='balanced'))]
            ),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionElasticNet',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__C': loguniform(1e-3, 1e2),
            }
        },

        'LogisticRegressionElasticNet_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000,
                                              class_weight='balanced'))]),
            'model_type': 'LogisticRegressionElasticNet',
            'model_subtype': 'Random Oversampling',
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-3, 1e2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-6, 1)
            }
        },

        'LogisticRegressionElasticNet_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000,
                                              class_weight='balanced'))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionElasticNet',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__C': loguniform(1e-3, 1e2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 7), min_samples_min_class) 
            }
        },

        'LogisticRegressionElasticNet_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000,
                                        class_weight='balanced'))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionElasticNet',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__C': loguniform(1e-3, 1e2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class),
                'up__m_neighbors': randint(14, 30),
                'up__kind': ['borderline-2']
            }
        },

        'LogisticRegressionElasticNet_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000,
                                              class_weight='balanced'))]),
            'n_search_iter': 150,
            'model_type': 'LogisticRegressionElasticNet',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-3, 1e2),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class),
                'up__m_neighbors': randint(14, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        ######################################################################################################
        'RidgeClassifier': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]
            ),
            'n_search_iter': 60,
            'model_type': 'RidgeClassifier',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__alpha': loguniform(1e-2, 5000)
            }
        },

        'RidgeClassifier_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            'model_type': 'RidgeClassifier',
            'model_subtype': 'Random Oversampling',
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__alpha': loguniform(1e-2, 5000),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-6, 1)
            }
        },

        'RidgeClassifier_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            'n_search_iter': 60,
            'model_type': 'RidgeClassifier',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__alpha': loguniform(1e-2, 5000),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 7), min_samples_min_class) 
            }
        },

        'RidgeClassifier_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            'n_search_iter': 60,
            'model_type': 'RidgeClassifier',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__alpha': loguniform(1e-2, 5000),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class),
                'up__m_neighbors': randint(14, 30),
                'up__kind': ['borderline-2']
            }
        },

        'RidgeClassifier_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', RidgeClassifier(max_iter=100000, class_weight='balanced'))]),
            'n_search_iter': 150,
            'model_type': 'RidgeClassifier',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced', None],
                'model__alpha': loguniform(1e-2, 5000),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class),
                'up__m_neighbors': randint(14, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        ######################################################################################################
        'SVM_rbf': {  # Best, but very similar to the rest
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf', class_weight='balanced'))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [False, True],
                'model__C': loguniform(1e1, 5e3),
                'model__gamma': loguniform(5e-5, 0.01),
            }
        },

        'SVM_rbf_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [False, True],
                'model__C': loguniform(1e1, 5e3),
                'model__gamma': loguniform(5e-5, 0.01),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-7, 1e-4)
            }
        },

        'SVM_rbf_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [False, True],
                'model__C': loguniform(1e1, 5e3),
                'model__gamma': loguniform(5e-5, 0.01),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                 'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class) 
            }
        },

        'SVM_rbf_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [False, True],
                'model__C': loguniform(1e1, 5e3),
                'model__gamma': loguniform(5e-5, 0.01),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__kind': ['borderline-2']
            }
        },

        'SVM_rbf_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 150,
            'model_type': 'SVM',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [False, True],
                'model__C': loguniform(1e1, 5e3),
                'model__gamma': loguniform(5e-5, 0.01),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        ######################################################################################################
        'SVM_sigmoid': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [False],
                'model__C': loguniform(1e-2, 600),
                'model__gamma': loguniform(1e-2, 500),
            }
        },

        'SVM_sigmoid_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [False],
                'model__C': loguniform(1e-2, 600),
                'model__gamma': loguniform(1e-2, 500),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-7, 1e-4)
            }
        },

        'SVM_sigmoid_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [False],
                'model__C': loguniform(1e-2, 600),
                'model__gamma': loguniform(1e-2, 500),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 5), min_samples_min_class) 
            }
        },

        'SVM_sigmoid_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [False],
                'model__C': loguniform(1e-2, 600),
                'model__gamma': loguniform(1e-2, 500),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__kind': ['borderline-2']
            }
        },

        'SVM_sigmoid_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 150,
            'model_type': 'SVM',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [False],
                'model__C': loguniform(1e-2, 600),
                'model__gamma': loguniform(1e-2, 500),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        'RakelD_SVM': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', RakelD())
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'SVM',
            'model_subtype': 'RakelD Partitioning of labels',
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__base_classifier': [SVC(kernel='sigmoid', class_weight='balanced')],
                'model__base_classifier__C': loguniform(1e-2, 600),
                'model__base_classifier__gamma': loguniform(1e-2, 500),
                'model__base_classifier_require_dense': [False],
                'model__labelset_size': range(1, 6)
            }
        },

        ######################################################################################################
        'LinearSVMDual': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000, class_weight='balanced'))]),
            'n_search_iter': 60,
            'model_type': 'LinearSVM',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-6, 0.5)
            }
        },

        'LinearSVMDual_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000))]),
            'n_search_iter': 60,
            'model_type': 'LinearSVM',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-7, 1e-3)
            }
        },

        'LinearSVMDual_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000))]),
            'n_search_iter': 60,
            'model_type': 'LinearSVM',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class)
            }
        },

        'LinearSVMDual_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000))]),
            'n_search_iter': 60,
            'model_type': 'LinearSVM',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LinearSVMDual_SVMSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SVMSMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000))]),
            'n_search_iter': 60,
            'model_type': 'LinearSVM',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-6, 0.5),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        'RakelD_LineaSVM': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', RakelD())
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'LinearSVM',
            'model_subtype': 'RakelD Partitioning of labels',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__base_classifier': [LinearSVC(dual=True, penalty='l2', max_iter=50000, class_weight='balanced')],
                'model__base_classifier__C': loguniform(1e-6, 0.5),
                'model__base_classifier_require_dense': [False],
                'model__labelset_size': range(1, 6)
            }
        },

        ######################################################################################################
        'kNN': {  
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', KNeighborsClassifier())]
            ),
            'n_search_iter': 60,
            'model_type': 'KNN',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__n_neighbors': randint(2, 30),
                'model__weights': ['uniform', 'distance'],
                'model__metric': ['cosine', 'l1', 'l2']
            }
        },

        'kNN_ROS': {  
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', KNeighborsClassifier())]
            ),
            'n_search_iter': 60,
            'model_type': 'KNN',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__n_neighbors': randint(2, 30),
                'model__weights': ['uniform', 'distance'],
                'model__metric': ['cosine', 'l1', 'l2'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
            }
        },

        'kNN_SMOTE': {  
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', KNeighborsClassifier())]
            ),
            'n_search_iter': 60,
            'model_type': 'KNN',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__n_neighbors': randint(2, 30),
                'model__weights': ['uniform', 'distance'],
                'model__metric': ['cosine', 'l1', 'l2'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class)
            }
        },

        'kNN_BorderlineSMOTE': {  
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', KNeighborsClassifier())]
            ),
            'n_search_iter': 60,
            'model_type': 'KNN',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__n_neighbors': randint(2, 30),
                'model__weights': ['uniform', 'distance'],
                'model__metric': ['cosine', 'l1', 'l2'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'kNN_SVMSMOTE': {  
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SVMSMOTE()),
                 ('model', KNeighborsClassifier())]
            ),
            'n_search_iter': 150,
            'model_type': 'KNN',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__n_neighbors': randint(2, 30),
                'model__weights': ['uniform', 'distance'],
                'model__metric': ['cosine', 'l1', 'l2'],
                'up__sampling_strategy': ['not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        ######################################################################################################
        'XGBoost_narrow': {
            'model': XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0),
            'model_type': 'XGBoost',
            'model_subtype': 'No Upsampling',
            'n_search_iter': 60,
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
            'model': XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0),
            'n_search_iter': 60,
            'model_type': 'XGBoost',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'n_estimators': randint(300, 400),
                'reg_lambda': loguniform(15, 100),
                'colsample_bytree': uniform(loc=0.65, scale=0.35),
                'max_features': randint(5, 40),
                'learning_rate': loguniform(2e-2, 0.15),
                'max_depth': randint(15, 40),
                'min_child_weight': loguniform(1e-3, 1),
                'max_delta_step': uniform(loc=2, scale=10),
                'scale_pos_weight': uniform(loc=1, scale=19),
                'subsample': uniform(loc=0.75, scale=0.24),
                'colsample_bynode': uniform(loc=0.7, scale=0.21)
            }
        },

        'XGBoost_narrow_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0))]),

            # silent=True,
            'n_search_iter': 60,
            'model_type': 'XGBoost',
            'model_subtype': 'Random Oversampling',
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
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-7, 1e-3)
            }
        },

        'XGBoost_broad_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0))]),

            # silent=True,
            'n_search_iter': 60,
            'model_type': 'XGBoost',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(300, 400),
                'model__reg_lambda': loguniform(15, 100),
                'model__colsample_bytree': uniform(loc=0.65, scale=0.35),
                'model__max_features': randint(5, 40),
                'model__learning_rate': loguniform(2e-2, 0.15),
                'model__max_depth': randint(15, 40),
                'model__min_child_weight': loguniform(1e-3, 1),
                'model__max_delta_step': uniform(loc=2, scale=10),
                'model__scale_pos_weight': uniform(loc=1, scale=19),
                'model__subsample': uniform(loc=0.75, scale=0.24),
                'model__colsample_bynode': uniform(loc=0.7, scale=0.21),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-7, 1e-3)
            }
        },

        'XGBoost_narrow_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0))]),

            # silent=True,
            'n_search_iter': 60,
            'model_type': 'XGBoost',
            'model_subtype': 'SMOTE',
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
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class)
            }
        },

        'XGBoost_broad_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0))]),

            # silent=True,
            'n_search_iter': 60,
            'model_type': 'XGBoost',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(300, 400),
                'model__reg_lambda': loguniform(15, 100),
                'model__colsample_bytree': uniform(loc=0.65, scale=0.35),
                'model__max_features': randint(5, 40),
                'model__learning_rate': loguniform(2e-2, 0.15),
                'model__max_depth': randint(15, 40),
                'model__min_child_weight': loguniform(1e-3, 1),
                'model__max_delta_step': uniform(loc=2, scale=10),
                'model__scale_pos_weight': uniform(loc=1, scale=19),
                'model__subsample': uniform(loc=0.75, scale=0.24),
                'model__colsample_bynode': uniform(loc=0.7, scale=0.21),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class)
            }
        },

        'XGBoost_narrow_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0))]),
            # silent=True,
            'n_search_iter': 60,
            'model_type': 'XGBoost',
            'model_subtype': 'BorderlineSMOTE',
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
                'up__sampling_strategy': ['not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'XGBoost_broad_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0))]),

            # silent=True,
            'n_search_iter': 60,
            'model_type': 'XGBoost',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(300, 400),
                'model__reg_lambda': loguniform(15, 100),
                'model__colsample_bytree': uniform(loc=0.65, scale=0.35),
                'model__max_features': randint(5, 40),
                'model__learning_rate': loguniform(2e-2, 0.15),
                'model__max_depth': randint(15, 40),
                'model__min_child_weight': loguniform(1e-3, 1),
                'model__max_delta_step': uniform(loc=2, scale=10),
                'model__scale_pos_weight': uniform(loc=1, scale=19),
                'model__subsample': uniform(loc=0.75, scale=0.24),
                'model__colsample_bynode': uniform(loc=0.7, scale=0.21),
                'up__sampling_strategy': ['not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'XGBoost_narrow_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0))]),

            # silent=True,
            'n_search_iter': 150,
            'model_type': 'XGBoost',
            'model_subtype': 'SVMSMOTE',
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
                'up__sampling_strategy': ['not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        'XGBoost_broad_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0))]),

            # silent=True,
            'n_search_iter': 150,
            'model_type': 'XGBoost',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__n_estimators': randint(300, 400),
                'model__reg_lambda': loguniform(15, 100),
                'model__colsample_bytree': uniform(loc=0.65, scale=0.35),
                'model__max_features': randint(5, 40),
                'model__learning_rate': loguniform(2e-2, 0.15),
                'model__max_depth': randint(15, 40),
                'model__min_child_weight': loguniform(1e-3, 1),
                'model__max_delta_step': uniform(loc=2, scale=10),
                'model__scale_pos_weight': uniform(loc=1, scale=19),
                'model__subsample': uniform(loc=0.75, scale=0.24),
                'model__colsample_bynode': uniform(loc=0.7, scale=0.21),
                'up__sampling_strategy': ['not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        ######################################################################################################
        'ComplementNaiveBayes': { 
            'model': ComplementNB(),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'alpha': loguniform(1e-5, 1),
                'norm': [False]
            }
        },

        'ComplementNaiveBayes_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', ComplementNB())]),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-6, 500)
            }
        },

        'ComplementNaiveBayes_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class)
            }
        },

        'ComplementNaiveBayes_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(12, 30),
                'up__kind': ['borderline-2']
            }
        },

        'ComplementNaiveBayes_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 150,
            'model_type': 'NaiveBayes',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'model__alpha': loguniform(1e-5, 1),
                'model__norm': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
            }
        },

        'RakelD_ComplementNB': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', RakelD())
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'ComplementNB',
            'model_subtype': 'RakelD Partitioning of labels',
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__base_classifier': [ComplementNB(norm=False)],
                'model__base_classifier__alpha': loguniform(1e-5, 1),
                'model__base_classifier_require_dense': [False],
                'model__labelset_size': range(1, 6)
            }
        },

        ######################################################################################################

        'NaiveBayes': {  
            'model': MultinomialNB(),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'alpha': loguniform(1e-5, 2),
                'fit_prior': [False],
            }
        },

        'NaiveBayes_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', MultinomialNB())]),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-6, 500)
            }
        },

        'NaiveBayes_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', MultinomialNB())]),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class)
            }
        },

        'NaiveBayes_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', MultinomialNB())]),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(12, 30),
                'up__kind': ['borderline-2']
            }
        },

        'NaiveBayes_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', MultinomialNB())]),
            'n_search_iter': 150,
            'model_type': 'NaiveBayes',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'model__alpha': loguniform(1e-5, 2),
                'model__fit_prior': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
            }
        },

        ######################################################################################################

        'RandomForest': {
            'model': RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1),
            'n_search_iter': 60,
            'model_type': 'RandomForest',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'max_features': randint(13, 60),
                'n_estimators': [50, 100, 200],
                'max_depth': randint(5, 80),
                'ccp_alpha': loguniform(1e-5, 0.01),
                'min_samples_leaf': randint(10, 15),
                'bootstrap': [False],
                'max_leaf_nodes': randint(5, 40)
            }
        },

        'RandomForest_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', RandomForestClassifier(n_jobs=-1))]),
            'n_search_iter': 60,
            'model_type': 'RandomForest',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'model__class_weight': ['balanced_subsample'],
                'model__max_features': randint(13, 60),
                'model__criterion': ['gini', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(5, 80),
                'model__min_samples_leaf': randint(10, 15),
                'model__ccp_alpha': loguniform(1e-5, 0.01),
                'model__bootstrap': [False],
                'model__max_leaf_nodes': randint(5, 40),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__shrinkage': loguniform(1e-7, 1e-3)
            }
        },

        'RandomForest_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', RandomForestClassifier(n_jobs=-1))]),
            'n_search_iter': 60,
            'model_type': 'RandomForest',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'model__class_weight': ['balanced_subsample'],
                'model__max_features': randint(13, 60),
                'model__criterion': ['gini', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(5, 80),
                'model__min_samples_leaf': randint(10, 15),
                'model__ccp_alpha': loguniform(1e-5, 0.01),
                'model__bootstrap': [False],
                'model__max_leaf_nodes': randint(5, 40),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
            }
        },

        'RandomForest_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', RandomForestClassifier(n_jobs=-1))]),
            'n_search_iter': 60,
            'model_type': 'RandomForest',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'model__class_weight': ['balanced_subsample'],
                'model__max_features': randint(13, 60),
                'model__criterion': ['gini', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(5, 80),
                'model__min_samples_leaf': randint(10, 15),
                'model__ccp_alpha': loguniform(1e-5, 0.01),
                'model__bootstrap': [False],
                'model__max_leaf_nodes': randint(5, 40),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__kind': ['borderline-2']
            }
        },

        'RandomForest_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', RandomForestClassifier(n_jobs=-1))]),
            'n_search_iter': 150,
            'model_type': 'RandomForest',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'model__class_weight': ['balanced_subsample'],
                'model__criterion': ['gini', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_features': randint(13, 60),
                'model__min_samples_leaf': randint(10, 15),
                'model__max_depth': randint(5, 80),
                'model__ccp_alpha': loguniform(1e-5, 0.01),
                'model__bootstrap': [False],
                'model__max_leaf_nodes': randint(5, 40),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'] + up_sampling_prop_list,
                'up__k_neighbors': randint(min(min_samples_min_class - 1, 3), min_samples_min_class),
                'up__m_neighbors': randint(3, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        #############################################################################################################
        #############################################################################################################
        ################################ Scikit-Multilearn models ###################################################

        'BRkNNaClassifier': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', BRkNNaClassifier())
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'Binary Relevance kNN',
            'model_subtype': 'Natively Multilabel',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__k': randint(2, 60)
            }
        },

        'BRkNNbClassifier': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', BRkNNbClassifier())
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'Binary Relevance kNN',
            'model_subtype': 'Natively Multilabel',
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__k': randint(2, 60)
            }
        },

        'MLkNN': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', MLkNN())
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'Multilabel k Nearest Neighbours',
            'model_subtype': 'Natively Multilabel',
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__k': randint(2, 60),
                'model__s': loguniform(1e-4, 1e2)
            }
        },

        'MLARAM': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', MLARAM())
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'Multi-label ARAM¶',
            'model_subtype': 'Natively Multilabel',
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__vigilance': uniform(0.75, 24.9999),
                'model__threshold': uniform(0.01, 0.1)
            }
        },
    }

if __name__ == '__main__':
    print(MODEL_LIST.keys())
