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

min_samples_min_class = 14
up_sampling_prop_list = [0.95, 0.99, 0.9925, 0.995, 0.999]

MODEL_LIST = \
    {
        'DummyProbSampling': {
            'model': DummyClassifier(strategy='stratified'),
            'n_search_iter': 1,
            'model_type': 'Dummy Classifier',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'estimator__strategy': ['stratified'],
            }
        },

        'DummyUniformSampling': {
            'model': DummyClassifier(strategy='uniform'),
            'n_search_iter': 1,
            'model_type': 'Dummy Classifier',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'estimator__strategy': ['uniform'],
            }
        },

        'DummyMostFrequent': {
            'model': DummyClassifier(strategy='prior'),
            'n_search_iter': 1,
            'model_type': 'Dummy Classifier',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'estimator__strategy': ['prior'],
            }
        },

        'LogisticRegression': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', LogisticRegression(penalty=None, max_iter=100000))]
            ),
            'n_search_iter': 1,
            'model_type': 'LogisticRegression',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegression_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty=None, max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegression',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [True]
            }
        },

        'LogisticRegression_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', SMOTE()),
                 ('model', LogisticRegression(penalty=None, max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegression',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegression_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty=None,  max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegression',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegression_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', LogisticRegression(penalty=None,  max_iter=100000))]),
            'n_search_iter': 150,
            'model_type': 'LogisticRegression',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        ######################################################################################################
        'LogisticRegressionRidge': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', LogisticRegression(penalty='l2', 
                                              max_iter=100000))]
            ),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegressionRidge_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='l2', max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegressionRidge_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l2', max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegressionRidge_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l2', max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],

            }
        },

        'LogisticRegressionRidge_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', LogisticRegression(penalty='l2', max_iter=100000))]),
            'n_search_iter': 150,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],

            }
        },

        'RakelD_LogisticRegression': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', RakelD(base_classifier=LogisticRegression(penalty='l2', max_iter=100000)))
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'LogisticRegressionRidge',
            'model_subtype': 'RakelD Partitioning of labels',
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__base_classifier': [LogisticRegression(penalty='l2', max_iter=100000)],
            }
        },
        ######################################################################################################

        'LogisticRegressionLasso': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear',
                                              max_iter=1000))]
            ),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionLasso',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegressionLasso_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear',
                                              max_iter=1000))]),
            'model_type': 'LogisticRegressionLasso',
            'model_subtype': 'Random Oversampling',
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegressionLasso_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear',
                                              max_iter=1000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionLasso',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegressionLasso_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear',
                                              max_iter=1000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionLasso',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegressionLasso_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', LogisticRegression(penalty='l1', solver='liblinear',
                                              max_iter=1000))]),
            'n_search_iter': 150,
            'model_type': 'LogisticRegressionLasso',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },


        ######################################################################################################

        'LogisticRegressionElasticNet': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000))]
            ),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionElasticNet',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegressionElasticNet_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000))]),
            'model_type': 'LogisticRegressionElasticNet',
            'model_subtype': 'Random Oversampling',
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegressionElasticNet_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', SMOTE()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionElasticNet',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegressionElasticNet_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000))]),
            'n_search_iter': 60,
            'model_type': 'LogisticRegressionElasticNet',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LogisticRegressionElasticNet_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000))]),
            'n_search_iter': 150,
            'model_type': 'LogisticRegressionElasticNet',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        ######################################################################################################
        'RidgeClassifier': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', RidgeClassifier(max_iter=100000))]
            ),
            'n_search_iter': 60,
            'model_type': 'RidgeClassifier',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'RidgeClassifier_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', RandomOverSampler()),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'model_type': 'RidgeClassifier',
            'model_subtype': 'Random Oversampling',
            'n_search_iter': 60,
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'RidgeClassifier_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', SMOTE()),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'RidgeClassifier',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'RidgeClassifier_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', BorderlineSMOTE()),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'n_search_iter': 60,
            'model_type': 'RidgeClassifier',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'RidgeClassifier_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', RidgeClassifier(max_iter=100000))]),
            'n_search_iter': 150,
            'model_type': 'RidgeClassifier',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        ######################################################################################################
        'SVM_rbf': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', SVC(kernel='rbf', ))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'SVM_rbf_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', RandomOverSampler()),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'SVM_rbf_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', SMOTE()),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'SVM_rbf_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', BorderlineSMOTE()),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 60,
            'model_type': 'SVM',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'SVM_rbf_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', SVC(kernel='rbf'))]),
            'n_search_iter': 150,
            'model_type': 'SVM',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'RakelD_SVM': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', RakelD(base_classifier=SVC()))
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'SVM',
            'model_subtype': 'RakelD Partitioning of labels',
            'hyperparam_space': {
                'preproc__with_std': [True]
            }
        },
        ######################################################################################################
        'LinearSVM': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', LinearSVC(max_iter=50000))]),
            'n_search_iter': 60,
            'model_type': 'LinearSVM',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LinearSVM_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', RandomOverSampler()),
                 ('model', LinearSVC(max_iter=50000))]),
            'n_search_iter': 60,
            'model_type': 'LinearSVM',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LinearSVM_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', SMOTE()),
                 ('model', LinearSVC(max_iter=50000))]),
            'n_search_iter': 60,
            'model_type': 'LinearSVM',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'LinearSVM_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', BorderlineSMOTE()),
                 ('model', LinearSVC(max_iter=50000))]),
            'n_search_iter': 60,
            'model_type': 'LinearSVM',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],   
            }
        },

        'LinearSVM_SVMSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', SVMSMOTE()),
                 ('model', LinearSVC(max_iter=50000))]),
            'n_search_iter': 60,
            'model_type': 'LinearSVM',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],                
            }
        },

        'RakelD_LineaSVM': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', RakelD(base_classifier=LinearSVC(max_iter=50000)))
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'LinearSVM',
            'model_subtype': 'RakelD Partitioning of labels',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        ######################################################################################################
        'kNN': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', KNeighborsClassifier())]
            ),
            'n_search_iter': 60,
            'model_type': 'KNN',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'kNN_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', RandomOverSampler()),
                 ('model', KNeighborsClassifier())]
            ),
            'n_search_iter': 60,
            'model_type': 'KNN',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'kNN_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', SMOTE()),
                 ('model', KNeighborsClassifier())]
            ),
            'n_search_iter': 60,
            'model_type': 'KNN',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'kNN_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', BorderlineSMOTE()),
                 ('model', KNeighborsClassifier())]
            ),
            'n_search_iter': 60,
            'model_type': 'KNN',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'kNN_SVMSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('up', SVMSMOTE()),
                 ('model', KNeighborsClassifier())]
            ),
            'n_search_iter': 60,
            'model_type': 'KNN',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        ######################################################################################################
        'XGBoost_narrow': {
            'model': XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree'),
            'model_type': 'XGBoost',
            'model_subtype': 'No Upsampling',
            'n_search_iter': 60,
            'hyperparam_space': {
            }
        },

        'XGBoost_narrow_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()), ('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist',
                                         booster='gbtree'))]),

            # silent=True,
            'n_search_iter': 60,
            'model_type': 'XGBoost',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'XGBoost_narrow_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()), ('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree'))]),

            # silent=True,
            'n_search_iter': 60,
            'model_type': 'XGBoost',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'XGBoost_narrow_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()), ('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree'))]),
            # silent=True,
            'n_search_iter': 60,
            'model_type': 'XGBoost',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        'XGBoost_narrow_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False, with_std=True)),
                 ('model', XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree'))]),

            # silent=True,
            'n_search_iter': 150,
            'model_type': 'XGBoost',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
                'preproc__with_std': [True],
            }
        },

        ######################################################################################################
        'ComplementNaiveBayes': {
            'model': ComplementNB(),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
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
            }
        },

        'RakelD_ComplementNB': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', RakelD(ComplementNB()))
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'ComplementNB',
            'model_subtype': 'RakelD Partitioning of labels',
            'hyperparam_space': {
            }
        },

        ######################################################################################################

        'NaiveBayes': {
            'model': MultinomialNB(),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
            }
        },

        'NaiveBayes_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', ComplementNB())]),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
            }
        },

        'NaiveBayes_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
            }
        },

        'NaiveBayes_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 60,
            'model_type': 'NaiveBayes',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {
            }
        },

        'NaiveBayes_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 150,
            'model_type': 'NaiveBayes',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {
            }
        },

        ######################################################################################################

        'RandomForest': {
            'model': RandomForestClassifier(),
            'n_search_iter': 60,
            'model_type': 'RandomForest',
            'model_subtype': 'No Upsampling',
            'hyperparam_space': {
            }
        },

        'RandomForest_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 60,
            'model_type': 'RandomForest',
            'model_subtype': 'Random Oversampling',
            'hyperparam_space': {
            }
        },

        'RandomForest_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 60,
            'model_type': 'RandomForest',
            'model_subtype': 'SMOTE',
            'hyperparam_space': {
            }
        },

        'RandomForest_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 60,
            'model_type': 'RandomForest',
            'model_subtype': 'BorderlineSMOTE',
            'hyperparam_space': {

            }
        },

        'RandomForest_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 60,
            'model_type': 'RandomForest',
            'model_subtype': 'SVMSMOTE',
            'hyperparam_space': {

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
                'preproc__with_std': [True],
            }
        },

        'MLkNN': {
            'model': Pipeline([
                ('preproc', StandardScaler(with_mean=False, with_std=True)),
                ('model', MLkNN())
            ]),
            'n_search_iter': 60,
            'wrap_mlb_clf': False,
            'model_type': 'Multilabel k Nearest Neighbours¶',
            'model_subtype': 'Natively Multilabel',
            'hyperparam_space': {
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
            }
        },
    }


if __name__ == '__main__':
    print(MODEL_LIST.keys())
