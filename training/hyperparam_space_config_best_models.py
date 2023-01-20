from scipy.stats import loguniform, randint, uniform
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

        ######################################################################################################
        'LogisticRegressionRidgeDual': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                              max_iter=100000))]
            ),
            #'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'C': loguniform(1e-4, 1e3)
            }
        },

        'LogisticRegressionRidgeDual_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            #'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-4, 1e3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LogisticRegressionRidgeDual_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            #'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LogisticRegressionRidgeDual_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            #'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-4, 1e3),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        ######################################################################################################
        'SVM_sigmoid': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            #'n_search_iter': 120,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.05, 100),
                'model__gamma': loguniform(1e-1, 100),
            }
        },

        'SVM_sigmoid_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', SVC(kernel='sigmoid'))]),
            #'n_search_iter': 120,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.05, 100),
                'model__gamma': loguniform(1e-1, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'SVM_sigmoid_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            #'n_search_iter': 120,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.05, 100),
                'model__gamma': loguniform(1e-1, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'SVM_sigmoid_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            #'n_search_iter': 120,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.05, 100),
                'model__gamma': loguniform(1e-1, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },


        ######################################################################################################
        'LinearSVMDual': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4, class_weight='balanced'))]),
            #'n_search_iter': 120,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-5, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LinearSVMDual_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            #'n_search_iter': 120,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-5, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },


        'LinearSVMDual_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            #'n_search_iter': 120,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-5, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LinearSVMDual_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            #'n_search_iter': 120,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-5, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        ######################################################################################################
        'RandomForest_v4': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            #'n_search_iter': 80,
            'hyperparam_space': {
                'max_features': randint(2, 100),
                'n_estimators': [50, 100, 200],
                'max_depth': randint(2, 100),
                'ccp_alpha': loguniform(1e-6, 0.1),
                'min_samples_leaf': randint(5, 25),
                'bootstrap': [True, False],
                'max_leaf_nodes': randint(1, 100)
            }
        },

        'RandomForest_ROS': {
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

        ######################################################################################################
        'ComplementNaiveBayes_broadV2': {
            'model': ComplementNB(),
            #'n_search_iter': 100,
            'hyperparam_space': {
                'alpha': loguniform(1e-4, 100),
                'norm': [True, False]
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

        'ComplementNaiveBayes_SMOTE': {
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

        'ComplementNaiveBayes_BorderlineSMOTE': {
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
        }
    }

if __name__ == '__main__':
    print(MODEL_LIST.keys())
