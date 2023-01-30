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
        'LogisticRegressionRidgeDual': { # BEST
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, class_weight='balanced',
                                              max_iter=100000))]
            ),
            'n_search_iter': 5,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-6, 1.5e-3),
                'model__class_weight': ['balanced']
            }
        },

        'LogisticRegressionRidgeDual_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-6, 1e-2),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not majority', 0.99, 0.9925, 0.995, 0.999],
                'up__shrinkage': loguniform(1e-6, 1)
            }
        },

        'LogisticRegressionRidgeDual_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__C': loguniform(1e-6, 1e-2),
                'up__sampling_strategy': ['minority', 'not majority', 0.99, 0.995, 0.999],
                'up__k_neighbors': randint(7, 25)
            }
        },

        'LogisticRegressionRidgeDual_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', LogisticRegression(penalty='l2', solver='liblinear', dual=True, max_iter=100000))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced'],
                'model__C': loguniform(1e-6, 1e-2),
                'up__sampling_strategy': ['minority', 'not majority', 0.99, 0.995, 0.999],
                'up__k_neighbors': randint(5, 25),
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
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__class_weight': ['balanced', None],
                'model__C': loguniform(1e-6, 1e-2),
                'up__sampling_strategy': ['minority', 'not majority', 0.99, 0.995, 0.999],
                'up__k_neighbors': randint(5, 25),
                'up__m_neighbors': randint(14, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        ######################################################################################################

        'SVM_sigmoid': {# Best, but very similar to the rest
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [False],
                'model__C': loguniform(0.08, 20),
                'model__gamma': loguniform(0.5e-2, 1.5),
            }
        },

        'SVM_sigmoid_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [False],
                'model__C': loguniform(0.1, 10),
                'model__gamma': loguniform(0.5e-2, 1.5),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__shrinkage': loguniform(1e-7, 1e-4)
            }
        },

        'SVM_sigmoid_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [False],
                'model__C': loguniform(0.06, 10),
                'model__gamma': loguniform(1e-2, 1.5),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__k_neighbors': randint(7, 30)
            }
        },

        'SVM_sigmoid_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [False],
                'model__C': uniform(0.07, 10),
                'model__gamma': loguniform(1e-2, 0.5),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__k_neighbors': randint(3, 30),
                'up__m_neighbors': randint(3, 30),
                'up__kind': ['borderline-2']
            }
        },

        'SVM_sigmoid_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [False],
                'model__C': uniform(0.06, 10),
                'model__gamma': loguniform(1e-2, 1.5),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__k_neighbors': randint(3, 30),
                'up__m_neighbors': randint(3, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        },

        ######################################################################################################
        'LinearSVMDual': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000, class_weight='balanced'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': uniform(1e-5, 1e-4)
            }
        },

        'LinearSVMDual_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(2e-5, 1.5e-4),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__shrinkage': loguniform(1e-7, 1e-3)
            }
        },

        'LinearSVMDual_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-6, 1e-4),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__k_neighbors': randint(3, 30)
            }
        },

        'LinearSVMDual_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-5, 1.5e-4),
                'model__class_weight': ['balanced'],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__k_neighbors': randint(3, 30),
                'up__m_neighbors': randint(3, 30),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        ######################################################################################################
        'XGBoost_broad': {
            'model': XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0),
            # silent=True,
            'n_search_iter': 100,
            'hyperparam_space': {
                'n_estimators': randint(300, 400),
                'reg_lambda': loguniform(15, 100),
                'colsample_bytree': uniform(loc=0.65, scale=0.35),
                'max_features': randint(5, 40),
                'learning_rate': loguniform(2e-2, 0.15),
                'max_depth': randint(15, 40),
                'min_child_weight': loguniform(1e-3, 1),
                'max_delta_step': uniform(loc=2, scale=7),
                'scale_pos_weight': uniform(loc=7.5, scale=7.5),
                'subsample': uniform(loc=0.75, scale=0.24),
                'colsample_bynode': uniform(loc=0.7, scale=0.21)
            }
        },

        ######################################################################################################
        'ComplementNaiveBayes': { # Best, but they were all pretty much the same
            'model': ComplementNB(),
            'n_search_iter': 100,
            'hyperparam_space': {
                'alpha': loguniform(1e-6, 0.4),
                'norm': [False]
            }
        },

        'ComplementNaiveBayes_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', ComplementNB())]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-4, 1e-2),
                'model__norm': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__shrinkage': loguniform(1e-6, 500)
            }
        },

        'ComplementNaiveBayes_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-2, 0.2),
                'model__norm': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__k_neighbors': randint(3, 7)
            }
        },

        'ComplementNaiveBayes_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-6, 0.01),
                'model__norm': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__k_neighbors': randint(3, 7),
                'up__m_neighbors': randint(12, 30),
                'up__kind': ['borderline-2']
            }
        },

        'ComplementNaiveBayes_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', ComplementNB())]),
            'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [False],
                'model__alpha': loguniform(1e-6, 0.01),
                'model__norm': [False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__k_neighbors': randint(3, 7),
                'up__m_neighbors': randint(3, 30),
            }
        },

        ######################################################################################################

        'RandomForest': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 80,
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
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
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
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__shrinkage': loguniform(1e-7, 1e-3)
            }
        },

        'RandomForest_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
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
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__k_neighbors': randint(3, 30),
            }
        },

        'RandomForest_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
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
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__k_neighbors': randint(3, 30),
                'up__m_neighbors': randint(3, 30),
                'up__kind': ['borderline-2']
            }
        },

        'RandomForest_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
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
                'up__sampling_strategy': ['minority', 'not minority', 'not majority', 0.99, 0.995, 0.999, 0.9, 0.8],
                'up__k_neighbors': randint(3, 30),
                'up__m_neighbors': randint(3, 30),
                'up__out_step': loguniform(1e-6, 1e-3)
            }
        }

    }

if __name__ == '__main__':
    print(MODEL_LIST.keys())
