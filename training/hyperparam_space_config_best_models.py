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
            'n_search_iter': 100,
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
            }
        },

        ######################################################################################################

        'SVM_sigmoid': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.05, 10),
                'model__gamma': loguniform(1e-1, 10),
            }
        },

        'SVM_sigmoid_broad': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid', class_weight='balanced'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-3, 100),
                'model__gamma': loguniform(1e-1, 100),
            }
        },

        'SVM_sigmoid_ROS': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.05, 10),
                'model__gamma': loguniform(1e-1, 10),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'SVM_sigmoid_ROS_v2': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.01, 100),
                'model__gamma': loguniform(1e-3, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'SVM_sigmoid_SMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.05, 10),
                'model__gamma': loguniform(1e-1, 10),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'SVM_sigmoid_SMOTE_v2': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.01, 100),
                'model__gamma': loguniform(1e-3, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': randint(3, 10)
            }
        },

        'SVM_sigmoid_BorderlineSMOTE': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.05, 10),
                'model__gamma': loguniform(1e-1, 10),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'SVM_sigmoid_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.01, 100),
                'model__gamma': loguniform(1e-3, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'SVM_sigmoid_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.05, 10),
                'model__gamma': loguniform(1e-1, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'SVM_sigmoid_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', SVC(kernel='sigmoid'))]),
            'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(0.01, 100),
                'model__gamma': loguniform(1e-3, 100),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################
        'LinearSVMDual_narrow': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000, class_weight='balanced'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-6, 1e-3),
            }
        },

        'LinearSVMDual': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000, class_weight='balanced'))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-7, 1e-1)
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
                'model__C': loguniform(1e-6, 1e-3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'LinearSVMDual_ROS_v2': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', RandomOverSampler()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-7, 1e-1),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
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
                'model__C': loguniform(1e-6, 1e-3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'LinearSVMDual_SMOTE_v2': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', SMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-7, 1e-1),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': randint(3, 10)
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
                'model__C': loguniform(1e-6, 1e-3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LinearSVMDual_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('preproc', StandardScaler(with_mean=False)),
                 ('up', BorderlineSMOTE()),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=50000))]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'preproc__with_std': [True],
                'model__C': loguniform(1e-7, 1e-1),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'LinearSVMDual_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-6, 1e-3),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        'LinearSVMDual_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', LinearSVC(dual=True, penalty='l2', max_iter=5e4))]),
            'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__C': loguniform(1e-7, 1e-1),
                'model__class_weight': ['balanced', None],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },

        ######################################################################################################
        'RandomForest': {
            'model': RandomForestClassifier(class_weight="balanced_subsample"),
            'n_search_iter': 80,
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
            'n_search_iter': 80,
            'hyperparam_space': {
                'model__max_features': ['sqrt', 'log2'],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 25),
                'model__min_samples_leaf': randint(5, 25),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4),
            }
        },


        'RandomForest_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
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



        'RandomForest_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'model__max_features': ['sqrt', 'log2'],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 25),
                'model__min_samples_leaf': randint(5, 25),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3]
            }
        },

        'RandomForest_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
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

        'RandomForest_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'model__max_features': ['sqrt', 'log2'],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 25),
                'model__min_samples_leaf': randint(5, 25),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 7],
                'up__kind': ['borderline-1']
            }
        },

        'RandomForest_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
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

        'RandomForest_SVMSMOTE': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'model__max_features': ['sqrt', 'log2'],
                'model__criterion': ['gini', 'entropy', 'log_loss'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': randint(2, 25),
                'model__min_samples_leaf': randint(5, 25),
                'model__ccp_alpha': loguniform(1e-6, 5e-1),
                'model__bootstrap': [True, False],
                'model__max_leaf_nodes': randint(1, 100),
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
            }
        },

        'RandomForest_SVMSMOTE_v2': {
            'model': Pipeline(
                [('up', SVMSMOTE()),
                 ('model', RandomForestClassifier())]),
            'n_search_iter': 80,
            'hyperparam_space': {
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

        'XGBoost_broad': {
            'model': XGBClassifier(verbosity=0, tree_method='hist', booster='gbtree', gamma=0),
            # silent=True,
            'n_search_iter': 100,
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

        ######################################################################################################
        'ComplementNaiveBayes_narrow': {
            'model': ComplementNB(),
            'n_search_iter': 100,
            'hyperparam_space': {
                'alpha': loguniform(1e-2, 1),
                'norm': [True, False]
            }
        },
        
        'ComplementNaiveBayes_broad': {
            'model': ComplementNB(),
            'n_search_iter': 100,
            'hyperparam_space': {
                'alpha': loguniform(1e-4, 100),
                'norm': [True, False]
            }
        },


        'ComplementNaiveBayes_ROS': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', ComplementNB())]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-2, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'ComplementNaiveBayes_ROS_v2': {
            'model': Pipeline(
                [('up', RandomOverSampler()),
                 ('model', ComplementNB())]),
            'n_search_iter': 100,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-4, 100),
                'model__norm': [True, False],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__shrinkage': loguniform(1e-4, 1e4)
            }
        },

        'ComplementNaiveBayes_SMOTE': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-2, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10)
            }
        },

        'ComplementNaiveBayes_SMOTE_v2': {
            'model': Pipeline(
                [('up', SMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-4, 100),
                'model__norm': [True, False],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': randint(3, 10)
            }
        },

        'ComplementNaiveBayes_BorderlineSMOTE': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-2, 1),
                'model__norm': [True, False],
                'up__sampling_strategy': ['minority', 'not minority', 'not majority'],
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'ComplementNaiveBayes_BorderlineSMOTE_v2': {
            'model': Pipeline(
                [('up', BorderlineSMOTE()),
                 ('model', ComplementNB())]),
            'n_search_iter': 80,
            'hyperparam_space': {
                'model__alpha': loguniform(1e-4, 100),
                'model__norm': [True, False],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': randint(3, 10),
                'up__m_neighbors': randint(3, 20),
                'up__kind': ['borderline-1', 'borderline-2']
            }
        },

        'ComplementNaiveBayes_SVMSMOTE_v1': {
            'model': Pipeline(
                [('up', SVMSMOTE()), ('preproc', StandardScaler(with_mean=False)),
                 ('model', ComplementNB())]),
            'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__alpha': loguniform(1e-2, 1),
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
            'n_search_iter': 150,
            'hyperparam_space': {
                'preproc__with_std': [True, False],
                'model__alpha': loguniform(1e-4, 100),
                'model__norm': [True, False],
                'up__sampling_strategy': uniform(0.7, 0.3),
                'up__k_neighbors': [2, 3],
                'up__m_neighbors': [2, 3],
                'up__out_step': loguniform(1e-6, 1)
            }
        },
    }

if __name__ == '__main__':
    print(MODEL_LIST.keys())
