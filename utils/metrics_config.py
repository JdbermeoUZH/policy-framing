from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, auc, precision_recall_curve

DEFAULT_SCORING_FUNCTIONS = ('f1_micro', 'f1_macro', 'accuracy', 'precision_micro',
                             'precision_macro', 'recall_micro', 'recall_macro')

scoring_functions = {
    'f1_micro': make_scorer(f1_score, average='weighted', zero_division=0),
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
    'accuracy': make_scorer(accuracy_score),
    'precision_micro': make_scorer(precision_score, average='weighted', zero_division=0),
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_micro': make_scorer(recall_score, average='weighted', zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0)
}

