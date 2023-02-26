from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve

DEFAULT_SCORING_FUNCTIONS = ('f1_micro', 'f1_macro', 'accuracy', 'precision_micro',
                             'precision_macro', 'recall_micro', 'recall_macro', 'roc_auc_score')

scoring_functions = {
    'f1_micro': make_scorer(f1_score, average='weighted', zero_division=0),
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
    'accuracy': make_scorer(accuracy_score),
    'precision_micro': make_scorer(precision_score, average='weighted', zero_division=0),
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_micro': make_scorer(recall_score, average='weighted', zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
    'roc_auc_score_micro': make_scorer(roc_auc_score, average='weighted', zero_division=0),
    'roc_auc_score_macro': make_scorer(roc_auc_score, average='macro', zero_division=0)
}

metric_functions = {
    'f1_micro': lambda y_true, y_pred: f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
    'f1_macro': lambda y_true, y_pred: f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0),
    'accuracy': lambda y_true, y_pred: accuracy_score(y_true=y_true, y_pred=y_pred),
    'precision_micro': lambda y_true, y_pred: precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
    'precision_macro': lambda y_true, y_pred: precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0),
    'recall_micro': lambda y_true, y_pred: recall_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
    'recall_macro': lambda y_true, y_pred: recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
}


def compute_multi_label_metrics(y_true, y_pred, prefix: str = None):
    metrics = {}
    for metric_name, metric_fn in metric_functions.items():

        if prefix is None:
            metrics[metric_name] = metric_fn(y_true=y_true, y_pred=y_pred)
        else:
            metrics[f'{prefix}_{metric_name}'] = metric_fn(y_true=y_true, y_pred=y_pred)

    return metrics
