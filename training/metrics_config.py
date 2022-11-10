from sklearn.metrics import make_scorer, f1_score

scoring = {'f1_score' : make_scorer(f1_score, average='weighted', zero)}