from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import KFold, GridSearchCV


def train_model(classifiers, X_train, y_train):
    best_score = 0
    best_model = None
    for classifier_name, classifier in classifiers.items():
        model = classifier['model']
        params = classifier['params']
        kf = KFold(n_splits=5)
        scoring = make_scorer(f1_score, zero_division=1)
        grid_search = GridSearchCV(model, params, cv=kf, n_jobs=-1, scoring=scoring, verbose=0, return_train_score=True)
        grid_search.fit(X_train, y_train)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_

    print(f"Estimator: {best_model}\nScore: {best_score:.3f}")
    return best_model
