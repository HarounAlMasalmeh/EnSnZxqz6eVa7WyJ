from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.features.build_features import build_features
from src.models.predict_model import predict_model
from src.models.train_model import train_model

if __name__ == "__main__":
    data_file_path = "../data/raw/term-deposit-marketing-2020.csv"
    X_train, X_test, y_train, y_test = build_features(data_file_path)

    classifiers = {
        'KNeighborsClassifier': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [i for i in range(1, 5)],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [i for i in range(1, 3)]
            }
        },
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'GaussianNB': {
            'model': GaussianNB(),
            'params': {}
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [10, 50],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
    }

    model = train_model(classifiers, X_train, y_train)
    y_pred = predict_model(model, X_test)
    score = f1_score(y_test, y_pred)
    print(f"F1 Score: {score:.3f}")
