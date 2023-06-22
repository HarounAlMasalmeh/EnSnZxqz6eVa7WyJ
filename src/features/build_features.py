import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split


def build_features(file_path):
    df = pd.read_csv(file_path, encoding='unicode_escape', sep=',', header=0)
    df.replace({'no': 0, 'yes': 1}, inplace=True)
    df['job'].replace(
        {'management': 1, 'technician': 2, 'entrepreneur': 3, 'blue-collar': 4, 'unknown': 0, 'retired': 5, 'admin': 6,
         'services': 7, 'self-employed': 8, 'unemployed': 9, 'housemaid': 10, 'student': 11}, inplace=True)
    df['month'].replace(
        {'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'oct': 10, 'nov': 11, 'dec': 12, 'jan': 1, 'feb': 2, 'mar': 3,
         'apr': 4}, inplace=True)
    encoded_data = pd.get_dummies(df, columns=['marital', 'contact', 'education'])

    y = encoded_data['y']
    features = encoded_data.drop(['y'], axis=1)
    undersample = NearMiss(version=2, n_neighbors=5)
    x, y = undersample.fit_resample(features, y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=10)

    return X_train, X_test, y_train, y_test
