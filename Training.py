import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from History_Bits import HB
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def preprocessing(dataset):
    remove_index = ['date', 'open', 'high', 'low', 'close', 'volume', 'Name']
    df = dataset.drop(remove_index, axis=1)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)
    X = df.drop('y', axis=1)
    y = df['y'].copy()
    return X, y


def main():
    Train = pd.read_csv('Training.csv')
    # Test = pd.read_csv('Holdout.csv')
    X_train, y_train = preprocessing(Train)
    # X_test, y_test = preprocessing(Test)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    index = X_train.index
    columns = X_train.columns
    X_scaled = pd.DataFrame(data=X_scaled, index=index, columns=columns)
    clf = HB(Num_of_group=6, group_size=2)
    scores = cross_val_score(clf, X_scaled, y_train, cv=20)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
