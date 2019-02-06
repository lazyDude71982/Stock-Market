import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve,ShuffleSplit,cross_val_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from History_Bits import HB

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

def load_stock(companyname=""):
    csv_path = os.path.join('',companyname)
    return pd.read_csv(csv_path)

def preprocessing(dataset):
    df=dataset.drop(['date','open','high','low','close','volume','Name'],axis=1)
    df=df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)
    X=df.drop('y',axis=1)
    y=df['y'].copy()
    return X,y
	
def main():
    Train=load_stock('Training_aapl.csv')
    Test=load_stock('Test_aapl.csv')
    X_train,y_train=preprocessing(Train)
    X_test,y_test=preprocessing(Test)
    scaler=MinMaxScaler()
    X_scaled=scaler.fit_transform(X_train)
    X_scaled=pd.DataFrame(X_scaled,index=X_train.index,columns=X_train.columns)
    X_test=pd.DataFrame(X_test,index=X_test.index,columns=X_test.columns)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    clf=HB(Num_of_group=6,group_size=2)
    plot_learning_curve(clf,"Kya karu mai", X_scaled, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    #print(clf.thresholds)
    #scores=cross_val_score(bag_clf,X_scaled,y_train,cv=20)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    plt.show()
main()    