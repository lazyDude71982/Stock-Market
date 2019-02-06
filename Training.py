import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score
from History_Bits import HB

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
    Train=load_stock('Training_Info.csv')
    Test=load_stock('Test_Info.csv')
    X_train,y_train=preprocessing(Train)
    X_test,y_test=preprocessing(Test)
    scaler=MinMaxScaler()
    X_scaled=scaler.fit_transform(X_train)
    X_scaled=pd.DataFrame(X_scaled,index=X_train.index,columns=X_train.columns)
    X_test=pd.DataFrame(X_test,index=X_test.index,columns=X_test.columns)
    clf=HB(Num_of_group=6,group_size=2)
    #clf=clf.fit(X_scaled,y_train)
    #y_pred=clf.predict(X_test)
    #print(accuracy_score(y_test,y_pred))
    #print(clf.thresholds)
    scores=cross_val_score(clf,X_scaled,y_train,cv=20)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
main()    