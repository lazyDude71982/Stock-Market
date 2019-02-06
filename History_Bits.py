# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 18:10:11 2019

@author: aditya
"""
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

class HB(BaseEstimator,ClassifierMixin):
    def __init__(self,Num_of_group=2,group_size=2,learn_rate=1):
        self.Num_of_group=Num_of_group
        self.group_size=group_size
        self.Groups=[]
        self.group_speed=[]
        self.thresholds=[]
        self.learn_rate=learn_rate
        self.classLimit=(2**(self.Num_of_group))/2
        self.forest=ExtraTreesClassifier(n_estimators=250,random_state=0)
    def _CreateGroups(self,kvpair):
        a=np.array([i for i in range(1,101)])
        z=0
        for i in range(0,self.Num_of_group):
            group=[]
            for j in range(0,self.group_size):
                try:
                    group.append([kvpair[z][0],np.percentile(a,int(kvpair[z][1]*100))])
                except IndexError:
                    print(z)
                z+=1
            self.Groups.append(group)
            self.group_speed.append(np.percentile(a,sum([i[1] for i in group])/len(group)))    
    def _getHBits(self,Xrow):
        HBits=""
        for i in range(0,len(self.Groups)):
            summ=0
            for j in range(0,len(self.Groups[i])):
                summ+=(Xrow[self.Groups[i][j][0]]*self.Groups[i][j][1])
            if summ>self.thresholds[i]:
                HBits=HBits+"1"
            else:
                HBits=HBits+"0"
        return HBits                
    
    def _predictOnce(self,Xrow):
        HBits=self._getHBits(Xrow)
        BitScore=int(HBits,2)
        if BitScore>=self.classLimit:
            return 1
        else:
            return -1
        
    def predict(self,X):
        ypredict=[]
        for i in range(0,X.shape[0]):
            ypredict.append(self._predictOnce(X.iloc[i,:]))
        return np.array(ypredict)    
            
        
    def updateThreshold(self,y_pred,y_exep):
        err=(y_pred-y_exep)/2
        self.thresholds=[self.thresholds[i]+(self.learn_rate*err*self.group_speed[i]) for i in range(0,len(self.thresholds))]
        
    def fit(self,X,y=None):
        self.forest=self.forest.fit(X,y)
        important=self.forest.feature_importances_
        kvpair=list(zip(X.columns,important))
        kvpair=sorted(kvpair,key=lambda i:i[1],reverse=True)
        self._CreateGroups(kvpair)
        y=np.array(y)
        self.thresholds=[0 for i in range(0,len(self.Groups))]
        for i in range(0,X.shape[0]):
            ypredict=self._predictOnce(X.iloc[i,:])
            if ypredict!=y[i]:
                self.updateThreshold(ypredict,y[i])
        return self    
            
            
            
            