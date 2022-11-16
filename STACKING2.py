# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def get_stacking(clf, x_train, y_train, x_test, n_folds= 5):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    """
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)

        second_level_train_set[test_index] = clf.predict(x_tst)
        test_nfolds_sets[:,i] = clf.predict(x_test)
    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set

def loadData(filePath):
    PD = pd.read_csv(filepath_or_buffer = filePath)   
    num = len(PD)
    a = list(range(0,num,1))
    b = list(range(0,num,5))
    for i in b:
        a.remove(i)
    data_train = PD.iloc[a,:]
    data_test  = PD.iloc[b,:]
    data_num1 = len(data_train)
    data_num2 = len(data_test)
    XList_train = []
    XList_test  = []
    param = PD.columns.values
    for row in range(0, data_num1):
        tmp_list = []
        for low in range(len(param)-1):
            pn = param[low]
            tmp_list.append(data_train.iloc[row][pn])
        XList_train.append(tmp_list)
    ylist_train =data_train.level.values
    for row in range(0, data_num2):
        tmp_list = []
        for low in range(len(param)-1):
            pn = param[low]
            tmp_list.append(data_test.iloc[row][pn])
        XList_test.append(tmp_list)
    ylist_test =data_test.level.values    
    ylist_lon =data_test.lo.values
    ylist_lat =data_test.la.values
    return np.array(XList_train),np.array(XList_test),ylist_train,ylist_test,ylist_lon,ylist_lat

def calculate(inpath,outpath1,outpath2):    
    rf_model = RandomForestClassifier()
    gdbc_model = GradientBoostingClassifier()
    knn_model = KNeighborsClassifier()
    xgb_model = xgb.XGBClassifier()
    svc_model = SVC()
    
    train_x, test_x, train_y, test_y,lon_y,lat_y  = loadData(inpath + '.csv')
    
    
    train_sets = []
    test_sets = []
    for clf in [rf_model, gdbc_model, svc_model,knn_model,xgb_model]:
        train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
        train_sets.append(train_set)
        test_sets.append(test_set)
        
    leval1_array = np.array(test_sets).T
    leval1_pd= pd.DataFrame(leval1_array, columns=['RF','GBDT','SVM','KNN','XGB']) 
    leval1_pd.to_csv(outpath1 + 'classflier.csv', index=None)
    
    meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)
    
    #使用决策树作为我们的次级分类器
    dt_model = xgb.XGBClassifier()
    dt_model.fit(meta_train, train_y)
    df_predict = dt_model.predict(meta_test)
    
    value = np.array([lon_y,lat_y, test_y, df_predict]).T
    
    leval2_pd= pd.DataFrame(value, columns=['LON','LAT','REAL','XGB']) 
    leval2_pd.to_csv(outpath2 + 'classflier.csv', index=None)
    r2 = r2_score(test_y,df_predict)
#    print(df_predict)
    print(r2)
    
#if __name__=="__main__":
#    b = 2 
#    