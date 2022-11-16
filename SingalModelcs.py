# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import warnings
warnings.filterwarnings("ignore")    


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

def loadData(filePath,param):
    PD = pd.read_csv(filepath_or_buffer = filePath)   
    num = len(PD)
    a = list(range(0,num,1))
    b = list(range(0,num,3))
    for i in b:
        a.remove(i)
    data_train = PD.iloc[a,:]
    data_test  = PD.iloc[b,:]
    data_num1 = len(data_train)
    data_num2 = len(data_test)
    XList_train = []
    XList_test  = []
#    param = PD.columns.values
    for row in range(0, data_num1):
        tmp_list = []
        for low in range(len(param)):
            pn = param[low]
            tmp_list.append(data_train.iloc[row][pn])
        XList_train.append(tmp_list)
    ylist_train =data_train.level.values
    for row in range(0, data_num2):
        tmp_list = []
        for low in range(len(param)):
            pn = param[low]
            tmp_list.append(data_test.iloc[row][pn])
        XList_test.append(tmp_list)
    ylist_test =data_test.level.values    
    
    return np.array(XList_train),np.array(XList_test),ylist_train,ylist_test
    
    
    
    

from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt
#rf_model = RandomForestClassifier(n_estimators = 10,max_depth = None,min_samples_split = 2,min_samples_leaf =1,max_features= 1,random_state = 0)
rf_model = RandomForestClassifier(n_estimators = 21,max_depth = 10,min_samples_split = 4,min_samples_leaf =1,max_features= 5,random_state = 0)
gdbc_model = GradientBoostingClassifier(n_estimators = 70,max_depth = 3,min_samples_split=60,min_samples_leaf = 10,max_features = 1,subsample = 1,learning_rate = 0.05,random_state = 10)
knn_model = KNeighborsClassifier(n_neighbors = 6,weights = 'uniform')
xgb_model = xgb.XGBClassifier(learning_rate =0.7,n_estimators=50,max_depth=1,min_child_weight=5,seed =0,subsample =0.9,colsample_bytree = 0.8,gamma = 0.1,reg_alpha = 0.05,reg_lambda =0.05)
svc_model = SVC(kernel = 'linear',gamma = 0.001 ,C= 100, probability = True)



param2 = np.array(['Vegetation', 'Maximum_snowdepth', 'SDSameMonthStd02',
 'SDSameMonthMean01', 'lat', 'SDSameMonthStd01', 
 'SDSameMonthMean03','SCMchangerate5', 'Road_densi',
 'Distance_f', 'SCDavg', 'Particle_s', 'XJmonthcgrt2', 
 'SCDchangerate', 'Curve_numb', 'lon',
 'XJmonthcgrt3' ], dtype='<U32')
param1 = np.array(['Maximum_snowdepth','landuse2000','Distance_f','Particle_s','SDSameMonthStd02','SDSameMonthMean03','SDSameMonthMean01',
 'SCMchangerate5','Road_densi','Vegetation','lat','XJAVHRR_av',
 'SCDchangerate','SDSameMonthStd01','Elevation','Agricultur','snowclass', 
 'SCSchangerate5','SDSameMonthStd03','SCDavg','XJmonthcgrt3','Curve_numb', 
 'lon','SDSameMonthMean02','XJmonthcgrt2','yearchangerate','Slope',
 'Runoff_CV','Relative_E','Variance_c','XJmonthcgrt1'], dtype='<U32')
#filepath = r'D:\\Stacking_okData\\XZD2Zmean9.csv' 
param3 = np.array(['VegetationCover', 'Distance_from_River', 'Road_density' ,'Particle_size',
 'MeanSD_SameMonthJanuary','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR', 'SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')
param4 = np.array(['VegetationCover', 'Distance_from_River', 'Road_density' ,'Particle_size',
 'MeanSD_SameMonthJanuary','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR', 'SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')
filepath = r'D:\\Stacking_okData\\XZD2Zmean9Newdata1.csv' 
train_x, test_x, train_y, test_y = loadData(filepath, param4)

#使用决策树作为我们的次级分类器
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import r2_score
aa = ['romdamforest','gbdt','svc','knn','xgb']
num =0
for clf in [rf_model, gdbc_model, svc_model,knn_model,xgb_model]:
    clf.fit(train_x, train_y)
    df_predict = clf.predict(test_x)
    r2 = r2_score(test_y,df_predict)
    mse = mean_squared_error(test_y, df_predict)
    rmse = sqrt(mean_squared_error(test_y, df_predict))
    yscore1 = clf.predict_proba(test_x)
    fpr,tpr,threshold = roc_curve(test_y,yscore1[:, 1])
    TP = np.sum(np.logical_and(np.equal(test_y,1),np.equal(df_predict,1)))
    FP = np.sum(np.logical_and(np.equal(test_y,0),np.equal(df_predict,1)))
    TN = np.sum(np.logical_and(np.equal(test_y,0),np.equal(df_predict,0)))
    FN = np.sum(np.logical_and(np.equal(test_y,1),np.equal(df_predict,0)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    error_rate = (FN + FP) / (TP + FP + TN + FN)
    F1_Score = 2*precision*recall/(precision+recall)
    FPR = FP / (TN + FP)
    FNR = FN / (TP + FN)
    roc_auc = auc(fpr,tpr)
    print('{0} precision :{1} recall(TPR):{2} accuracy:{3} error_rate:{4} F1_Score:{5} FPR:{6} FNR:{7}'.format(aa[num],precision,recall,accuracy,error_rate,F1_Score,FPR,FNR)) 
    print('{1}:预测结果{0}'.format(df_predict,aa[num]))
    print('{4}: r2:{0} mse:{1} rmse:{2} auc:{3}'.format(r2,mse,rmse,roc_auc,aa[num]))
    num = num + 1