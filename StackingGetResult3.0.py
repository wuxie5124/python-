# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import warnings
import glob
import shutil
import os
warnings.filterwarnings("ignore")    

inbasePath = r'D:\\Stacking_okData\\minmaxTIFFtoAscii\\'
outbasePath = r"D:\\Stacking_okData\\minmaxTIFFtoAscii\\Result\\"

prgbasePath = r"D:\\Stacking_okData\\prj\\"
#这是存放一个.prj后缀文件的文件夹

#filepath = 'D:\Stacking_okData\\mean9.csv'
filepath = r'D:\\Stacking_okData\\XZD2Zmean9Newdata1.csv' 
param4 = np.array(['VegetationCover', 'Distance_from_River', 'Road_density' ,'Particle_size',
 'MeanSD_SameMonthJanuary','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR', 'SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')
param1 = np.array(['Vegetation', 'Particle_s', 'Distance_f', 'Road_densi', 
        'Agricultur', 'Curve_numb','Maximum_snow depth', 'SD', 'lat', 'XJAVHRR_av',
       'landuse2000', 'XJmonthcgrt2','Elevation','Population','GDP','XJmonthcgrt3'
       ,'lon'], dtype='<U32')

pd1 = pd.read_csv(filepath_or_buffer = filepath)

qudiaofeature = list(set(pd1.columns).difference(set(param4)))

w = list(pd1.columns)

qudiaofeature.remove('level')

for x in qudiaofeature:
    w.remove(x)

filelist = w[2:-1]

prjfile = 'Runoff_CV.prj'
#这是一个.prj文件，用于复制，可以是任意的.prj文件，必须放在 prgbasePath 这个位置里面


paramOutFileName = ["real_result3.txt"]

if os.path.exists(outbasePath + paramOutFileName[0][0:-4] +'.prj') == False:
    for i in range(len(paramOutFileName)):
        shutil.copy2(prgbasePath + prjfile, outbasePath)
        os.rename(outbasePath + prjfile, outbasePath + paramOutFileName[i][0:-4] +'.prj')
        
paramindexFileName = ["index.txt"]
paramFile = []
paramOutFile = []
indexFile = []


paramFile = []
paramOutFile = []
indexFile = []

def openFile(x):
    for item in x:
        paramFile.append(open(inbasePath + item +'.txt'))
        
def openOutFile(x):
    for item in x:
        paramOutFile.append(open(outbasePath + item,'w'))     
def openindexFile(x):
    for item in x:
        indexFile.append(open(outbasePath + item))
        
def process(fil):
    b = []
    lineall = fil.readlines()
    for i in range(6,len(lineall)):
        b.append(list(map(float,lineall[i].strip().split())))
    return b    

def writefile(fil):
   for i in range(0,6):
       paramOutFile[0].write(indexFile[0].readline())
   for row in fil:
       rowstr = list(map(str,row))
       for i in range(len(rowstr)-1):
           paramOutFile[0].write(rowstr[i] + " ")
       paramOutFile[0].write(rowstr[len(rowstr)-1] + "\n")
def closeFile(x):
    for item in x:
        item.close()
        
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
    b = list(range(0,num,5))
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
    
    return np.array(XList_train),np.array(XList_test),ylist_train,ylist_test        
        
        
            
openFile(filelist)  
openOutFile(paramOutFileName) 
openindexFile(paramindexFileName)

listall = []
        
for file in paramFile:         
    listall.append(np.array(process(file)))

yOrigin = 34.33627
xOrigin = 73.44696
pixelHeight = 0.0083333333
pixelWidth = 0.0083333333
xl = [] 
yl = []
for i in range(1781):
    y = yOrigin + pixelHeight * (1781-i)
    xll = []
    yll = []
    for j in range(2752):
        x = xOrigin + pixelWidth * j
        xll.append(x)
        yll.append(y)
    xl.append(xll)
    yl.append(yll)
xnp = np.array(xl)
ynp = np.array(yl)

a = listall[0]
for item in listall:
    a[item == -9999] = -9999

nplall = []
nplall.append(xnp[a != -9999])
nplall.append(ynp[a != -9999])

for item in listall:
    nplall.append(item[a != -9999])

allt = []
for item in nplall:
    allt.append(item.tolist())
    
alltT = np.array(allt).T


# =============================================================================
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

#rf_model = RandomForestClassifier(n_estimators = 90,min_samples_split = 100,min_samples_leaf =20,max_depth = 8,max_features= 'sqrt',random_state = 0)
#gdbc_model = GradientBoostingClassifier(n_estimators = 100,max_depth = 3,min_samples_split=100,min_samples_leaf = 40,max_features = 9,subsample = 1,learning_rate = 0.1,random_state = 10)
#knn_model = KNeighborsClassifier(n_neighbors = 49,weights = 'uniform')
#xgb_model = xgb.XGBClassifier(learning_rate =0.1,n_estimators=150,max_depth=4,min_child_weight=3,seed =0,subsample =0.9,colsample_bytree = 0.8,gamma = 0.1,reg_alpha = 0.05,reg_lambda =0.1)
#svc_model = SVC( C= 10, kernel = 'linear')
#rf_model = RandomForestClassifier(n_estimators = 50,max_depth = 1,min_samples_split = 10,min_samples_leaf =10,max_features= 2,random_state = 0)
rf_model = RandomForestClassifier(n_estimators = 21,max_depth = 10,min_samples_split = 4,min_samples_leaf =1,max_features= 5,random_state = 0)
#gdbc_model = GradientBoostingClassifier(n_estimators = 70,max_depth = 3,min_samples_split=60,min_samples_leaf = 10,max_features = 1,subsample = 1,learning_rate = 0.05,random_state = 10)
#knn_model = KNeighborsClassifier(n_neighbors = 6,weights = 'uniform')
#xgb_model = xgb.XGBClassifier(learning_rate =0.7,n_estimators=50,max_depth=1,min_child_weight=5,seed =0,subsample =0.9,colsample_bytree = 0.8,gamma = 0.1,reg_alpha = 0.05,reg_lambda =0.05)
#svc_model = SVC(kernel = 'linear',gamma = 0.001 ,C= 100, probability = True)

#filepath = 'Normalized_primitive8.csv'
#param = np.array(['la','lo','Curve_numb','Distance_f','Elevation','Particle_s', 'Road_densi','SD', 'Vegetation','XJAVHRR_av', 'std_SCD_ch', 'xjmonthcha'], dtype='<U32')

train_x, test_x, train_y, test_y  = loadData(filepath,w)


train_sets = []
test_sets = []

#for clf in [rf_model, gdbc_model,knn_model]:
for clf in [rf_model, gdbc_model, svc_model,knn_model,xgb_model]:
    train_set, test_set = get_stacking(clf, train_x, train_y, alltT)
    train_sets.append(train_set)
    test_sets.append(test_set)
    
leval1_array = np.array(test_sets).T
leval1_pd= pd.DataFrame(leval1_array, columns=['RF','GBDT','SVC','KNN','XGB']) 
leval1_pd.to_csv('firstlevelresult1.csv', index=None)

meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)

#使用决策树作为我们的次级分类器
dt_model = xgb.XGBClassifier()
dt_model.fit(meta_train, train_y)
df_predict = dt_model.predict(meta_test)

######################################################################################
a[a != -9999] = df_predict
writefile(a)

closeFile(paramFile)
closeFile(paramOutFile)



