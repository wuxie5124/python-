# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import warnings
warnings.filterwarnings("ignore")  
import glob  
import shutil
import os

inbasePath = r'D:\\Stacking_okData\\minmaxTIFFtoAscii\\'
outbasePath = r"D:\\Stacking_okData\\minmaxTIFFtoAscii\\Result\\"

prgbasePath = r"D:\\Stacking_okData\\prj\\"
#这是存放一个.prj后缀文件的文件夹

filepath = r'D:\\Stacking_okData\\XZD2Zmean9Newdata.csv' 

param1 = np.array(['Maximum_snowdepth','landuse2000','Distance_f','Particle_s','SDSameMonthStd02','SDSameMonthMean03','SDSameMonthMean01',
 'SCMchangerate5','Road_densi','Vegetation','lat','XJAVHRR_av',
 'SCDchangerate','SDSameMonthStd01','Elevation','Agricultur','snowclass', 
 'SCSchangerate5','SDSameMonthStd03','SCDavg','XJmonthcgrt3','Curve_numb', 
 'lon','SDSameMonthMean02','XJmonthcgrt2','yearchangerate','Slope',
 'Runoff_CV','Relative_E','Variance_c','XJmonthcgrt1'], dtype='<U32')
param2 = np.array(['VegetationCover', 'Distance_from_River', 'Road_density' ,'Particle_size',
 'MeanSD_SameMonthJanuary','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR',
 'SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
 'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation',
 'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
 'DEM_Variance_Coefficient', 'Relative_Elevation'], dtype='<U32')
param027 = np.array(['VegetationCover','Distance_from_River', 'Road_density' ,'Particle_size',
 'MeanSD_SameMonthJanuary','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR', 'SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')
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
param4 = np.array(['Distance_from_River', 'Road_density' ,'Particle_size',
 'MeanSD_SameMonthJanuary','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR', 'SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')    
param5 = np.array([
 'MeanSD_SameMonthJanuary','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR', 'SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32') 
param6 = np.array([
 'Latitude',
 'MeanSD_SameMonthMarch',
 'AverageannualSnowdays', 'SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')  
param7 = np.array([
 'Latitude',
 'MeanSD_SameMonthMarch',
 'AverageannualSnowdays',
 'SDmonthchangerate_February',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')
param8 = np.array([
 'Latitude',
 'MeanSD_SameMonthMarch',
 'AverageannualSnowdays',
 'SDmonthchangerate_February',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')
param9 = np.array([ 'Slope','Sdyearchangerate', 'SDmonthchangerate_March', 'Longitude','SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR','MeanSD_SameMonthJanuary'], dtype='<U32')
param10= np.array(['Distance_from_River', 'Road_density' ,'Particle_size','Slope','Sdyearchangerate', 'SDmonthchangerate_March', 'Longitude','SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR','MeanSD_SameMonthJanuary'], dtype='<U32')   
param27 = np.array(['VegetationCover', 'Slope','Sdyearchangerate', 'SDmonthchangerate_March', 'Longitude','SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR','MeanSD_SameMonthJanuary'], dtype='<U32')
param28= np.array(['Distance_from_River', 'Road_density' ,'Particle_size','Slope','Sdyearchangerate', 'SDmonthchangerate_March', 'Longitude','SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR','MeanSD_SameMonthJanuary'], dtype='<U32') 
param26 = np.array(['Distance_from_River', 'Road_density' ,'Particle_size',
 'MeanSD_SameMonthJanuary','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR', 'SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')
param23 = np.array([
 'MeanSD_SameMonthJanuary','Annmasndepth', 'Latitude','Curvenumber',
 'LanduseType', 'MeanSD_SameMonthMarch', 'MeanSD_SameMonthFebruary',
 'SDhStd_SameMonthFebruary', 'AverageannualSnowdays', 'YearlySnowDayS_AVHRR', 'SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')
param13 = np.array([
'SDStd_SameMonthJanuary', 'SCSchangerate5', 'SDStd_SameMonthMarch',
'SCDchangerate', 'SDmonthchangerate_February', 'SCMchangerate5', 'Elevation',
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')
param6 = np.array([
'SDmonthchangerate_March', 'Longitude', 'Sdyearchangerate', 'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')
Param03 = np.array([
'Slope',
'DEM_Variance_Coefficient','Relative_Elevation'], dtype='<U32')
pd1 = pd.read_csv(filepath_or_buffer = filepath)

qudiaofeature = list(set(pd1.columns).difference(set(param27)))

w = list(pd1.columns)

qudiaofeature.remove('level')

for x in qudiaofeature:
    w.remove(x)

filelist = w[2:-1]



prjfile = 'Runoff_CV.prj'
#这是一个.prj文件，用于复制，可以是任意的.prj文件，必须放在 prgbasePath 这个位置里面


paramOutFileName = ["real_resultrf.txt","real_resultgbdt.txt","real_resultxgb.txt","real_resultsvm.txt","real_resultknn.txt"]

if os.path.exists(outbasePath + paramOutFileName[0][0:-4] +'.prj') == False:
    for i in range(len(paramOutFileName)):
        shutil.copy2(prgbasePath + prjfile, outbasePath)
        os.rename(outbasePath + prjfile, outbasePath + paramOutFileName[i][0:-4] +'.prj')

paramindexFileName = ["index.txt"]

#这个索引文件是放在 outbasePath 里面的，可以复制一个放进去

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
def writehead():
    for i in range(0,6):
        ccc = indexFile[0].readline()
        for k in range(len(paramOutFile)):
            paramOutFile[k].write(ccc)
    
def writefile(fil,j):
   for row in fil:
       rowstr = list(map(str,row))
#       print(len(rowstr))
       for i in range(len(rowstr)-1):
           paramOutFile[j].write(rowstr[i] + " ")
       paramOutFile[j].write(rowstr[len(rowstr)-1] + "\n")
def closeFile(x):
    for item in x:
        item.close()
        
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

rf_model = RandomForestClassifier(n_estimators = 21,max_depth = 10,min_samples_split = 4,min_samples_leaf =1,max_features= 5,random_state = 0)
gdbc_model = GradientBoostingClassifier(n_estimators = 70,max_depth = 3,min_samples_split=60,min_samples_leaf = 10,max_features = 1,subsample = 1,learning_rate = 0.05,random_state = 10)
knn_model = KNeighborsClassifier(n_neighbors = 10,weights = 'uniform')
xgb_model = xgb.XGBClassifier(learning_rate =0.7,n_estimators=50,max_depth=1,min_child_weight=5,seed =0,subsample =0.9,colsample_bytree = 0.8,gamma = 0.1,reg_alpha = 0.05,reg_lambda =0.05)
svc_model = SVC(kernel = 'linear',gamma = 0.001 ,C= 10, probability = True)



train_x, test_x, train_y, test_y  = loadData(filepath,w)

num = 0
writehead()
for clf in [rf_model, gdbc_model, svc_model,knn_model,xgb_model]:
    clf.fit(train_x, train_y)
    df_predict = clf.predict(alltT)
    print(len(df_predict))
    print(len(df_predict[df_predict == 0]))
    a[a != -9999] = df_predict
    writefile(a,num)
    num = num + 1
closeFile(paramFile)
closeFile(paramOutFile)
