# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 16:57:50 2022

@author: A
"""

import numpy as np
import pandas as pd

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