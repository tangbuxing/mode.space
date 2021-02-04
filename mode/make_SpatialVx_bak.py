# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:40:04 2020

@author: 1
"""

#======================  make.SpatialVx()  =======================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import math

def isAllDigital(array):      #这个函数是用来判断list是否为一维纯数值，如果是二维，返回false
    ret = True
    for v in array:
    	ret = ret and isinstance(v,(int,float,np.int,np.int32))
    	if not ret:
    		return False
    return ret


def makeSpatialVx(X, Xhat, loc, thresholds = [None, None], projection = False, 
                  subset = None, timevals = None,reggrid = True, Map = False, 
                  locbyrow = False, fieldtype = "", units = "", dataname = "", 
                  obsname = "X", modelname = "Xhat", q = (0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95), 
                  qs = None ):
    #参数：X, Xhat, threshold, loc, projection, subset, timevals,reggrid, map, locbyrow, fieldtype, units, dataname, obsname, modelname, q, qs
        
    #if (X.shape == None):
    if (type(X) == list):    #判断X是否为列表，X实际为二维矩阵
        nobs = X.shape[0] * X.shape[1]
        xdim = None
    else:
        nobs = 1
        xdim = X.shape
        
    #if (Xhat.shape == None):
    if (type(Xhat) == list):    #判断X是否为列表
        nforecast = Xhat.shape[0] * Xhat.shape[1]
        ydim = None
    else:
        nforecast = 1
        ydim = Xhat.shape
        
    
    if (xdim != ydim):
        try:
            sys.exit(0)
        except:
            print("make.SpatialVx: dim of X{0}  must be the same as dim of Xhat{1} ".format(X.shape,Xhat.shape))
    
    out = {"X": X, "Xhat": Xhat}    #将矩阵存为列表
    
   
    if (all(thresholds) == None):
        if (nobs > 1):    #猜测是输入的观测数据大于一个时次，对每个时次的数据分别求分位数
            othresh = np.zeros((9,1))    #定义othresh为空数组，长度与q一致，9
            for i in range(1, nobs):
                #X_1 = np.array(X)
                #X_flat = X_1.flatten()    #dataframe没有flatten属性，先转array
                quantile_X = np.quantile(X, q)  #计算X的分位数
                othresh = np.hstack([quantile_X, othresh])
        else:
            othresh = np.quantile(X, q)    #数据大小为（9，1）是写死的，因为所要求的的分位数的个数是确定的
        
        if (nforecast > 1):    #猜测是输入的观测数据大于一个时次，对每个时次的数据分别求分位数
            fthresh = np.zeros()    #定义fthresh为空数组，长度未知
            for i in range(1, nforecast):
                quantile_Xhat = np.quantile(Xhat, q)  #计算Xhat的q分位数
                fthresh = np.hstack([fthresh, quantile_Xhat])  #原函数里面直接生成的矩阵，没有赋值到任何变量
        else:
            fthresh = np.quantile(Xhat, q)    #数据大小为（9，1）是写死的，因为所要求的的分位数的个数是确定的
        qs = str(q)
        #X = othresh
        #Xhat = fthresh
        thresholds = [othresh, fthresh]
        
    elif (type(thresholds) == list):    #判断thresholds是否为一个list
        threshnames = ["X","Xhat"]    #获取列表的名称
        if (("X"not in threshnames) or ("Xhat" not in threshnames)):
            try:
                sys.exit(0)
            except:
                print("make.SpatialVx: invalid thresholds argument.  List must have components named X and Xhat.")
       
        odim = np.array([len(thresholds), np.ndim(thresholds)])    #thresholds 第一项(X)的数组长度赋值给odim
        fdim = np.array([len(thresholds), np.ndim(thresholds)])    #thresholds 第二项(Xhat)的数组长度赋值给fdim
        
    
        if(odim[0] != fdim[0]):
            try:
                sys.exit(0)
            except:
                print("make.SpatialVx: invalid thresholds argument.  X must have same number of thresholds (rows) as Xhat.")
        nth = odim[0]
        
        if(any(odim) == None):
            thresholds[0] = np.array(thresholds[0]).reshape(len(thresholds[0]),nobs)
        elif(odim[1] == 1 and nobs > 1):
            thresholds[0] = np.array(thresholds[0]).reshape(len(thresholds[0]),nobs)
        elif (odim[1] > 1 and odim[1] != nobs):
            try:
                sys.exit(0)
            except:
                print("make.SpatialVx: invalid thresholds argument.")
        threshold = 'threshold'
        qs = []
        for i in range(nth):
            qs.append(threshold +' '+ str(i))
       
        
    elif(type(thresholds) == list):    #判断thresholds是否为vector,R里面检查是否最多只有一个属性：name。即查看其属性，如果没有属性或者只有一个name属性，才返回TRUE

        qs = str(thresholds)
        nth = len(thresholds)
        thresholds = [np.array(thresholds).reshape(nth,nobs),np.array(thresholds).reshape(nth,nforecast)]
        #else stop("make.SpatialVx: invalid thresholds argument.  Must be a vector or list.")
    else:
        try:
            sys.exit(0)
        except:
            print("make.SpatialVx: invalid thresholds argument.  Must be a vector or list.")
        
    if (loc.any() == None):    #判断loc是否为空,如果为空的话，生成坐标信息
        rep00_1 = np.arange(1, xdim[0], 1)
        rep00 = np.tile(rep00_1, xdim[1])
        rep0_1 = np.arange(1, xdim[1], 1)
        rep0 = rep0_1.repeat(xdim[0], axis = 0)    #按行进行元素重复
        loc = np.vstack((rep00, rep0))
    if (timevals == None):
        if (len(xdim) == 3):
            timevals = np.arnge(1, xdim[2])
        else:
            timevals = 1
    if (dataname != None):
        msg = dataname
    else :
        msg = ""
    if (fieldtype != "" and units != ""):
        msg = msg + '\n' + fieldtype +' ' + '(' + units + ')'
    elif (fieldtype != ""):
        msg = msg + '\n' + fieldtype
    elif (units != ""):
        msg = msg + ' ' + '(' + units + ')'
        
       
    out = {"X": X, "Xhat": Xhat, "class":"SpatialVx", "xdim":xdim, "time":timevals,\
               "thresholds":thresholds, "loc":loc, "locbyrow":locbyrow, \
               "subset":subset, "dataname":dataname, "obsname":obsname, \
               "modelname":modelname, "nobs":nobs, "nforecast":nforecast, \
               "fieldtype":fieldtype , "units":units , "projection":projection , \
               "reggrid":reggrid , "Map":Map , "qs":qs, "msg":msg}
    return out

'''

hold = makeSpatialVx(X = pd.read_csv(r'F:\Work\MODE\tra_test\QPEF\QPE\0802.csv'),
                     Xhat = pd.read_csv(r'F:\Work\MODE\tra_test\QPEF\QPF\0801.csv'),
                     thresholds = [0.01, 20.01], loc = pd.read_csv("F:\\Work\\MODE\\tra_test\\FeatureFinder\\ICPg240Locs.csv"),
                     projection = True, subset = None, timevals = None, reggrid = True, 
                     Map = True, locbyrow = True, fieldtype = "Precipitation", units = ("mm/h"), 
                     dataname = "ICP Perturbed Cases", obsname = "pert000" ,modelname = "pert004",
                     q = (0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95) ,qs = None)            
#loc参数最好不要为空,否则需要多一些参数重新生成
hold = makeSpatialVx(X = pd.read_csv(r'F:\Work\MODE\tra_test\QPEF\QPE\0802.csv'),
                     Xhat = pd.read_csv(r'F:\Work\MODE\tra_test\QPEF\QPF\0801.csv'), 
                     loc = pd.read_csv("F:\\Work\\MODE\\tra_test\\FeatureFinder\\ICPg240Locs.csv"),
                     fieldtype = "Precipitation")
'''           


















