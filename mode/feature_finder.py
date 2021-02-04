# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:58:07 2020

@author: 1
"""

#============================  FeatureFinder()  ==============================
#Function:Identify spatial features within a verification set using a threshold-based method.

import matplotlib.pyplot as plt
from numpy.fft import ifftshift
from skimage import measure,color
import numpy as np
import pandas as pd
import cv2 as cv
import xarray as xr
import sys
import copy
import data_pre
import time
#sys.path.append(r'F:\Work\MODE\Submit')    #导入的函数路径
#from MODE import make_spatialVx   #导入makeSpatialVx函数


#start = time.clock()    #开始时间

#连通域标记
def labelsConnection(data):
    labels=measure.label(data,connectivity=2)  #connectivity表示连接的模式，1代表4连通，2代表8连通
    dst=color.label2rgb(labels)  #根据不同的标记显示不同的颜色
    #print('regions number:',labels.max()+1)  #显示连通区域块数(从0开始标记)
    return dst

#每个连通域内格点数统计,并且判断格点数量是否在阈值范围内
def propCounts(data_2, minsize, maxsize):
    labels = measure.label(data_2,connectivity=2)
    properties = measure.regionprops(labels)
    #print(properties)
    valid_label = []
    nums = []
    area = []
    numsID = []
    current_bw = np.array(())
    
    labelsfeature = {}
    for prop in properties:
        #print(prop.area)
        nums.append(prop.area)    #统计连通域内的格点数
        numsID = list(map(lambda x :any(x > [minsize]) and any(x < [maxsize]), nums))    #比较格点数是否在阈值范围内

    for i in range(len(numsID)):
        if numsID[i]:
            valid_label_0 = i
            area_0 = nums[i]
            area.append(area_0)
            #print(numsID[i], valid_label_0)
            valid_label.append(valid_label_0)
            current_bw = np.in1d(labels, list(valid_label)).reshape(labels.shape)
            #labels = prop._label_image    #读取的prop._label_image数据是已经进行过旋转的结果
    for i in range(len(valid_label)):
        n = valid_label[i]
        #print(n)
        f = np.where(labels, labels == n + 1, 0)    
        f = {'labels_{}'.format(i + 1):f}
        labelsfeature.update(f)
        #添加属性
        xrange = [0.5, np.shape(data_2)[1] + 0.5]
        yrange = [0.5, np.shape(data_2)[0] + 0.5]
        dim = np.shape(data_2)
        xcol = np.arange(1, np.shape(data_2)[1] + 1, 1)
        yrow = np.arange(1, np.shape(data_2)[0] + 1, 1)
        warnings = ["Row index corresponds to increasing y coordinate; column to increasing x","Transpose matrices to get the standard presentation in R",\
                    "Example: image(result$xcol,result$yrow,t(result$d))"]
        labelsfeature.update({'Type':'mask', 'xrange': xrange, 'yrange': yrange, \
                              'dim': dim, 'xstep': 1, 'ystep': 1, 'area': area, 
                              'warnings': warnings, 'xcol':xcol, 'yrow':yrow})
                
        
    return nums, numsID, current_bw, properties, labelsfeature

#R里面新定义的Nfun函数，求和函数,实际函数功能是统计连通域内面积大小,python用prop.area代替
def newFun(Obj):
    #axis为:1：计算每一行的和，axis为0：计算每一列的和,keepdims为保持原有维度
    return np.sum(np.sum(Obj, axis=0, keepdims=True))
        
def feature_finder(grd_ob, grd_fo, smooth, threshold, minsize, maxsize = float("Inf"), 
                   do_smooth = True ):
    
    #读观测数据
    #dataset_ob = xr.open_dataset(filename_ob)  #通过xarray程序库读取nc文件中的所有内容
    #X = np.squeeze(grd_ob.variables['data0'])
    
    #读预报数据
    #dataset_fcst = xr.open_dataset(filename_fo)  #通过xarray程序库读取nc文件中的所有内容
    #Xhat = np.squeeze(grd_fo.variables['data0'])
    
    X = np.squeeze(np.array(grd_ob))
    Xhat = np.squeeze(np.array(grd_fo))
    
    #读经纬度，并形成R里面的格式
    lon = grd_ob['lon']
    lat = grd_ob['lat']    
    X_lon, Y_lat = np.meshgrid(lon, lat)
    loc1 = X_lon.reshape(X_lon.size, order = 'F')
    loc2 = Y_lat.reshape(Y_lat.size, order = 'F')
    loc = np.vstack((loc1, loc2)).T    
    
    Object = {"grd_ob": X, "grd_fo": Xhat, "loc":loc}    #hold是make_saptialVx的计算结果
    thresh = threshold
    smoothpar = smooth
    smoothfun = "disk2dsmooth"
    smoothfunargs = None
    idfun = "disjointer"
    zerodown = False
    timepoint = 1
    obs = 1
    model = 1
    fac = 1
    
    if type(minsize) == list :
        minsize = np.array(minsize)
    if type(maxsize) == list:
        maxsize = np.array(maxsize)
    if type(smooth) == list:
        smooth = np.array(smooth)
    if type(thresh) == list:
       thresh = np.array(thresh) 
    
    #theCall <- match.call()    #调用match函数

    if (np.size(minsize) == 1):
        minsize = np.tile(minsize, 2)        #输入观测和预报数据，因而将最小值和最大值展开为两个数
    if (np.size(maxsize) == 1):    #error:object of type 'float' has no len()
        maxsize = np.tile(maxsize, 2)
        
    if (len(minsize) != 2):
        try:
            sys.exit(0)
        except:
            print("FeatureFinder: invalid min.size argument.  Must have length one or two.")
    if (len(maxsize) != 2):
        try:
            sys.exit(0)
        except:
            print("FeatureFinder: invalid max.size argument.  Must have length one or two.")
    if (any(minsize) < 1):    #a为list
        try:
            sys.exit(0)
        except:
            print("FeatureFinder: invalid min.size argument.  Must be >= 1.")
    if (any(maxsize) < any(minsize)):
        try:
            sys.exit(0)
        except:
            print("FeatureFinder: invalid max.size argument.  Must be >= min.size argument.")
    a = Object.copy() 
    dat = {'X':Object['grd_ob'], 'Xhat':Object['grd_fo']}    #引用hold里的X,Xhat要素并赋值
    X = dat['X']
    Y = dat['Xhat']
    xdim = X.shape
    
    
    if (do_smooth):
        if (np.size(smoothpar) == 1):
            smoothpar = np.tile(smoothpar, 2)    #观测场和预报场的平滑参数一致
        elif (len(smoothpar) > 2):
            try:
                sys.exit(0)
            except:
                print("FeatureFinder: invalid smoothpar argument.  Must have length one or two.")
        
         #调用disk2dsmooth中的kernel2dsmooth卷积平滑，python里面目前用2D卷积平滑替代
        kernel_X = np.ones((smoothpar[0], smoothpar[0]), np.float32)/5    #X的卷积核
        kernel_Y =  np.ones((smoothpar[1], smoothpar[1]), np.float32)/5    #Y的卷积核
        Xsm = cv.filter2D(np.rot90(X, 4), -1, kernel_X)    #对X做2D卷积平滑,旋转4次=没有旋转，不做旋转会报错（opencv版本问题）
        Ysm = cv.filter2D(np.rot90(Y, 4), -1, kernel_Y)    #对Y做2D卷积平滑,旋转4次=没有旋转，不做旋转会报错（opencv版本问题）
        if (zerodown):
             Xsm = np.where(Xsm > 0, Xsm, 0)    #Xsm中大于0的值被0代替
             Ysm = np.where(Ysm > 0, Ysm, 0)    #Ysm中大于0的值被0代替             
    else:
        Xsm = X
        Ysm = Y
    
    if (np.size(thresh) == 1):
        thresh = np.tile(thresh, 2)
    thresh = thresh * fac
    #二值化，首先生成0矩阵，然后将大于阈值的部分赋值为1
    #但是python里面在进行连通域分析的时候已经进行过二值化，不必单独进行二值化
    #sIx = np.zeros((xdim[0], xdim[1]))
    #sIy = np.zeros((xdim[0], xdim[1]))
    #sIx = np.where(Xsm < thresh[0], Xsm, 1)
    #sIy = np.where(Ysm < thresh[1], Xsm, 1)
    
    #连通域分析
    #连通域分析的时候，分析的是经过模糊处理的图像
    #R里面的阈值为5,2D卷积平滑时增强图像，阈值设为310才能结果对应
    Xfeats = labelsConnection(Xsm > thresh[0])    
    Yfeats = labelsConnection(Ysm > thresh[1])
    if (len(Xfeats) == 0):
        Xfeats = None
    if (len(Yfeats) == 0):
        Yfeats = None
    #如果对连通域的面积大小做限制，则需要执行下面的判断
    if (any(minsize > 1) or any(maxsize < X.size)):
        #统计每个连通域内的格点数
        if (np.all(Xfeats) != None):
            Xnums = propCounts(np.rot90(Xsm > thresh[0], 4), minsize[0], maxsize[0])[0]
            XnumsID = propCounts(np.rot90(Xsm > thresh[0], 4), minsize[0], maxsize[0])[1]
            Xfeats = propCounts(np.rot90(Xsm > thresh[0], 4), minsize[0], maxsize[0])[2]
            Xprop = propCounts(np.rot90(Xsm > thresh[0], 4), minsize[0], maxsize[0])[3]

        if (np.all(Yfeats) != None):
            Ynums = propCounts(np.rot90(Ysm > thresh[1], 4), minsize[1], maxsize[1])[0]
            YnumsID = propCounts(np.rot90(Ysm > thresh[1], 4), minsize[1], maxsize[1])[1]
            Yfeats = propCounts(np.rot90(Ysm > thresh[1], 4), minsize[1], maxsize[1])[2]
            Yprop = propCounts(np.rot90(Ysm > thresh[1], 4), minsize[1], maxsize[1])[3]

    
    #Xlab = np.zeros(xdim[0], xdim[1])
    #Ylab = np.zeros(xdim[0], xdim[1])
    
    if (np.all(Xfeats) != None):
        Xlab = Xfeats            
    else:
        Xfeats = None
        
    if (np.all(Yfeats) != None):
        Ylab = Yfeats            
    else:
        Yfeats = None    
    
    #返回的out列表里面包括:观测数据（X）,预报数据（Xhat），连通域分析后单个区域（X.feats,Yfeats）并包括了很多的属性信息
    #X.feats,Yfeats包括了很多的属性信息：x,y的范围xrange,yrange，维度dim,步长xstep,ystep,行列信息等；标记后的连通域（Xlab,Ylab）；
    #识别函数、标记函数名称：Convolution Threshold。
    Xprop = propCounts(Xsm > thresh[0], minsize[0], maxsize[0])[3]
    Yprop = propCounts(Ysm > thresh[1], minsize[1], maxsize[1])[3]
    Xlabelsfeature = propCounts(Xsm > thresh[0], minsize[0], maxsize[0])[4]
    Ylabelsfeature = propCounts(Ysm > thresh[1], minsize[1], maxsize[1])[4]
    
    '''
    out = {'X':Object['X'], 'Xhat':Object['Xhat'], 'loc':Object['loc'], "Xlabeled":Xfeats, "Ylabeled":Yfeats,
           "identifier_function" : "convthresh", "identifier_label" : "Convolution Threshold", 
           "attr_timepoint" : "time_point", "attr_model": "model", "attr_call": "theCall",
           "Xprop":Xprop, "Yprop":Yprop, "Xlabelsfeature":Xlabelsfeature, "Ylabelsfeature":Ylabelsfeature}
    '''
    out = {'grd_ob':Object['grd_ob'], 'grd_fo':Object['grd_fo'], 'loc':Object['loc'], 
           "grd_ob_labeled":Xfeats, "grd_fo_labeled":Yfeats,
           "identifier_label" : "Convolution Threshold", 
           "grd_ob_prop":Xprop, "grd_fo_prop":Yprop, 
           "grd_ob_features":Xlabelsfeature, "grd_fo_features":Ylabelsfeature}    
    
    Xlabels = data_pre.pick_labels(copy.deepcopy(Xlabelsfeature))
    xtmp = Xlabels.copy()
    Xlabeled = data_pre.relabeled(xtmp)  
    Ylabels = data_pre.pick_labels(copy.deepcopy(Ylabelsfeature))    
    ytmp = Ylabels.copy()
    Ylabeled = data_pre.relabeled(ytmp)  
    out.update({ "grd_ob_labeled":Xlabeled, "grd_fo_labeled":Ylabeled})
    #每个格点的长、宽
    a = np.unique(out['loc'][:, 0])[1]-np.unique(out['loc'][:, 0])[0]    #第一列为经度间隔
    b = np.unique(out['loc'][:, 1])[1]-np.unique(out['loc'][:, 1])[0]    #第二列为维度间隔
    S = a * b
    ob_area = (np.array(out['grd_ob_features']['area'])*S).tolist()
    out['grd_ob_features']['area'] = ob_area
    
    fo_area = (np.array(out['grd_fo_features']['area'])*S).tolist()
    out['grd_fo_features']['area'] = fo_area    
    
    return out
        
        
    
    
    
    
#=============================  Example  ===================================
'''
#make.SpatialVx参数
hold = make_SpatialVx_PA1.makeSpatialVx(X = pd.read_csv(r'F:\Work\MODE\tra_test\QPEF\QPE\0802.csv'),
                     Xhat = pd.read_csv(r'F:\Work\MODE\tra_test\QPEF\QPF\0801.csv'),
                     thresholds = [0.01, 20.01], loc = pd.read_csv("F:\\Work\\MODE\\tra_test\\FeatureFinder\\ICPg240Locs.csv"),
                     projection = True, subset = None, timevals = None, reggrid = True, 
                     Map = True, locbyrow = True, fieldtype = "Precipitation", units = ("mm/h"), 
                     dataname = "ICP Perturbed Cases", obsname = "pert000" ,modelname = "pert004",
                     q = (0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95) ,qs = None)
'''                    
#FeatureFinder参数
'''
Object = hold.copy()    #make.SpatialVx生成的结果
smoothfun = "disk2dsmooth"
dosmooth = True
smoothpar = 17    #卷积核的大小
smoothfunargs = None
thresh = 310    #R里面的阈值为5,2D卷积平滑增强图像，阈值设为310才能结果对应
idfun = "disjointer"
minsize = np.array([10, 5])    #判断连通域大小的下限，超过两个值的话用数组
maxsize = float("Inf")    #判断连通域大小的上限，默认无穷大,如果是数值的话，可以直接赋值，超过两个值的话用数组
fac = 1
zerodown = False
timepoint = 1
obs = 1
model = 1

look = featureFinder(Object, smoothfun, dosmooth, smoothpar, smoothfunargs,\
                     thresh, idfun, minsize, maxsize, fac, zerodown, timepoint, obs, model)
'''

'''
#thresh是阈值，翻译后是扩大了约60倍的数据
look_FeatureFinder = featureFinder(Object = hold.copy(), smoothfun = "disk2dsmooth", 
                     dosmooth = True, smoothpar = 17, smoothfunargs = None,
                     thresh = 1800, idfun = "disjointer", minsize = np.array([1]),
                     maxsize = float("Inf"), fac = 1, zerodown = False, timepoint = 1,
                     obs = 1, model = 1)

look = featureFinder(Object = hold.copy(), smoothpar = 17, thresh = 25)
'''
#计算程序运行时间
#end = time.clock()
#print('Running time: %s s'%(end-start))


















