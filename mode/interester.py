import numpy as npy
import pandas as pd
import time
import math
import copy
from mode.utils import get_attributes_for_feat, remove_key_from_list
from mode.feature_comps import feature_comps
from mode.feature_axis import feature_axis


def interester(grd_features, properties=None,
               weights=None, b1=None,
               b2=None, show=False, *params):
    x = copy.deepcopy(grd_features)
    if b2 is None:
        b2 = npy.array([100, 90, 0.8, 0.25, 85, 400, 200, 200, 400, 0.25, 200])
    if b1 is None:
        b1 = npy.array([35, 30, 0, 0, 0.5, 35, 20, 40, 120, 1, 40])
    if weights is None:
        weights = npy.array([0.24, 0.12, 0.17, 0.12, 0, 0, 0, 0, 0, 0, 0.35])
    if properties is None:
        properties = npy.array(["cent.dist", "angle.diff", "area.ratio", "int.area",
                                "bdelta", "haus", "ph", "med", "msd", "fom", "minsep"])
    if show:
        begin_tiid = time.time()
    if (x['grd_ob_features'] is None or 'labels_1' not in x['grd_ob_features'].keys()) or \
            (x['grd_fo_features'] is None or 'labels_1' not in x['grd_fo_features'].keys()):
        print("interester: No features in one or both of the fields.  Returning NULL.")
        return None
    zerow = (weights == 0)
    nozerow = ~zerow
    np = npy.sum(nozerow)
    if np == 0:
        print("interester: all weights are zero so that no interest is calculated.  Returning NULL.")
        return None
    if np < weights.size:
        # 过滤零
        # weights = list(filter(lambda we:we!= 0, weights))
        # b1、b2删除weights中索引为零的元素
        properties = properties[nozerow]
        weights = weights[nozerow]
        b1 = b1[nozerow]
        b2 = b2[nozerow]
    a0 = npy.full(np, None)
    a1 = npy.full(np, None)
    # type.ind < - ! is.element(properties, c("area.ratio", "int.area", "fom"))
    type_ind = []
    for i, val in enumerate(properties):
        type_ind.append(val not in ["area.ratio", "int.area", "fom"])
    type_ind = npy.array(type_ind)
    if type_ind.size > 0:
        for i, val in enumerate(type_ind):
            if val:
                a1[i] = -1 / (b2[i] - b1[i])
                a0[i] = 1 - a1[i] * b1[i]
    # type.ind < - is.element(properties, c("area.ratio", "int.area"))
    type_ind = []
    for i, val in enumerate(properties):
        type_ind.append(val in ["area.ratio", "int.area"])
    type_ind = npy.array(type_ind)
    if type_ind.size > 0:
        for i, val in enumerate(type_ind):
            if val:
                a1[i] = 1 / (b2[i] - b1[i])
                a0[i] = 1 - a1[i] * b2[i]

    def ipwlin(x, b1, b2, a0, a1, property, *params):
        dn = []
        for j, value in enumerate(properties):
            dn.append(value in ["cent.dist", "angle.diff", "bdelta", "haus", "ph", "med", "msd", "minsep"])
        up = []
        for j, value in enumerate(properties):
            up.append(value in ["area.ratio", "int.area"])
        fom = (property == "fom")
        res = npy.zeros(len(x))
        x_is_na = is_na(x)
        if any(x_is_na):
            bool_array = npy.array(dn) * npy.array(x_is_na)
            if any(bool_array):
                for j, value in enumerate(bool_array):
                    if value:
                        x[list(x)[j]] = b2[j]
            bool_array = npy.array(up) * npy.array(x_is_na)
            if any(bool_array):
                for j, value in enumerate(bool_array):
                    if value:
                        x[list(x)[j]] = b1[j]
            if any(fom):
                for j, value in enumerate(fom):
                    if value:
                        x[list(x)[j]] = 100
        if any(dn):
            dn2 = []
            dn3 = []
            for j, value in enumerate(list(x)):
                dn2.append(x[value] <= b1[j])
                dn3.append(x[value] > b2[j])
            bool_array = npy.array(dn) * npy.array(dn2)
            if any(bool_array):
                res[npy.array(bool_array)] = 1
            bool_array = npy.array(dn) * npy.array(dn3)
            if any(bool_array):
                res[npy.array(bool_array)] = 0
            bool_array = npy.array(dn) * ~(npy.array(dn2) + npy.array(dn3))
            if any(bool_array):
                look = []
                for j, value in enumerate(bool_array):
                    if value:
                        look.append(a0[j] + a1[j] * x[list(x)[j]])
                look = npy.array(look)
                if any(look < 0):
                    look[look < 0] = 0
                res[bool_array] = look

        if any(up):
            up2 = []
            up3 = []
            for j, value in enumerate(list(x)):
                up2.append(x[value] < b1[j])
                up3.append(x[value] >= b2[j])
            bool_array = npy.array(up) * npy.array(up2)
            if any(bool_array):
                res[npy.array(bool_array)] = 0
            bool_array = npy.array(up) * npy.array(up3)
            if any(bool_array):
                res[npy.array(bool_array)] = 1
            bool_array = npy.array(up) * ~(npy.array(up2) + npy.array(up3))
            if any(bool_array):
                look = []
                for j, value in enumerate(bool_array):
                    if value:
                        look.append(a0[j] + a1[j] * x[list(x)[j]])
                look = npy.array(look)
                if any(look > 1):
                    look[look > 1] = 1
                res[bool_array] = look
        if any(fom):
            look = b1[fom] * math.exp(-0.5 * ((x[fom] - 1) / b2[fom]) ** 4)
            if look < 0:
                look = 0
            elif look > 1:
                look = 1
            res[fom] = look
        # 赋予name，names(res) < - names(x)
        result = {}
        for j, value in enumerate(list(x)):
            result[value] = res[j]
        return result

    def ifun(id, bigX, Xhat, b1, b2, a0, a1, p, *params):
        if show:
            print("Forecast feature: " + id[1] + " vs Observed feature: " + id[0] + "\n")
        XtmpAttributes = get_attributes_for_feat(bigX)
        YtmpAttributes = get_attributes_for_feat(Xhat)
        remove_list = ['Type', 'xrange', 'yrange', 'dim', 'xstep', 'ystep', 'warnings', 'xcol', 'yrow', 'area']
        xkeys = remove_key_from_list(list(bigX.keys()), remove_list)
        ykeys = remove_key_from_list(list(Xhat.keys()), remove_list)
        Xtmp = {"m": bigX[xkeys[id[0]]]}
        Ytmp = {"m": Xhat[ykeys[id[1]]]}
        Xtmp.update(XtmpAttributes)
        Ytmp.update(YtmpAttributes)
        A = feature_comps(grd_fo=Ytmp, grd_ob=Xtmp, which_comps=p)
        nomen = A.keys()
        res = ipwlin(A, b1=b1, b2=b2, a0=a0, a1=a1, property=p)
        if "angle.diff" in p:
            fax = feature_axis(Xtmp)
            fay = feature_axis(Ytmp)
            if fax is None or fay is None:
                con = 0
            else:
                aspX = fax['aspect_ratio']
                aspY = fay['aspect_ratio']
                conX = ((aspX - 1) ** 2 / (aspX ** 2 + 1)) ** 0.3
                conY = ((aspY - 1) ** 2 / (aspY ** 2 + 1)) ** 0.3
                con = math.sqrt(conX * conY)
            res["angle.diff"] = res["angle.diff"] * con
        if "cent.dist" in p and "area.ratio" in p:
            # 子集 all( is.element(c("cent.dist", "area.ratio"), p))
            res["cent.dist"] = res["cent.dist"] * res["area.ratio"]
        return res

    N = len(x['grd_ob_features'].keys()) - 10
    M = len(x['grd_fo_features'].keys()) - 10
    ind = npy.stack((npy.tile(npy.arange(N), M), (npy.arange(M)).repeat(N)), axis=-1)
    if show:
        print(
            "\n\nFinding interest between each pair of " + N + " observed features and " + M + " forecast features.\n\n")
    res1 = npy.apply_along_axis(ifun, axis=1, arr=ind, bigX=x['grd_ob_features'], Xhat=x['grd_fo_features'],
                                b1=b1, b2=b2, a0=a0, a1=a1, p=properties, *params)
    res_array_length_x = len(res1[0])
    res_array = npy.empty([0, res_array_length_x])
    for i in range(len(res1)):
        res_inside = npy.zeros((1, res_array_length_x))
        for j, val in enumerate(list(res1[i])):
            res_inside[0, j] = res1[i][val]
        res_array = npy.append(res_array, res_inside, axis=0)
    res_array_interest = pd.DataFrame(res_array, columns = properties.tolist())
    #out = {'interest': res_array_interest,
           #'total_interest': npy.sum(res_array * npy.tile(weights, (N * M, 1)), axis=1).reshape((N, M))}
    out = {'interest': npy.nan_to_num(res_array_interest),
           'total_interest': npy.array(npy.sum(npy.nan_to_num(res_array * npy.tile(weights, (N * M, 1))), axis=1).reshape((M, N))).transpose()}
    # a = attributes(x)
    # a['names'] = None
    # attributes(out) = a
    # out['total_interest'] = matrix(colSums(res1 * matrix(weights, np, N * M), na.rm = TRUE), N, M)
    if show:
        print(time.time() - begin_tiid)
    return out


def is_na(x):
    res = []
    for i, val in enumerate(list(x)):
        res.append(x[val] is None)
    return res

'''
if __name__ == '__main__':
    data = npy.load("../../deltammResult_PA2.npy", allow_pickle=True).tolist()
    z = interester(data)
    print("hello")
'''