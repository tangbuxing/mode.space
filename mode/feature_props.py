# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
import copy
from mode.feature_axis import feature_axis


def feature_props(grd_feature, im=None, which_comps=None, areafac=1, q=None, loc=None):
    x = copy.deepcopy(grd_feature)
    if which_comps is None:
        which_comps = ["centroid", "area", "axis", "intensity"]
    if q is None:
        q = [0.25, 0.9]
    out = {}
    if "centroid" in which_comps:
        if loc is None:
            xd = x["m"].shape
            dim0 = xd[0]
            dim1 = xd[1]
            range0 = np.tile(np.arange(dim0), dim1)
            range1 = (np.arange(dim1)).repeat(dim0)
            loc = np.stack((range0, range1), axis=-1)
        xbool = np.reshape(x["m"], x["m"].size, 'F')
        xcen = np.mean(loc[:, 0][xbool == 1])
        ycen = np.mean(loc[:, 1][xbool == 1])
        out['centroid'] = {"x": xcen, "y": ycen}
    if "area" in which_comps:
        out["area"] = np.sum(x["m"]) * areafac
    if "axis" in which_comps:
        out["axis"] = feature_axis(x, areafac)
    if "intensity" in which_comps:
        ivec = {}
        df = pd.DataFrame(np.array(im[x]), columns=q)
        for i, val in q:
            ivec[val] = df.quantile(val)
        out["intensity"] = ivec
    return out
