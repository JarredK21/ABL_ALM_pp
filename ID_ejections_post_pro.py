from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import math
import time
from multiprocessing import Pool
from scipy import interpolate
from matplotlib.patches import Circle

def d_dt_calc(H):

    d_dt = []
    for it in Time_steps[:-1]:
        d_dt.append(np.subtract(H[it+1],H[it])/dt)

    return d_dt


def isInside(x, y):
     
    if ((x - 2560) * (x - 2560) +
        (y - 90) * (y - 90) < 63 * 63):
        return True
    else:
        return False


#def Update_data(it):



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df = Dataset(in_dir+"Threshold_heights_Dataset.nc")

Time = np.array(df.variables["Time"])
dt = Time[1]-Time[0]
Time_steps = np.arange(0,len(Time))
ys = np.array(df.variables["ys"])

#thresholds to output data
# thresholds = [-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.4]
# thresholds.reverse()
thresholds = [-1.4]
for threshold in thresholds:
    group = df.groups["{}".format(abs(threshold))]

    H = np.array(group.variables["Height_ejection"])

    dH_dt = np.array(d_dt_calc(H))

    idx = int(len(ys)/2)

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8),sharex=True)
    ax1.plot(Time[:-1],dH_dt[:,idx],"-k")

    for it in Time_steps[:-1]:
        if 0 < dH_dt[it,idx] < 10:
            ax1.plot(Time[it],dH_dt[it,idx],"ob",markersize=3)
        else:
            ax1.plot(Time[it],dH_dt[it,idx],"or",markersize=3)

    tau = 5
    idx_tau = np.searchsorted(Time,Time[0]+5)
    avg_dH_dt = []
    for it in np.arange(0,len(Time_steps)-idx_tau,1):
        avg_dH_dt.append(np.average(dH_dt[it:it+idx_tau]))
    
    ax2.plot(Time[idx_tau:],avg_dH_dt,"--b")

    ax3.plot(Time,H[:,idx],"-k")
    ax3.axhline(y=27,linestyle="--",color="k")

    ax1.grid()
    ax2.grid()
    ax2.axhline(y=0,linestyle="-",color="k")
    ax3.grid()
    plt.tight_layout()
    plt.show()

    # with Pool() as pool:
    #     for H_it in pool.imap(Update_data,Time_steps):

