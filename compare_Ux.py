import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy.stats import pearsonr
import glob 
from scipy import interpolate
from netCDF4 import Dataset
import pandas as pd
import math


def remove_nan(signal):

    
    new_signal = list()
    for element in signal:
        if not math.isnan(element):
            new_signal.append(element)
    return new_signal

dir = "../../../jarred/NAWEA_23/post_processing/plots/"
df = pd.read_csv("../../../jarred/NAWEA_23/post_processing/out.csv")

dq = pd.read_csv("../../../jarred/NAWEA_23/post_processing/out2.csv")

dv = Dataset("../../../jarred/NAWEA_23/post_processing/WTG01.nc")
WTG01 = dv.groups["WTG01"]

time = df["Time_OF"]
time = remove_nan(time)

time2 = dq["Time_sample"]
time2 = remove_nan(time2)
time2[0] = time[0]
time2[-1] = time[-1]

time3 = np.array(WTG01.variables["time"])
time3 = time3 - time3[0]
tstart = 50
tend = 350
tstart_idx = np.searchsorted(time3,tstart)
tend_idx = np.searchsorted(time3,tend)
time3 = time3[tstart_idx:tend_idx]

Var = "Ux_0.0"

signal = df[Var]
signal = remove_nan(signal)

signal2 = dq[Var]
signal2 = remove_nan(signal2)
f = interpolate.interp1d(time2,signal2)
signal2 = f(time)

vel_array = WTG01.variables["vel"]
vel_x_array = np.array(vel_array[:,0:-1,0])
vel_y_array = np.array(vel_array[:,0:-1,1])

velx = []; vely = []
for velx_i,vely_i in zip(vel_x_array,vel_y_array):
    velx.append(np.average(velx_i))
    vely.append(np.average(vely_i))

np.array(velx); np.array(vely)
hvelmag = np.add( np.multiply(velx,np.cos(np.radians(29))) , np.multiply( vely,np.sin(np.radians(29))) )
hvelmag = hvelmag[tstart_idx:tend_idx]

diff = abs(np.subtract(signal,hvelmag))

corr, _ = pearsonr(signal,signal2)

fig = plt.figure(figsize=(14,8))
plt.plot(time,signal,"r")
plt.plot(time,signal2,"b")
plt.plot(time3,hvelmag[:],"g")
plt.xlabel("Time [s]")
plt.ylabel("Ux' [m/s]")
plt.legend(["OpenFAST Ux' sampling method","AMR-Wind Ux' sampling method- corr = {}".format(round(corr,3)),"actuator line sampling method"])
plt.tight_layout()
plt.savefig(dir+"Ux_signals.png")
plt.close(fig)