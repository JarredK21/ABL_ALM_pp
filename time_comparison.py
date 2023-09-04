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

#modify code

def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r

in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"

offsets = [0.0]

a = Dataset(in_dir+"Dataset.nc")

Time_starts = np.arange(0,1200,10) #mod for end time

#include important correlations
corr_Ux_torq = []
corr_torq_LSSMR = []
corr_IA_MR = []
for Time_start in Time_starts:

    Time_OF = np.array(a.variables["time_OF"])
    Time_sampling = np.array(a.variables["time_sampling"])
    Time_sampling[0] = Time_OF[0]
    Time_sampling[-1] = Time_OF[-1]
    dt = Time_OF[1] - Time_OF[0]

    Time_start_idx = np.searchsorted(Time_OF,Time_start)

    RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:])
    MR = np.array(a.variables["RtAeroMrh"][Time_start_idx:])
    Vx = np.array(a.variables["RtAeroVxh"][Time_start_idx:])

    group = a.groups["0.0"]

    Ux = np.array(group.variables["Ux"])

    f = interpolate.interp1d(Time_sampling,Ux)
    Ux = f(Time_OF)
    Ux = Ux[Time_start_idx:]

    corr_Ux_torq.append(correlation_coef(Ux,RtAeroMxh))
    corr_torq_MR.append(correlation_coef(RtAeroMxh, MR))
    corr_Vx_torq.append(correlation_coef(Vx,RtAeroMxh))

fig = plt.figure()
plt.plot(Time_starts, corr_Ux_torq)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Rotor averaged velocity and Torque")
plt.savefig(in_dir+"start_time_corr1.png")
plt.close()


fig = plt.figure()
plt.plot(Time_starts, corr_torq_MR)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Torque and OOPBM")
plt.savefig(in_dir+"start_time_corr2.png")
plt.close()


fig = plt.figure()
plt.plot(Time_starts, corr_Vx_torq)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between blade averaged velocity and Torque")
plt.savefig(in_dir+"start_time_corr3.png")
plt.close()