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

def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r

def remove_nan(Var):

    x = a[Var]
    new_x = x[np.logical_not(np.isnan(x))]
    new_x = np.array(new_x)

    return new_x


in_dir = "../../NAWEA_23/post_processing/"

a = pd.read_csv(in_dir+'out.csv')

Time_starts = np.arange(50,350,10)

corr_Ux_torq = []
corr_torq_MR = []
for Time_start in Time_starts:

    Time_OF = remove_nan(Var="Time_OF")

    Time_start_idx = np.searchsorted(Time_OF,Time_start)

    RtAeroMxh = remove_nan(Var="RtAeroMxh")
    RtAeroMxh = RtAeroMxh[Time_start_idx:]
    MR = remove_nan(Var="MR")
    MR = MR[Time_start_idx:]

    Ux = remove_nan(Var="Ux_0.0")
    Ux = Ux[Time_start_idx:]

    corr_Ux_torq.append(correlation_coef(Ux,RtAeroMxh))
    corr_torq_MR.append(correlation_coef(RtAeroMxh, MR))


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