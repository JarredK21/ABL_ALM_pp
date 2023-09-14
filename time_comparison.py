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
import pandas
import math

#modify code

def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r

in_dir = "../../NREL_5MW_MCBL_R_CRPM_2/post_processing/"

offsets = [0.0]

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]
Time_end = Time_sampling[-1]
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_starts = np.arange(0,Time_end-100,10) #mod for end time

#include important correlations
corr_Ux_torq = []
corr_torq_MR = []
corr_torq_LSSMR = []
corr_Ux_MR = []
corr_Ux_LSS_MR = []
for Time_start in Time_starts:

    Time_OF = np.array(a.variables["time_OF"])
    Time_sampling = np.array(a.variables["time_sampling"])
    Time_sampling = Time_sampling - Time_sampling[0]
    Time_end = Time_sampling[-1]
    Time_end_idx = np.searchsorted(Time_OF,Time_end)

    dt = Time_OF[1] - Time_OF[0]

    Time_start_idx = np.searchsorted(Time_OF,Time_start)

    Time_OF = Time_OF[Time_start_idx:Time_end_idx]

    RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:Time_end_idx])
    RtAeroMyh = np.array(a.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
    RtAeroMzh = np.array(a.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])
    MR = np.sqrt( np.add(np.square(RtAeroMyh), np.square(RtAeroMzh)) ) 
    LSShftMys = np.array(a.variables["LSShftMys"][Time_start_idx:Time_end_idx])
    LSShftMzs = np.array(a.variables["LSShftMzs"][Time_start_idx:Time_end_idx])
    LSSMR = np.sqrt( np.add(np.square(LSShftMys), np.square(LSShftMzs)) ) 

    group = a.groups["0.0"]

    csvFile = pandas.read_csv(in_dir+'Ux.csv')

    Ux = np.array(csvFile["Ux"].to_list())

    f = interpolate.interp1d(Time_sampling,Ux)
    Ux = f(Time_OF)

    corr_Ux_torq.append(correlation_coef(Ux,RtAeroMxh))
    corr_torq_MR.append(correlation_coef(RtAeroMxh, MR))
    corr_torq_LSSMR.append(correlation_coef(RtAeroMxh,LSSMR))
    corr_Ux_MR.append(correlation_coef(Ux,MR))
    corr_Ux_LSS_MR.append(correlation_coef(Ux,LSSMR))

fig = plt.figure()
plt.plot(Time_starts, corr_Ux_torq)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between rotor averaged velocity and Torque")
plt.tight_layout()
plt.savefig(in_dir+"start_time_corr1.png")
plt.close()


fig = plt.figure()
plt.plot(Time_starts, corr_torq_MR)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Torque and rotor OOPBM")
plt.tight_layout()
plt.savefig(in_dir+"start_time_corr2.png")
plt.close()

fig = plt.figure()
plt.plot(Time_starts, corr_torq_LSSMR)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Torque and LSS OOPBM")
plt.tight_layout()
plt.savefig(in_dir+"start_time_corr3.png")
plt.close()


fig = plt.figure()
plt.plot(Time_starts, corr_Ux_MR)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Rotor averaged velocity and rotor OOPBM")
plt.tight_layout()
plt.savefig(in_dir+"start_time_corr4.png")
plt.close()

fig = plt.figure()
plt.plot(Time_starts, corr_Ux_LSS_MR)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between rotor averaged velocuty and LSS OOPBM")
plt.tight_layout()
plt.savefig(in_dir+"start_time_corr5.png")
plt.close()