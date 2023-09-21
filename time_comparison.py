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
out_dir = in_dir+"time vs correlation/"

offsets = [0.0]

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_end_idx = np.searchsorted(Time_OF,1000)
Time_end = Time_OF[Time_end_idx]


Time_starts = np.arange(0,Time_end-100,10) #mod for end time

#include important correlations
corr_Ux_torq = []
corr_torq_MR = []
corr_torq_LSSMR = []
corr_Ux_MR = []
corr_Ux_LSS_MR = []
corr_IA_MR = []
corr_IA_Mx = []
corr_IA_Ux = []
for Time_start in Time_starts:

    Time_OF = np.array(a.variables["time_OF"])
    Time_sampling = np.array(a.variables["time_sampling"])
    Time_sampling = Time_sampling - Time_sampling[0]
    #Time_end = Time_sampling[-1]
    #Time_end_idx = np.searchsorted(Time_OF,1000)

    dt = Time_OF[1] - Time_OF[0]

    Time_start_idx = np.searchsorted(Time_OF,Time_start)

    Time_OF = Time_OF[Time_start_idx:Time_end_idx]

    RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:Time_end_idx])
    RtAeroMyh = np.array(a.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
    RtAeroMzh = np.array(a.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])
    RtAeroMR = np.sqrt( np.add(np.square(RtAeroMyh), np.square(RtAeroMzh)) ) 
    LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
    LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])
    LSSMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) ) 

    group = a.groups["0.0"]

    Ux = np.array(group.variables["Ux"])

    IA = np.array(group.variables["IA"])

    f = interpolate.interp1d(Time_sampling,Ux)
    Ux = f(Time_OF)

    f = interpolate.interp1d(Time_sampling,IA)
    IA = f(Time_OF)

    corr_Ux_torq.append(correlation_coef(Ux,RtAeroMxh))
    corr_torq_MR.append(correlation_coef(RtAeroMxh, RtAeroMR))
    corr_torq_LSSMR.append(correlation_coef(RtAeroMxh,LSSMR))
    corr_Ux_MR.append(correlation_coef(Ux,RtAeroMR))
    corr_Ux_LSS_MR.append(correlation_coef(Ux,LSSMR))
    corr_IA_MR.append(correlation_coef(IA,RtAeroMR))
    corr_IA_Mx.append(correlation_coef(IA,RtAeroMxh))
    corr_IA_Ux.append(correlation_coef(IA,Ux))

fig = plt.figure()
plt.plot(Time_starts, corr_Ux_torq)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between rotor averaged velocity and Torque")
plt.tight_layout()
plt.savefig(out_dir+"start_time_corr1_v2.png")
plt.close()


fig = plt.figure()
plt.plot(Time_starts, corr_torq_MR)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Torque and rotor OOPBM")
plt.tight_layout()
plt.savefig(out_dir+"start_time_corr2_v2.png")
plt.close()

fig = plt.figure()
plt.plot(Time_starts, corr_torq_LSSMR)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Torque and LSS OOPBM")
plt.tight_layout()
plt.savefig(out_dir+"start_time_corr3_v2.png")
plt.close()


fig = plt.figure()
plt.plot(Time_starts, corr_Ux_MR)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Rotor averaged velocity and rotor OOPBM")
plt.tight_layout()
plt.savefig(out_dir+"start_time_corr4_v2.png")
plt.close()

fig = plt.figure()
plt.plot(Time_starts, corr_Ux_LSS_MR)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between rotor averaged velocity and LSS OOPBM")
plt.tight_layout()
plt.savefig(out_dir+"start_time_corr5_v2.png")
plt.close()

fig = plt.figure()
plt.plot(Time_starts, corr_IA_MR)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Asymmetry parameter and Rotor OOPBM")
plt.tight_layout()
plt.savefig(out_dir+"start_time_corr6_v2.png")
plt.close()

fig = plt.figure()
plt.plot(Time_starts, corr_IA_Mx)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Asymmetry parameter and Rotor Torque")
plt.tight_layout()
plt.savefig(out_dir+"start_time_corr7_v2.png")
plt.close()

fig = plt.figure()
plt.plot(Time_starts, corr_IA_Ux)
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Asymmetry parameter and Rotor averaged velocity")
plt.tight_layout()
plt.savefig(out_dir+"start_time_corr8_v2.png")
plt.close()