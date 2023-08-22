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


in_dirs = ["../../ALM_sensitivity_analysis_nhalf/fllc_Ex2/post_processing/", 
           "../../ALM_sensitivity_analysis_nhalf/eps_c_0.5/post_processing/",
           "../../ALM_sensitivity_analysis_nhalf/eps_6_lv4/post_processing/"]

corr_Ux_torq = []
corr_torq_MR = []
for i in np.arange(0,len(in_dirs)):
    in_dir = in_dirs[i]

    a = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()
            

    Time_starts = np.arange(0,24,1)

    corr_Ux_torq_i = []
    corr_torq_MR_i = []
    for Time_start in Time_starts:

        Time_OF = np.array(a["Time_[s]"])

        Time_start_idx = np.searchsorted(Time_OF,Time_start)

        Time_OF = Time_OF[Time_start_idx:]

        RtAeroMxh = np.array(a["RtAeroMxh_[N-m]"][Time_start_idx:])
        signaly = np.array(a["RtAeroMyh_[N-m]"][Time_start_idx:])
        signalz = np.array(a["RtAeroMzh_[N-m]"][Time_start_idx:])
        MR = np.sqrt( np.square(signaly) + np.square(signalz) ) 
        Ux = np.array(a["Wind1VelX_[m/s]"][Time_start_idx:])

        corr_Ux_torq_i.append(correlation_coef(Ux,RtAeroMxh))
        corr_torq_MR_i.append(correlation_coef(RtAeroMxh, MR))

    corr_torq_MR.append(corr_torq_MR_i)
    corr_Ux_torq.append(corr_Ux_torq_i)


fig = plt.figure()
plt.plot(Time_starts, corr_Ux_torq[0],"r-")
plt.plot(Time_starts, corr_Ux_torq[1],"b-")
plt.plot(Time_starts, corr_Ux_torq[2],"-g")
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Rotor averaged velocity and Torque")
plt.legend(["FLLC","epc/c classical", "Fixed eps classical"])
plt.savefig("../../ALM_sensitivity_analysis_nhalf/start_time_corr1.png")
plt.close()


fig = plt.figure()
plt.plot(Time_starts, corr_torq_MR[0],"r-")
plt.plot(Time_starts, corr_torq_MR[1],"b-")
plt.plot(Time_starts, corr_torq_MR[2],"-g")
plt.xlabel("Start time [s]")
plt.ylabel("correlation between Torque and OOPBM")
plt.legend(["FLLC","epc/c classical", "Fixed eps classical"])
plt.savefig("../../ALM_sensitivity_analysis_nhalf/start_time_corr2.png")
plt.close()