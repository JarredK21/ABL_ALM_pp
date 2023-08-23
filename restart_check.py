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
from multiprocessing import Pool
import time

start_time = time.time()

OF_files = ["../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out",
             "../../NREL_5MW_MCBL_R_CRPM_100320/NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out",
             "../../NREL_5MW_MCBL_R_CRPM_100320_2/NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out"]


Variables = ["Wind1VelX","RtAeroFxh","RtAeroMxh","MR"]
units = ["[m/s]","[N]","[N-m]","[N-m]"]
colors = ["r", "b", "g"]


for iv in np.arange(0,len(Variables)):
    Variable = Variables[iv]
    unit = units[iv]

    fig = plt.figure(figsize=(14,8))

    ic = 0
    for file in OF_files:


        #openfast data
        da = io.fast_output_file.FASTOutputFile(file).toDataFrame()

        restart_time = 137.748
        Time_OF = np.array(da["Time_[s]"])
        if ic == 1 or ic == 2:
            Time_OF = Time_OF+restart_time


        if Variable == "MR" or Variable == "Theta":
            signaly = np.array(da["RtAeroMyh_[N-m]"])
            signalz = np.array(da["RtAeroMzh_[N-m]"])
            
            if Variable == "MR":
                signal = np.sqrt( np.square(signaly) + np.square(signalz) ) 
            elif Variable == "Theta": 
                signal = np.arctan2(signalz,signaly)

        else:
            txt = "{0}_{1}".format(Variable,units[iv])
            signal = np.array(da[txt])

        if ic == 1:
            plt.plot(Time_OF,signal+(0.1*np.max(signal)),color=colors[ic])
        elif ic == 2:
            plt.plot(Time_OF,signal+(0.2*np.max(signal)),color=colors[ic])
        else:
            plt.plot(Time_OF,signal,color=colors[ic])
        ic+=1
        print("line 195",time.time()-start_time)

    plt.xlabel("Time [s]")
    plt.ylabel("{0} {1}".format(Variable,unit))
    plt.xlim([120,160])
    plt.legend(["(1)", "(2a)+20%max", "(2b)"])
    plt.savefig("./{0}".format(Variable))
    plt.close(fig)