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



#create netcdf file
ncfile = Dataset("./Dataset_2.nc",mode="w",format='NETCDF4') #change name
ncfile.title = "OpenFast data sampling output combined"

#create global dimensions
OF_dim = ncfile.createDimension("OF",None)

#create variables
time_OF = ncfile.createVariable("time_OF", np.float64, ('OF',),zlib=True)

RtAeroVxh = ncfile.createVariable("RtAeroVxh", np.float64, ('OF',),zlib=True)
RtAeroFxh = ncfile.createVariable("RtAeroFxh", np.float64, ('OF',),zlib=True)
RtAeroMxh = ncfile.createVariable("RtAeroMxh", np.float64, ('OF',),zlib=True)
RtAeroMrh = ncfile.createVariable("RtAeroMrh", np.float64, ('OF',),zlib=True)
Theta = ncfile.createVariable("Theta", np.float64, ('OF',),zlib=True)
LSShftFya = ncfile.createVariable("LSShftFya", np.float64, ('OF',),zlib=True)
LSShftFza = ncfile.createVariable("LSShftFza", np.float64, ('OF',),zlib=True)
LSShftFys = ncfile.createVariable("LSShftFys", np.float64, ('OF',),zlib=True)
LSShftFzs = ncfile.createVariable("LSShftFzs", np.float64, ('OF',),zlib=True)
LSShftMya = ncfile.createVariable("LSShftMya", np.float64, ('OF',),zlib=True)
LSShftMza = ncfile.createVariable("LSShftMza", np.float64, ('OF',),zlib=True)
LSShftMys = ncfile.createVariable("LSShftMys", np.float64, ('OF',),zlib=True)
LSShftMzs = ncfile.createVariable("LSShftMzs", np.float64, ('OF',),zlib=True)

df = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

time_OF[:] = np.array(df["Time_[s]"])

Variables = ["Wind1VelX","RtAeroFxh","RtAeroMxh","MR","Theta","LSShftFya","LSShftFza","LSShftFys","LSShftFzs",
             "LSShftMya","LSShftMza","LSShftMys","LSShftMzs"]
units = ["[m/s]","[N]","[N-m]","[N-m]","[rads]","[kN]","[kN]","[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]","[kN-m]"]


for iv in np.arange(0,len(Variables)):
    Variable = Variables[iv]

    if Variable == "MR" or Variable == "Theta":
        signaly = np.array(df["RtAeroMyh_[N-m]"])
        signalz = np.array(df["RtAeroMzh_[N-m]"])
        
        if Variable == "MR":
            signal = np.sqrt( np.square(signaly) + np.square(signalz) ) 
            RtAeroMrh[:] = signal; del signal
        elif Variable == "Theta": 
            signal = np.arctan2(signalz,signaly)
            Theta[:] = signal; del signal

    else:
        txt = "{0}_{1}".format(Variable,units[iv])
        signal = np.array(df[txt])
        if Variable == "RtAeroFxh":
            RtAeroFxh[:] = signal; del signal
        elif Variable == "RtAeroMxh":
            RtAeroMxh[:] = signal; del signal
        elif Variable == "Wind1VelX":
            RtAeroVxh[:] = signal; del signal
        elif Variable == "LSShftFya":
            print(signal[0])
            LSShftFya[:] = signal; del signal
        elif Variable == "LSShftFza":
            LSShftFza[:] = signal; del signal
        elif Variable == "LSShftFys":
            LSShftFys[:] = signal; del signal
        elif Variable == "LSShftFzs":
            LSShftFzs[:] = signal; del signal
        elif Variable == "LSShftMya":
            LSShftMya[:] = signal; del signal
        elif Variable == "LSShftMza":
            LSShftMza[:] = signal; del signal
        elif Variable == "LSShftMys":
            LSShftMys[:] = signal; del signal
        elif Variable == "LSShftMzs":
            LSShftMzs[:] = signal; del signal
