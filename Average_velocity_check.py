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

def remove_nan(Var):

    signal = df[Var]
    new_signal = list()
    for element in signal:
        if not np.isnan(element):
            new_signal.append(element)
    return new_signal


df = pd.read_csv("out.csv")

time_sample = remove_nan("Time_sample")

signal = remove_nan("Ux")

df = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

time_OF = df["Time_[s]"]
Ux_it_OF = df["RtVAvgxh_[m/s]"]

fig = plt.figure()
plt.plot(time_sample,signal,"r-")
plt.plot(time_OF,Ux_it_OF,"b-")
plt.xlabel("Time [s]")
plt.ylabel("Rotor velocity")
plt.legend(["Sampled", "OF"])
plt.savefig("test.png")
plt.close(fig)