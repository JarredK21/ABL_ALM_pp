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


def low_pass_filter(signal, cutoff,dt):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"

offsets = [0.0, 63.0, 126.0]

a = Dataset(in_dir+"Dataset.nc")

fig = plt.figure(figsize=(14,8))
for offset in offsets:

    Time_sampling = np.array(a.variables["time_sampling"])

    group = a.groups["{}".format(offset)]

    Ux = np.array(group.variables["Ux"])


    plt.plot(Time_sampling, Ux)

Time_OF = np.array(a.variables["time_OF"])
Ux = np.array(a.variables["RtAeroVxh"])
cutoff = 0.5*(12.1/60)
dt = Time_OF[1] - Time_OF[0]
Ux = low_pass_filter(Ux, cutoff,dt)
plt.plot(Time_OF, Ux)



plt.xlabel("Time [s]")
plt.ylabel("Ux")
plt.ylim(bottom=6)
plt.legend(["0.0", "-63.0", "-126.0","Blade averaged velocity"])
plt.savefig(in_dir+"velocity_comp.png")
plt.close()