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

df = pd.read_csv("../../../jarred/NAWEA_23/post_processing/out.csv")

dq = pd.read_csv("../../../jarred/NAWEA_23/post_processing/out2.csv")

time = df["Time_OF"]
time = remove_nan(time)

time2 = dq["Time_sample"]
time2 = remove_nan(time2)

Var = "Ux_0.0"

signal = df[Var]
signal = remove_nan(signal)

signal2 = dq[Var]
signal2 = remove_nan(signal2)

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(32,24))
ax1.plot(time,signal)
ax1.set_title("Ux' OF method",fontsize=18)
ax2.plot(time2,signal2)
ax2.set_title("Ux' sampling method",fontsize=18)
plt.tight_layout()
plt.savefig(dir+"Ux_signals.png")
plt.close(fig)