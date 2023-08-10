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


def temporal_spectra(signal,dt):

    fs =1/dt
    n = len(signal) 
    if n%2==0:
        nhalf = int(n/2+1)
    else:
        nhalf = int((n+1)/2)
    frq = np.arange(nhalf)*fs/n
    Y   = np.fft.fft(signal)
    PSD = abs(Y[range(nhalf)])**2 /(n*fs) # PSD
    PSD[1:-1] = PSD[1:-1]*2


    energy_contents_check(PSD,signal,dt)

    return frq, PSD


def energy_contents_check(e_fft,signal,dt):

    E = (1/dt)*np.sum(e_fft)

    q = np.sum(np.square(signal))

    E2 = q

    print(E, E2, abs(E2/E))    


df = pd.read_csv("out.csv")

time_sample = remove_nan("Time_sample")

dt_sample = time_sample[1] - time_sample[0]
print(dt_sample)

signal = remove_nan("Ux")

df = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

time_OF = df["Time_[s]"]
dt_OF = time_OF[1] - time_OF[0]
print(dt_OF)
Ux_it_OF = df["RtVAvgxh_[m/s]"]

fig = plt.figure()
plt.plot(time_sample,signal,"r-")
plt.plot(time_OF,Ux_it_OF,"b-")
plt.xlabel("Time [s]")
plt.ylabel("Rotor velocity")
plt.legend(["Sampled", "OF"])
plt.savefig("test.png")
plt.close(fig)


frq_sample, FFT_signal_sample = temporal_spectra(signal,dt_sample)

frq_OF, FFT_signal_OF = temporal_spectra(Ux_it_OF,dt_OF)

fig = plt.figure()
plt.plot(frq_sample,FFT_signal_sample,"r-")
plt.plot(frq_OF,FFT_signal_OF,"b-")
plt.yscale("log"); plt.xscale("log")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Spectral energy density [$m^2/s^2$]")
plt.legend(["Sampled", "OF"])
plt.savefig("sampled_OF_vel_spectra.png")
plt.close(fig)