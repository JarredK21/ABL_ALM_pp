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

    return frq, PSD

dir = "../post_processing/"
#openfast data
df = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

tstart = 50
tend = 350
time_OF = np.array(df["Time_[s]"])
dt = time_OF[1] - time_OF[0]
tstart_OF_idx = np.searchsorted(time_OF,tstart)
tend_OF_idx = np.searchsorted(time_OF,tend)

signaly = df["RtAeroMyh_[N-m]"][tstart_OF_idx:tend_OF_idx]
frqy, FFT_signaly = temporal_spectra(signaly,dt)
signalz = df["RtAeroMzh_[N-m]"][tstart_OF_idx:tend_OF_idx]
frqz, FFT_signalz = temporal_spectra(signalz,dt)


signal_MR = np.sqrt( np.square(signaly) + np.square(signalz) ) 
frqMR, FFT_signalMR = temporal_spectra(signal_MR,dt)

signal_T = np.arctan2(signalz,signaly)
frqT, FFT_signalT = temporal_spectra(signal_T,dt)

fig, ax1,ax2,ax3,ax4 = plt.subplots(4,1,figsize=(32,24))
ax1.plot(time_OF[tstart_OF_idx:tend_OF_idx],signaly)
ax2.plot(time_OF[tstart_OF_idx:tend_OF_idx],signalz)
ax3.plot(time_OF[tstart_OF_idx:tend_OF_idx],signal_MR)
ax4.plot(time_OF[tstart_OF_idx:tend_OF_idx],signal_T)
plt.savefig(dir+"1P_signals.png")
plt.close(fig)


fig, ax1,ax2,ax3,ax4 = plt.subplots(4,1,figsize=(32,24))
ax1.plot(frqy,FFT_signaly)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.axvline(12.1/60)
ax1.axvline((12.1/60)*3)

ax2.plot(frqz,FFT_signalz)
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.axvline(12.1/60)
ax2.axvline((12.1/60)*3)

ax3.plot(frqMR,FFT_signalMR)
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.axvline(12.1/60)
ax3.axvline((12.1/60)*3)

ax4.plot(frqT,FFT_signalT)
ax4.set_yscale('log')
ax4.set_xscale('log')
ax4.axvline(12.1/60)
ax4.axvline((12.1/60)*3)

plt.savefig(dir+"1P_FFT_signals.png")
plt.close(fig)