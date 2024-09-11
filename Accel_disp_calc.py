from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.signal import butter,filtfilt
from netCDF4 import Dataset
import pyFAST.input_output as io
from scipy.fft import fft, fftfreq, fftshift,ifft

def energy_contents_check(Var,e_fft,signal,dt):

    E = (1/dt)*np.sum(e_fft)

    q = np.sum(np.square(signal))

    E2 = q

    print(Var, E, E2, abs(E2/E))    


def temporal_spectra(signal,dt,Var):

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


    energy_contents_check(Var,PSD,signal,dt)

    return frq, PSD

def high_pass_filter(signal, cutoff,dt):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 6  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal

def low_pass_filter(signal, cutoff,dt):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 6  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


def hard_filter(signal,cutoff,dt,filter_type):

    N = len(signal)
    spectrum = fft(signal)
    F = fftfreq(N,dt)
    if filter_type=="lowpass":
        spectrum_filter = spectrum*(np.absolute(F)<cutoff)
    elif filter_type=="highpass":
        spectrum_filter = spectrum*(np.absolute(F)>cutoff)
    elif filter_type=="bandpass":
        spectrum_filter = spectrum*(np.absolute(F)>cutoff[0])
        spectrum_filter = spectrum_filter*(np.absolute(F)<cutoff[1])
        

    spectrum_filter = ifft(spectrum_filter)

    return spectrum_filter




in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

Time = np.array(df["Time_[s]"])
dt = Time[1]

FLy_E = np.array(df["B1N016FLy_[kN]"])


in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

FLy_R = np.array(df["B1N016FLy_[kN]"])


A_E = np.subtract(FLy_E,FLy_R)

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(FLy_R,dt,Var="Rigid")
plt.loglog(frq,PSD,"-b",label="Rigid")
frq,PSD = temporal_spectra(FLy_E,dt,Var="Elastic")
plt.loglog(frq,PSD,"-r",label="Elastic")
plt.grid()
plt.legend()

fig = plt.figure(figsize=(14,8))
plt.plot(Time,FLy_E,"-r",label="Elastic")
plt.plot(Time,FLy_R,"-b",label="Rigid")
plt.grid()
plt.legend()


fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(A_E,dt,Var="Accel")
plt.loglog(frq,PSD)
plt.grid()

A_E = np.true_divide(A_E,1.39E+02)
A_E[0] = 0.0
fig = plt.figure(figsize=(14,8))
plt.plot(Time,A_E,label="Aceleration")
plt.grid()

A_E = np.subtract(A_E,np.mean(A_E))

A_E = hard_filter(A_E,[8E-03,40],dt,"bandpass")

y_int = integrate.cumtrapz(A_E, Time,initial=0)

plt.plot(Time,y_int,"-r",label="Velocity")

A_E = np.subtract(A_E,np.mean(A_E))

A_E = hard_filter(A_E,[8E-03,40],dt,"bandpass")

# y_int_int = integrate.cumtrapz(y_int, Time,initial=0)


# plt.plot(Time,y_int_int,"-k",label="Displacement")
plt.legend()



fig = plt.figure()
frq,PSD=temporal_spectra(A_E,dt,Var="filtered accel")
plt.loglog(frq,PSD)
plt.grid()

# frq,PSD = temporal_spectra(y_int_int,dt,Var="Disp")
# fig = plt.figure()
# plt.loglog(frq,PSD)
# plt.grid()

plt.show()