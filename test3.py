from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
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


def dt_calc(y,dt):

    dy_dt = []
    for it in np.arange(0,len(y)-1):
        dy_dt.append((y[it+1]-y[it])/dt)

    return np.array(dy_dt)



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

    return np.real(spectrum_filter)


def correlation_coef(x,y):

    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r




in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

Time_OF = np.array(df["Time_[s]"])
dt = Time_OF[1] - Time_OF[0]
print(dt)

Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+10)

Time_OF = Time_OF[Start_time_idx:]

# Wind1 = np.array(df["Wind1VelX_[m/s]"][Start_time_idx:])
# Wind2 = np.array(df["Wind2VelX_[m/s]"][Start_time_idx:])
# Wind3 = np.array(df["Wind3VelX_[m/s]"][Start_time_idx:])
# Wind4 = np.array(df["Wind4VelX_[m/s]"][Start_time_idx:])
# Wind5 = np.array(df["Wind5VelX_[m/s]"][Start_time_idx:])#82.5m
# Wind6 = np.array(df["Wind6VelX_[m/s]"][Start_time_idx:])
# Wind7 = np.array(df["Wind7VelX_[m/s]"][Start_time_idx:])
# Wind8 = np.array(df["Wind8VelX_[m/s]"][Start_time_idx:])
# Wind9 = np.array(df["Wind9VelX_[m/s]"][Start_time_idx:])

# Z = [7.5, 22.5, 37.5, 52.5, 82.5, 97.5, 112.5, 127.5, 157.5]
# Mean_wind_profile = [np.average(Wind1),np.average(Wind2),np.average(Wind3),np.average(Wind4),np.average(Wind5),np.average(Wind6),np.average(Wind7),np.average(Wind8),np.average(Wind9)]

# plt.rcParams['font.size'] = 16

# out_dir=in_dir+"plots/"
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,Wind5)
# plt.xlabel("Time [s]")
# plt.ylabel("Streamwise velocity hub height [m/s]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"hub_height_Ux.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(Wind5,dt,Var="Wind5")
# plt.loglog(frq,PSD)
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD Streamwise velocity hub height [m/s]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"spectra_hub_height_Ux.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# plt.plot(Mean_wind_profile,Z)
# plt.xlabel("Mean horizontal velocity T = 60s [m/s]")
# plt.ylabel("Height from surface [m]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Ux_z.png")
# plt.close(fig)

# TDx = np.array(df["B1N021TDx_[m]"][Start_time_idx:])
# TDy = np.array(df["B1N021TDy_[m]"][Start_time_idx:])

# dvx_dt = dt_calc(TDx,dt); dAx_dt = dt_calc(dvx_dt,dt)
# dAx_dt_LPF = hard_filter(dAx_dt,1.0,dt,"lowpass")

# ADx = np.array(df["B1N021ALx_[m/s^2]"][Start_time_idx:]); ADx_LPF = hard_filter(ADx,1.0,dt,"lowpass")
# ADy = np.array(df["B1N021ALy_[m/s^2]"][Start_time_idx:])

FLx = np.array(df["B1N021FLx_[kN]"][Start_time_idx:])


fig,ax = plt.subplots()
ax.plot(Time_OF,FLx,"-r")


plt.figure()
frq,PSD = temporal_spectra(FLx,dt,"FLx")
plt.loglog(frq,PSD)

plt.show()