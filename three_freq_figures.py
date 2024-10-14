import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import statistics
from matplotlib.patches import Circle
from scipy.signal import butter,filtfilt
from scipy import interpolate
import time
from scipy.fft import fft, fftfreq, fftshift,ifft
import matplotlib.patches as patches



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


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(y)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt



def correlation_coef(x,y):

    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


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


def probability_dist(y):

    mu = np.mean(y)
    var = np.var(y)
    sd = np.std(y)
    no_bin = 1000
    X = np.linspace(np.min(y),np.max(y),no_bin)
    dX = X[1] - X[0]
    P = []
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
    print(np.sum(P)*dX)
    return P,X


def moments(y):
    mu = np.mean(y)
    std = np.std(y)
    N = len(y)

    skewness = (np.sum(np.power(np.subtract(y,mu),3)))/(N*std**3)
    kurotsis = (np.sum(np.power(np.subtract(y,mu),4)))/(N*std**4)

    return mu, std, skewness,kurotsis



def actuator_asymmetry_calc(it):
    R = np.linspace(0,63,300)
    dr = R[1] - R[0]
    hvelB1 = np.add(np.cos(np.radians(29))*uvelB1[it], np.sin(np.radians(29))*vvelB1[it])
    IyB1 = hvelB1*R*np.cos(np.radians(Azimuth[it]))
    IyB1 = np.sum(IyB1)*dr
    IzB1 = hvelB1*R*np.sin(np.radians(Azimuth[it]))
    IzB1 = np.sum(IzB1)*dr

    hvelB2 = np.add(np.cos(np.radians(29))*uvelB2[it], np.sin(np.radians(29))*vvelB2[it])
    AzB2 = Azimuth[it] + 120
    if AzB2 >= 360:
        AzB2-=360

    IyB2 = hvelB2*R*np.cos(np.radians(AzB2))
    IzB2 = hvelB2*R*np.sin(np.radians(AzB2))
    IyB2 = np.sum(IyB2)*dr
    IzB2 = np.sum(IzB2)*dr

    hvelB3 = np.add(np.cos(np.radians(29))*uvelB3[it], np.sin(np.radians(29))*vvelB3[it])
    AzB3 = Azimuth[it] + 240
    if AzB3 >= 360:
        AzB3-=360

    IyB3 = hvelB3*R*np.cos(np.radians(AzB3))
    IzB3 = hvelB3*R*np.sin(np.radians(AzB3))
    IyB3 = np.sum(IyB3)*dr
    IzB3 = np.sum(IzB3)*dr

    return IyB1+IyB2+IyB3, IzB1+IzB2+IzB3



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df_OF = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(df_OF.variables["Time_OF"])
dt = Time_OF[1] - Time_OF[0]

Time_start = 200
Time_start_idx = np.searchsorted(Time_OF,Time_start)

Time_OF = Time_OF[Time_start_idx:]

Time_steps = np.arange(0,len(Time_OF))

OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"][Time_start_idx:])
RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"][Time_start_idx:])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000


RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])
RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

FR = np.sqrt(np.add(np.square(RtAeroFys),np.square(RtAeroFzs)))
MR = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)




FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

FBR_LPF = hard_filter(FBR,0.3,dt,"lowpass")
FBR_BPF = hard_filter(FBR,[0.3,0.9,],dt,"bandpass")
FBR_HPF = hard_filter(FBR,[1.5,40],dt,"bandpass")


dLPF_FBR = np.array(dt_calc(FBR_LPF,dt))
zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_FBR)))[0]
dBPF_FBR = np.array(dt_calc(FBR_BPF,dt))
zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
dHPF_FBR = np.array(dt_calc(FBR_HPF,dt))
zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]

dF_LPF = []
dt_LPF = []
T_LPF = []
for i in np.arange(0,len(zero_crossings_index_LPF_FBR)-1):
    it_1 = zero_crossings_index_LPF_FBR[i]
    it_2 = zero_crossings_index_LPF_FBR[i+1]
    T_LPF.append(Time_OF[it_1])
    dt_LPF.append(Time_OF[it_2]-Time_OF[it_1])
    dF_LPF.append(FBR_LPF[it_2]-FBR_LPF[it_1])

dF_LPF = dF_LPF/((L1+L2)/L2)

dF_LPF_threshold = []
dt_LPF_threshold = []
T_LPF_threshold = []
threshold = np.mean(FBR_LPF)+2*np.std(FBR_LPF)
for i in np.arange(0,len(dF_LPF)):
    if dF_LPF[i] >= threshold:
        dF_LPF_threshold.append(dF_LPF[i])
        dt_LPF_threshold.append(dt_LPF[i])
        T_LPF_threshold.append(T_LPF[i])

dF_LPF_threshold = dF_LPF_threshold/((L1+L2)/L2)

dF_BPF = []
dt_BPF = []
T_BPF = []
for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-1):
    it_1 = zero_crossings_index_BPF_FBR[i]
    it_2 = zero_crossings_index_BPF_FBR[i+1]
    T_BPF.append(Time_OF[it_1])
    dt_BPF.append(Time_OF[it_2]-Time_OF[it_1])
    dF_BPF.append(FBR_BPF[it_2]-FBR_BPF[it_1])

dF_BPF = dF_BPF/((L1+L2)/L2)

dF_BPF_threshold = []
dt_BPF_threshold = []
T_BPF_threshold = []
threshold = np.mean(FBR_BPF)+2*np.std(FBR_BPF)
for i in np.arange(0,len(dF_BPF)):
    if dF_BPF[i] >= threshold:
        dF_BPF_threshold.append(dF_BPF[i])
        dt_BPF_threshold.append(dt_BPF[i])
        T_BPF_threshold.append(T_BPF[i])

dF_BPF_threshold = dF_BPF_threshold/((L1+L2)/L2)


dF_HPF = []
dt_HPF = []
T_HPF = []
for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):
    it_1 = zero_crossings_index_HPF_FBR[i]
    it_2 = zero_crossings_index_HPF_FBR[i+1]
    T_HPF.append(Time_OF[it_1])
    dt_HPF.append(Time_OF[it_2]-Time_OF[it_1])
    dF_HPF.append(FBR_HPF[it_2]-FBR_HPF[it_1])

dF_HPF = dF_HPF/((L1+L2)/L2)

dF_HPF_threshold = []
dt_HPF_threshold = []
T_HPF_threshold = []
threshold = np.mean(FBR_HPF)+2*np.std(FBR_HPF)
for i in np.arange(0,len(dF_HPF)):
    if dF_LPF[i] >= threshold:
        dF_HPF_threshold.append(dF_HPF[i])
        dt_HPF_threshold.append(dt_HPF[i])
        T_HPF_threshold.append(T_HPF[i])

dF_HPF_threshold = dF_HPF_threshold/((L1+L2)/L2)

MR = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))
MR_LPF = hard_filter(MR,0.3,dt,"lowpass")
MR_BPF = hard_filter(MR,[0.3,0.9,],dt,"bandpass")
MR_HPF = hard_filter(MR,[1.5,40],dt,"bandpass")


#blade asymmetry
df = Dataset(in_dir+"WTG01a.nc")

uvelB1 = np.array(df.variables["uvel"][Time_start_idx:,1:301])
vvelB1 = np.array(df.variables["vvel"][Time_start_idx:,1:301])
uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
hvelB1 = np.add(np.cos(np.radians(29))*uvelB1, np.sin(np.radians(29))*vvelB1)
uvelB2 = np.array(df.variables["uvel"][Time_start_idx:,301:601])
vvelB2 = np.array(df.variables["vvel"][Time_start_idx:,301:601])
uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
hvelB2 = np.add(np.cos(np.radians(29))*uvelB2, np.sin(np.radians(29))*vvelB2)
uvelB3 = np.array(df.variables["uvel"][Time_start_idx:,601:901])
vvelB3 = np.array(df.variables["vvel"][Time_start_idx:,601:901])
uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
hvelB3 = np.add(np.cos(np.radians(29))*uvelB3, np.sin(np.radians(29))*vvelB3)


Azimuth = np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:])
IBy = []
IBz = []
ix=0
with Pool() as pool:
    for Iy_it, Iz_it,in pool.imap(actuator_asymmetry_calc,Time_steps[:-1]):
        IBy.append(Iy_it); IBz.append(Iz_it)
        
        print(ix)
        ix+=1

IB = np.sqrt(np.add(np.square(IBy),np.square(IBz)))
IB_LPF = hard_filter(IB,0.3,dt,"lowpass")
IB_BPF = hard_filter(IB,[0.3,0.9,],dt,"bandpass")
IB_HPF = hard_filter(IB,[1.5,40],dt,"bandpass")




plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/Figures/"

cc=round(correlation_coef(FBR,MR),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF,MR,"-b")
ax.set_ylabel("Rotor out-of-plane bending moment magnitude [kN-m]")
ax.yaxis.label.set_color('blue')
ax2=ax.twinx()
ax2.plot(Time_OF,FBR,"-r",alpha=0.5)
ax2.set_ylabel("Bearing out-of-plane force magnitude [kN]")
ax2.yaxis.label.set_color('red')
ax.grid()
fig.supxlabel("Time [s]")
fig.suptitle("Correlation coefficient = {}".format(cc))
plt.tight_layout()
plt.savefig(out_dir+"cc_FBR_MR.png")
plt.close(fig)


cc=round(correlation_coef(IB,MR[:-1]),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF,MR,"-b")
ax.set_ylabel("Rotor out-of-plane bending moment magnitude [kN-m]")
ax.yaxis.label.set_color('blue')
ax2=ax.twinx()
ax2.plot(Time_OF[:-1],IB,"-r")
ax2.set_ylabel("Blade Asymmetry vector magnitude [$m^3/s$]")
ax2.yaxis.label.set_color('red')
ax.grid()
fig.supxlabel("Time [s]")
fig.suptitle("Correlation coefficient = {}".format(cc))
plt.tight_layout()
plt.savefig(out_dir+"cc_IB_MR.png")
plt.close(fig)


fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax1.scatter(T_LPF,dF_LPF,s=1)
ax1.set_title("$dF$ LPF $F_{B_R}$ [kN]")
ax1_1=ax1.twiny()
P,X = probability_dist(dF_LPF)
ax1_1.plot(P,X,"-k")
