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


def butterwort_low_pass_filer(signal,cutoff,dt):

    M = signal.shape
    H = np.zeros((M), dtype=np.float32)
    D0 = cutoff #cut off frequency

    fs =1/dt
    n = len(signal)
    if n%2==0:
        nhalf = int(n/2+1)
    else:
        nhalf = int((n+1)/2)
    frq = np.arange(nhalf)*fs/n


    for u in range(M):
        D = frq[u]
        if D >= D0:
            H[u] = 0
        else:
            H[u] = 1

    return H


def hard_low_pass_filter(signal,cutoff,dt):
    #FFT
    ufft = np.fft.fftshift(np.fft.fft(signal))


    #multiply filter
    H = butterwort_low_pass_filer(signal,cutoff,dt)
    ufft_filt = ufft * H

    #IFFT
    ufft_filt_shift = np.fft.ifftshift(ufft_filt)
    iufft_filt = np.real(np.fft.ifft(ufft_filt_shift))

    return iufft_filt


def low_pass_filter(signal, cutoff,dt):

    fs = 1/dt     # sample rate, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


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


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360
        

def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(y)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt


def dt_calc_theta(y,dt):
    d_dt = []
    for i in np.arange(0,len(Time_OF)-1):
        di_dt = (y[i+1]-y[i])/dt
        if abs(di_dt) > (180/dt):
            if di_dt < 0:
                d_dt.append((360-y[i]+y[i+1])/dt)
            elif di_dt > 0:
                d_dt.append((y[i+1]-360-y[i])/dt)
        else:
            d_dt.append((y[i+1]-y[i])/dt)

    return d_dt



def actuator_asymmetry_calc(it):
    hvelB1 = np.add(np.cos(np.radians(29))*uvelB1[it], np.sin(np.radians(29))*vvelB1[it])
    IyB1 = hvelB1*R*np.cos(np.radians(Azimuth[it]))
    IyB1 = np.sum(IyB1)
    IzB1 = hvelB1*R*np.sin(np.radians(Azimuth[it]))
    IzB1 = np.sum(IzB1)

    hvelB2 = np.add(np.cos(np.radians(29))*uvelB2[it], np.sin(np.radians(29))*vvelB2[it])
    AzB2 = Azimuth[it] + 120
    if AzB2 >= 360:
        AzB2-=360

    IyB2 = hvelB2*R*np.cos(np.radians(AzB2))
    IzB2 = hvelB2*R*np.sin(np.radians(AzB2))
    IyB2 = np.sum(IyB2)
    IzB2 = np.sum(IzB2)

    hvelB3 = np.add(np.cos(np.radians(29))*uvelB3[it], np.sin(np.radians(29))*vvelB3[it])
    AzB3 = Azimuth[it] + 240
    if AzB3 >= 360:
        AzB3-=360

    IyB3 = hvelB3*R*np.cos(np.radians(AzB3))
    IzB3 = hvelB3*R*np.sin(np.radians(AzB3))
    IyB3 = np.sum(IyB3)
    IzB3 = np.sum(IzB3)

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


MR = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))
LPF_1_MR = low_pass_filter(MR,0.3,dt)
LPF_2_MR = low_pass_filter(MR,0.9,dt)
LPF_3_MR = low_pass_filter(MR,1.5,dt)
BPF_MR = np.subtract(LPF_2_MR,LPF_1_MR)
HPF_MR = np.subtract(MR,LPF_3_MR)
HPF_MR = np.array(low_pass_filter(HPF_MR,40,dt))


#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
LPF_1_FBR = low_pass_filter(FBR,0.3,dt)
LPF_2_FBR = low_pass_filter(FBR,0.9,dt)
LPF_3_FBR = low_pass_filter(FBR,1.5,dt)
BPF_FBR = np.subtract(LPF_2_FBR,LPF_1_FBR)
HPF_FBR = np.subtract(FBR,LPF_3_FBR)
HPF_FBR = np.array(low_pass_filter(HPF_FBR,40,dt))



df = Dataset(in_dir+"WTG01.nc")

Time = np.array(df.variables["time"])
dt = Time[1] - Time[0]
Tstart_idx = np.searchsorted(Time,200)
Time = Time[Tstart_idx:]

uvelB1 = np.array(df.variables["uvel"][Tstart_idx:,1:301])
vvelB1 = np.array(df.variables["vvel"][Tstart_idx:,1:301])
uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
hvelB1 = np.add(np.cos(np.radians(29))*uvelB1, np.sin(np.radians(29))*vvelB1)
uvelB2 = np.array(df.variables["uvel"][Tstart_idx:,301:601])
vvelB2 = np.array(df.variables["vvel"][Tstart_idx:,301:601])
uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
hvelB2 = np.add(np.cos(np.radians(29))*uvelB2, np.sin(np.radians(29))*vvelB2)
uvelB3 = np.array(df.variables["uvel"][Tstart_idx:,601:901])
vvelB3 = np.array(df.variables["vvel"][Tstart_idx:,601:901])
uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
hvelB3 = np.add(np.cos(np.radians(29))*uvelB3, np.sin(np.radians(29))*vvelB3)

R = np.linspace(0,63,300)


Iy = []
Iz = []
ix=0
with Pool() as pool:
    for Iy_it, Iz_it in pool.imap(actuator_asymmetry_calc,np.arange(0,len(Time))):
        Iy.append(Iy_it); Iz.append(Iz_it)
        print(ix)
        ix+=1

IB = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
LPF_1_IB = low_pass_filter(IB,0.3,dt)
LPF_2_IB = low_pass_filter(IB,0.9,dt)
LPF_3_IB = low_pass_filter(IB,1.5,dt)
BPF_IB = np.subtract(LPF_2_IB,LPF_1_IB)
HPF_IB = np.subtract(IB,LPF_3_IB)
HPF_IB = np.array(low_pass_filter(HPF_IB,40,dt))

plt.rcParams['font.size'] = 16
out_dir=in_dir+"High_frequency_analysis/"
fig = plt.figure(figsize=(14,8))
P,X = probability_dist(HPF_IB)
plt.yscale("log")
plt.plot(X,P)
plt.xlabel("HPF Blade Asymmetry [$m^2/s$]")
plt.ylabel("Probability [-]")
plt.grid()
M = []
for m in moments(HPF_IB):
    M.append(round(m,2))

plt.title("standard deviation: {}, Skewness: {}, Flatness: {}".format(M[1],M[2],M[3]))
plt.tight_layout()
plt.savefig(out_dir+"PDF_HPF_IB.png")
plt.close()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(HPF_MR)
plt.yscale("log")
plt.plot(X,P)
plt.xlabel("HPF Out-of-plane bending moment [kN-m]")
plt.ylabel("Probability [-]")
plt.grid()
M = []
for m in moments(HPF_MR):
    M.append(round(m,2))

plt.title("standard deviation: {}, Skewness: {}, Flatness: {}".format(M[1],M[2],M[3]))
plt.tight_layout()
plt.savefig(out_dir+"PDF_HPF_MR.png")
plt.close()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(HPF_FBR)
plt.yscale("log")
plt.plot(X,P)
plt.xlabel("HPF Bearing force [kN]")
plt.ylabel("Probability [-]")
plt.grid()
M = []
for m in moments(HPF_FBR):
    M.append(round(m,2))

plt.title("standard deviation: {}, Skewness: {}, Flatness: {}".format(M[1],M[2],M[3]))
plt.tight_layout()
plt.savefig(out_dir+"PDF_HPF_FBR.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.yscale("log")
P,X = probability_dist(HPF_IB)
X = X/np.std(X)
plt.plot(X,P,"-g",label="Blade Asymmetry")
P,X = probability_dist(HPF_MR)
X = X/np.std(X)
plt.plot(X,P,"-r",label="Out-of-plane bending moment")
P,X = probability_dist(HPF_FBR)
X = X/np.std(X)
plt.plot(X,P,"-b",label="Bearing force magnitude")
plt.ylabel("probability [-]")
plt.xlabel("Variable normalized on standard deviation")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"PDF_HPF_IB_MR_FBR_norm.png")
plt.close()