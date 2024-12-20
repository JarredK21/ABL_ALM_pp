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




FBR_analysis = False
time_scale_analysis = False
plot_all_FBR_times = False
peak_to_peak_analysis = False
peak_to_peak_directional_analysis = False
peak_to_peak_weighted_analysis = False
dF_dt_analysis = False
dtheta_dt_analysis = False
dF_F_analysis = False

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

#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

plt.plot(Time_OF,FBR)
plt.show()


LSSTipMys = np.array(OpenFAST_vars.variables["LSSTipMys"][Time_start_idx:])
LSSTipMzs = np.array(OpenFAST_vars.variables["LSSTipMzs"][Time_start_idx:])

LSShftFxa = np.array(OpenFAST_vars.variables["LSShftFxa"][Time_start_idx:])
LSShftFys = np.array(OpenFAST_vars.variables["LSShftFys"][Time_start_idx:])
LSShftFzs = np.array(OpenFAST_vars.variables["LSShftFzs"][Time_start_idx:])

#Asymmetry
Time_sampling = np.array(df_OF.variables["Time_sampling"])
dt_sampling = Time_sampling[1] - Time_sampling[0]
Time_start = 200
Time_sampling_start_idx = np.searchsorted(Time_sampling,Time_start)

Time_sampling = Time_sampling[Time_sampling_start_idx:]

Rotor_avg_vars = df_OF.groups["Rotor_Avg_Variables"]
Rotor_avg_vars_63 = Rotor_avg_vars.groups["63.0"]
Iy = np.array(Rotor_avg_vars_63.variables["Iy"][Time_sampling_start_idx:])
Iz = np.array(Rotor_avg_vars_63.variables["Iz"][Time_sampling_start_idx:])
I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

IA = np.array(Rotor_avg_vars_63.variables["IA"][Time_sampling_start_idx:])
LPF_IA = low_pass_filter(IA,0.3,dt_sampling)


#OOPBM
OOPBM = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

#Filtering FBR aero
LPF_OOPBM = low_pass_filter(OOPBM,0.1,dt)
LPF_1_OOPBM = low_pass_filter(OOPBM,0.3,dt)
LPF_2_OOPBM = low_pass_filter(OOPBM,0.9,dt)
LPF_3_OOPBM = low_pass_filter(OOPBM,1.5,dt)

# f = interpolate.interp1d(Time_OF,LPF_1_OOPBM)
# LPF_1_OOPBM_interp = f(Time_sampling)

# Time_shift_idx = np.searchsorted(Time_sampling,Time_sampling[0]+4.6)

# out_dir=in_dir+"three_frequency_analysis/OOPBM_analysis/"
# plt.rcParams['font.size'] = 16
# cc = round(correlation_coef(LPF_1_OOPBM_interp[Time_shift_idx:],LPF_IA[:-Time_shift_idx]),2)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF[:-Time_shift_idx],LPF_1_OOPBM[Time_shift_idx:],"-r")
# ax.set_ylabel("LPF Out-of-plane bending moment magnitude [kN-m]")
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(Time_sampling[:-Time_shift_idx],LPF_IA[:-Time_shift_idx],"-b")
# ax2.set_ylabel("LPF Asymmetry Parameter [$m^4/s$]")
# fig.supxlabel("Time [s]")
# fig.suptitle("correlation coefficient = {}".format(cc))
# plt.tight_layout()
# plt.savefig(out_dir+"cc_LPF_IA_OOPBM.png")
# plt.close()


HPF_OOPBM = np.subtract(OOPBM,LPF_3_OOPBM)
HPF_OOPBM = np.array(low_pass_filter(HPF_OOPBM,40,dt))
BPF_OOPBM = np.subtract(LPF_2_OOPBM,LPF_1_OOPBM)
dLPF_OOPBM = np.array(dt_calc(LPF_OOPBM,dt))
dBPF_OOPBM = np.array(dt_calc(BPF_OOPBM,dt))
dHPF_OOPBM = np.array(dt_calc(HPF_OOPBM,dt))

LPF_1_OOPBM = np.array(hard_filter(OOPBM,0.3,dt,"lowpass"))
LPF_2_OOPBM = np.array(hard_filter(OOPBM,0.9,dt,"lowpass"))
BPF_OOPBM = np.array(hard_filter(OOPBM,[0.3,0.9],dt,"bandpass"))
HPF_OOPBM = np.array(hard_filter(OOPBM,[1.5,40],dt,"bandpass"))


#BPF calc
dBPF_OOPBM = np.array(dt_calc(BPF_OOPBM,dt))
zero_crossings_index_BPF_OOPBM = np.where(np.diff(np.sign(dBPF_OOPBM)))[0]
Env_BPF_OOPBM = []
Env_Times = []
for i in np.arange(0,len(zero_crossings_index_BPF_OOPBM),2):
    idx = zero_crossings_index_BPF_OOPBM[i]
    Env_BPF_OOPBM.append(BPF_OOPBM[idx]); Env_Times.append(Time_OF[idx])

f = interpolate.interp1d(Env_Times,Env_BPF_OOPBM)
Env_Times = np.arange(Env_Times[0],Env_Times[-1],0.39)
Env_BPF_OOPBM = f(Env_Times)
dt_BPF = Env_Times[1] - Env_Times[0]
Env_BPF_OOPBM = hard_filter(Env_BPF_OOPBM,0.3,dt_BPF,"lowpass")

Env_BPF_OOPBM = np.array(Env_BPF_OOPBM); Env_Times = np.array(Env_Times)

f_BPF = interpolate.interp1d(Env_Times,Env_BPF_OOPBM)
f_LPF = interpolate.interp1d(Time_OF,LPF_1_OOPBM)


#HPF calc
abs_HPF_OOPBM = abs(HPF_OOPBM)
windows = [9]
for window in windows:
    window_idx = np.searchsorted(Time_OF,Time_OF[0]+window)
    if (window_idx % 2) != 0:
        window_idx+=1
    Times_avg_HPF = Time_OF[int(window_idx/2):-int(window_idx/2)]
    avg_HPF_OOPBM = []
    for i in np.arange(0,len(Time_OF)-window_idx):
        avg_HPF_OOPBM.append(np.average(abs_HPF_OOPBM[i:i+window_idx]))
    
    idx_min = np.searchsorted(Times_avg_HPF,np.min(Env_Times)); idx_max = np.searchsorted(Times_avg_HPF,np.max(Env_Times))
    Env_BPF_OOPBM_interp = f_BPF(Times_avg_HPF[idx_min:idx_max])
    cc_BPF = round(correlation_coef(Env_BPF_OOPBM_interp,avg_HPF_OOPBM[idx_min:idx_max]),3)

    LPF_1_OOPBM_interp = f_LPF(Times_avg_HPF)
    cc_LPF = round(correlation_coef(LPF_1_OOPBM_interp,avg_HPF_OOPBM),3)

dt_HPF = Times_avg_HPF[1] - Times_avg_HPF[0]
Times_avg_HPF = np.array(Times_avg_HPF); avg_HPF_OOPBM = np.array(hard_filter(avg_HPF_OOPBM,0.3,dt_HPF,"lowpass"))
# fig = plt.figure(figsize=(14,8))
# P,X = probability_dist(LPF_1_FBR)
# plt.plot(X,P,"-g",label="LPF")
# P,X = probability_dist(Env_BPF_FBR)
# plt.plot(X,P,"-r",label="Env BPF")
# P,X = probability_dist(avg_HPF_FBR)
# plt.plot(X,P,"-b",label="Filtered (9s) HPF")
# plt.grid()

plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/OOPBM_analysis/"

perc_overlap_LPF_HPF = []
perc_overlap_HPF_LPF = []
perc_overlap_BPF_HPF = []
perc_overlap_HPF_BPF = []
thresholds = np.linspace(0,1,5)
ix = 0
for threshold in thresholds:
    print(threshold)


    idx_min = np.searchsorted(Time_OF,np.min(Times_avg_HPF)); idx_max = np.searchsorted(Time_OF,np.max(Times_avg_HPF))
    xco_array_LPF = []
    xco = []
    for it in np.arange(idx_min,idx_max,dtype=int):
        if len(xco) == 0 and LPF_1_OOPBM[it] >= np.mean(LPF_1_OOPBM)+threshold*np.std(LPF_1_OOPBM):
            xco.append(Time_OF[it])
        
        if len(xco) == 1 and LPF_1_OOPBM[it] < np.mean(LPF_1_OOPBM)+threshold*np.std(LPF_1_OOPBM):
            xco.append(Time_OF[it])
            xco_array_LPF.append(xco)
            xco = []
        print(it)


    xco = []
    xco_array_BPF = []
    for it in np.arange(0,len(Env_Times)):
        if len(xco) == 0 and Env_BPF_OOPBM[it] >= np.mean(Env_BPF_OOPBM)+threshold*np.std(Env_BPF_OOPBM):
            xco.append(Env_Times[it])
        
        if len(xco) == 1 and Env_BPF_OOPBM[it] < np.mean(Env_BPF_OOPBM)+threshold*np.std(Env_BPF_OOPBM):
            xco.append(Env_Times[it])
            xco_array_BPF.append(xco)
            xco = []
        print(it)


    xco_array_HPF = []
    xco = []
    for it in np.arange(0,len(Times_avg_HPF)):
        if len(xco) == 0 and avg_HPF_OOPBM[it] >= np.mean(avg_HPF_OOPBM)+threshold*np.std(avg_HPF_OOPBM):
            xco.append(Times_avg_HPF[it])
        
        if len(xco) == 1 and avg_HPF_OOPBM[it] < np.mean(avg_HPF_OOPBM)+threshold*np.std(avg_HPF_OOPBM):
            xco.append(Times_avg_HPF[it])
            xco_array_HPF.append(xco)
            xco = []
        print(it)

    T_overlap_LPF_HPF = 0
    T_LPF = 0
    for xco_LPF in xco_array_LPF:
        T_LPF+=(xco_LPF[1]-xco_LPF[0])
        for xco_HPF in xco_array_HPF:
            if xco_HPF[0] <= xco_LPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_LPF[1] <= xco_HPF[1]:
                T_overlap_LPF_HPF+=(xco_LPF[1] - xco_LPF[0])
            elif xco_HPF[0] <= xco_LPF[0] <= xco_HPF[1]:
                T_overlap_LPF_HPF+=(xco_HPF[1] - xco_LPF[0])
            elif xco_HPF[0] <= xco_LPF[1] <= xco_HPF[1]:
                T_overlap_LPF_HPF+=(xco_LPF[1] - xco_HPF[0])
            elif xco_LPF[0] <= xco_HPF[0] <= xco_LPF[1] and xco_LPF[0] <= xco_HPF[1] <= xco_LPF[1]:
                T_overlap_LPF_HPF+=(xco_HPF[1] - xco_HPF[0])
    

    perc_overlap_LPF_HPF.append(round((T_overlap_LPF_HPF/T_LPF)*100,1))

    T_overlap_HPF_LPF = 0
    T_HPF = 0
    for xco_HPF in xco_array_HPF:
        T_HPF+=(xco_HPF[1]-xco_HPF[0])
        for xco_LPF in xco_array_LPF:
            if xco_LPF[0] <= xco_HPF[0] <= xco_LPF[1] and xco_LPF[0] <= xco_HPF[1] <= xco_LPF[1]:
                T_overlap_HPF_LPF+=(xco_HPF[1] - xco_HPF[0])
            elif xco_LPF[0] <= xco_HPF[0] <= xco_LPF[1]:
                T_overlap_HPF_LPF+=(xco_LPF[1] - xco_HPF[0])
            elif xco_LPF[0] <= xco_HPF[1] <= xco_LPF[1]:
                T_overlap_HPF_LPF+=(xco_HPF[1] - xco_LPF[0])
            elif xco_HPF[0] <= xco_LPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_LPF[1] <= xco_HPF[1]:
                T_overlap_HPF_LPF+=(xco_LPF[1] - xco_LPF[0])

    perc_overlap_HPF_LPF.append(round((T_overlap_HPF_LPF/T_HPF)*100,1))



    T_overlap_BPF_HPF = 0
    T_BPF = 0
    for xco_BPF in xco_array_BPF:
        T_BPF+=(xco_BPF[1]-xco_BPF[0])
        for xco_HPF in xco_array_HPF:
            if xco_HPF[0] <= xco_BPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_BPF[1] <= xco_HPF[1]:
                T_overlap_BPF_HPF+=(xco_BPF[1] - xco_BPF[0])
            elif xco_HPF[0] <= xco_BPF[0] <= xco_HPF[1]:
                T_overlap_BPF_HPF+=(xco_HPF[1] - xco_BPF[0])
            elif xco_HPF[0] <= xco_BPF[1] <= xco_HPF[1]:
                T_overlap_BPF_HPF+=(xco_BPF[1] - xco_HPF[0])
            elif xco_BPF[0] <= xco_HPF[0] <= xco_BPF[1] and xco_BPF[0] <= xco_HPF[1] <= xco_BPF[1]:
                T_overlap_BPF_HPF+=(xco_HPF[1] - xco_HPF[0])

    perc_overlap_BPF_HPF.append(round((T_overlap_BPF_HPF/T_BPF)*100,1))

    T_overlap_HPF_BPF = 0
    T_HPF = 0
    for xco_HPF in xco_array_HPF:
        T_HPF+=(xco_HPF[1]-xco_HPF[0])
        for xco_BPF in xco_array_BPF:
            if xco_BPF[0] <= xco_HPF[0] <= xco_BPF[1] and xco_BPF[0] <= xco_HPF[1] <= xco_BPF[1]:
                T_overlap_HPF_BPF+=(xco_HPF[1] - xco_HPF[0])
            elif xco_BPF[0] <= xco_HPF[0] <= xco_BPF[1]:
                T_overlap_HPF_BPF+=(xco_BPF[1] - xco_HPF[0])
            elif xco_BPF[0] <= xco_HPF[1] <= xco_BPF[1]:
                T_overlap_HPF_BPF+=(xco_HPF[1] - xco_BPF[0])
            elif xco_HPF[0] <= xco_BPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_BPF[1] <= xco_HPF[1]:
                T_overlap_HPF_BPF+=(xco_BPF[1] - xco_BPF[0])

    perc_overlap_HPF_BPF.append(round((T_overlap_HPF_BPF/T_HPF)*100,1))




    fig,ax = plt.subplots(figsize=(14,8))
    idx_min = np.searchsorted(Time_OF,np.min(Times_avg_HPF)); idx_max = np.searchsorted(Time_OF,np.max(Times_avg_HPF))
    ax.plot(Time_OF[idx_min:idx_max],LPF_1_OOPBM[idx_min:idx_max],"-g")
    ax.set_ylabel("LPF OOPBM magnitude [kN-m]")
    ax.grid()
    ax.axhline(y=np.mean(LPF_1_OOPBM)+threshold*np.std(LPF_1_OOPBM),linestyle="--",color="g")

    for xco in xco_array_LPF:
        square = patches.Rectangle((xco[0],np.min(LPF_1_OOPBM)), (xco[1]-xco[0]), (np.max(LPF_1_OOPBM)-np.min(LPF_1_OOPBM)), fill=True,color="g",alpha=0.1)
        ax.add_patch(square)


    ax2=ax.twinx()
    ax2.plot(Times_avg_HPF,avg_HPF_OOPBM,"-b")
    ax2.axhline(y=np.mean(avg_HPF_OOPBM)+threshold*np.std(avg_HPF_OOPBM),linestyle="--",color="b")

    for xco in xco_array_HPF:
        square = patches.Rectangle((xco[0],np.min(avg_HPF_OOPBM)), (xco[1]-xco[0]), (np.max(avg_HPF_OOPBM)-np.min(avg_HPF_OOPBM)), fill=True,color="b",alpha=0.1)
        ax2.add_patch(square)

    ax2.set_ylabel("Filtered (9s) HPF OOPBM magnitude [kN-m]")
    fig.supxlabel("Time [s]")
    fig.suptitle("Threshold T={}: mean(x)+T*std(x)\ncorrelation coefficient = {}\nPercentage overlap LPF to HPF = {}\nPercentage overlap HPF to LPF = {}".format(threshold,cc_LPF,perc_overlap_LPF_HPF[ix],perc_overlap_HPF_LPF[ix]))
    plt.tight_layout()
    plt.savefig(out_dir+"LPF_HPF_OOPBM_T_{}.png".format(threshold))
    plt.close(fig)


    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Env_Times,Env_BPF_OOPBM,"-r")
    ax.axhline(y=np.mean(Env_BPF_OOPBM)+threshold*np.std(Env_BPF_OOPBM),linestyle="--",color="r")

    for xco in xco_array_BPF:
        square = patches.Rectangle((xco[0],np.min(Env_BPF_OOPBM), ), (xco[1]-xco[0]), (np.max(Env_BPF_OOPBM)-np.min(Env_BPF_OOPBM)), fill=True,color="r",alpha=0.1)
        ax.add_patch(square)

    ax.set_ylabel("Envelope BPF OOPBM magnitude [kN-m]")
    ax.grid()
    ax2=ax.twinx()
    ax2.plot(Times_avg_HPF,avg_HPF_OOPBM,"-b")
    ax2.axhline(y=np.mean(avg_HPF_OOPBM)+threshold*np.std(avg_HPF_OOPBM),linestyle="--",color="b")

    for xco in xco_array_HPF:
        square = patches.Rectangle((xco[0],np.min(avg_HPF_OOPBM)), (xco[1]-xco[0]), (np.max(avg_HPF_OOPBM)-np.min(avg_HPF_OOPBM)), fill=True,color="b",alpha=0.1)
        ax2.add_patch(square)

    ax2.set_ylabel("Filtered (9s) HPF OOPBM magnitude [kN-m]")
    fig.supxlabel("Time [s]")
    fig.suptitle("Threshold T={}: mean(x)+T*std(x)\ncorrelation coefficient = {}\nPercentage overlap BPF to HPF = {}\nPercentage overlap HPF to BPF = {}".format(threshold,cc_BPF,perc_overlap_BPF_HPF[ix],perc_overlap_HPF_BPF[ix]))
    plt.tight_layout()
    plt.savefig(out_dir+"BPF_HPF_OOPBM_T_{}.png".format(threshold))
    plt.close(fig)

    ix+=1



plt.figure(figsize=(14,8))
plt.plot(thresholds,perc_overlap_LPF_HPF,"-og",label="LPF to HPF")
plt.plot(thresholds,perc_overlap_HPF_LPF,"-ob",label="HPF to LPF")
plt.plot(thresholds,perc_overlap_BPF_HPF,"-or",label="BPF to HPF")
plt.plot(thresholds,perc_overlap_HPF_BPF,"-om",label="HPF to BPF")
plt.xlabel("Threshold $T$: $mean(x)+T*std(x)$ [kN]")
plt.ylabel("Percentage overlap [%]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Threshold_percentage_overlap.png")
plt.close(fig)

# plt.rcParams['font.size'] = 16
# out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
# fig = plt.figure(figsize=(14,8))
# plt.yscale("log")
# P,X=probability_dist(OOPBM)
# plt.plot(X,P,"-k",label="Total")
# P,X=probability_dist(LPF_1_OOPBM)
# plt.plot(X,P,"-g",label="LPF")
# P,X=probability_dist(BPF_OOPBM)
# plt.plot(X,P,"-r",label="BPF")
# P,X=probability_dist(HPF_OOPBM)
# plt.plot(X,P,"-b",label="HPF")
# plt.xlabel("Out-of-plane bending moment magntiude [kN-m]")
# plt.ylabel("log() Probability [-]")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(out_dir+"PDF_MR.png")
# plt.close()

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

LPF_1_FBR = np.array(hard_filter(FBR,0.3,dt,"lowpass"))
LPF_2_FBR = np.array(hard_filter(FBR,0.9,dt,"lowpass"))
BPF_FBR = np.array(hard_filter(FBR,[0.3,0.9],dt,"bandpass"))
HPF_FBR = np.array(hard_filter(FBR,[1.5,40],dt,"bandpass"))

##calculate percentage overlap of high activity events in LPF, BPF and HPF FB signals
# #BPF calc
# dBPF_FBR = np.array(dt_calc(BPF_FBR,dt))
# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
# Env_BPF_FBR = []
# Env_Times = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR),2):
#     idx = zero_crossings_index_BPF_FBR[i]
#     Env_BPF_FBR.append(BPF_FBR[idx]); Env_Times.append(Time_OF[idx])

# f = interpolate.interp1d(Env_Times,Env_BPF_FBR)
# Env_Times = np.arange(Env_Times[0],Env_Times[-1],0.39)
# Env_BPF_FBR = f(Env_Times)
# dt_BPF = Env_Times[1] - Env_Times[0]
# Env_BPF_FBR = hard_filter(Env_BPF_FBR,0.3,dt_BPF,"lowpass")

# Env_BPF_FBR = np.array(Env_BPF_FBR); Env_Times = np.array(Env_Times)

# f_BPF = interpolate.interp1d(Env_Times,Env_BPF_FBR)
# f_LPF = interpolate.interp1d(Time_OF,LPF_1_FBR)


# #HPF calc
# abs_HPF_FBR = abs(HPF_FBR)
# windows = [9]
# for window in windows:
#     window_idx = np.searchsorted(Time_OF,Time_OF[0]+window)
#     if (window_idx % 2) != 0:
#         window_idx+=1
#     Times_avg_HPF = Time_OF[int(window_idx/2):-int(window_idx/2)]
#     avg_HPF_FBR = []
#     for i in np.arange(0,len(Time_OF)-window_idx):
#         avg_HPF_FBR.append(np.average(abs_HPF_FBR[i:i+window_idx]))
    
#     idx_min = np.searchsorted(Times_avg_HPF,np.min(Env_Times)); idx_max = np.searchsorted(Times_avg_HPF,np.max(Env_Times))
#     Env_BPF_FBR_interp = f_BPF(Times_avg_HPF[idx_min:idx_max])
#     cc_BPF = round(correlation_coef(Env_BPF_FBR_interp,avg_HPF_FBR[idx_min:idx_max]),3)

#     LPF_1_FBR_interp = f_LPF(Times_avg_HPF)
#     cc_LPF = round(correlation_coef(LPF_1_FBR_interp,avg_HPF_FBR),3)

# dt_HPF = Times_avg_HPF[1] - Times_avg_HPF[0]
# Times_avg_HPF = np.array(Times_avg_HPF); avg_HPF_FBR = np.array(hard_filter(avg_HPF_FBR,0.3,dt_HPF,"lowpass"))
# # fig = plt.figure(figsize=(14,8))
# # P,X = probability_dist(LPF_1_FBR)
# # plt.plot(X,P,"-g",label="LPF")
# # P,X = probability_dist(Env_BPF_FBR)
# # plt.plot(X,P,"-r",label="Env BPF")
# # P,X = probability_dist(avg_HPF_FBR)
# # plt.plot(X,P,"-b",label="Filtered (9s) HPF")
# # plt.grid()

# plt.rcParams['font.size'] = 16
# out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"

# perc_overlap_LPF_HPF = []
# perc_overlap_HPF_LPF = []
# perc_overlap_BPF_HPF = []
# perc_overlap_HPF_BPF = []
# thresholds = np.linspace(0,1,5)
# ix = 0
# for threshold in thresholds:
#     print(threshold)


#     idx_min = np.searchsorted(Time_OF,np.min(Times_avg_HPF)); idx_max = np.searchsorted(Time_OF,np.max(Times_avg_HPF))
#     xco_array_LPF = []
#     xco = []
#     for it in np.arange(idx_min,idx_max,dtype=int):
#         if len(xco) == 0 and LPF_1_FBR[it] >= np.mean(LPF_1_FBR)+threshold*np.std(LPF_1_FBR):
#             xco.append(Time_OF[it])
        
#         if len(xco) == 1 and LPF_1_FBR[it] < np.mean(LPF_1_FBR)+threshold*np.std(LPF_1_FBR):
#             xco.append(Time_OF[it])
#             xco_array_LPF.append(xco)
#             xco = []
#         print(it)


#     xco = []
#     xco_array_BPF = []
#     for it in np.arange(0,len(Env_Times)):
#         if len(xco) == 0 and Env_BPF_FBR[it] >= np.mean(Env_BPF_FBR)+threshold*np.std(Env_BPF_FBR):
#             xco.append(Env_Times[it])
        
#         if len(xco) == 1 and Env_BPF_FBR[it] < np.mean(Env_BPF_FBR)+threshold*np.std(Env_BPF_FBR):
#             xco.append(Env_Times[it])
#             xco_array_BPF.append(xco)
#             xco = []
#         print(it)


#     xco_array_HPF = []
#     xco = []
#     for it in np.arange(0,len(Times_avg_HPF)):
#         if len(xco) == 0 and avg_HPF_FBR[it] >= np.mean(avg_HPF_FBR)+threshold*np.std(avg_HPF_FBR):
#             xco.append(Times_avg_HPF[it])
        
#         if len(xco) == 1 and avg_HPF_FBR[it] < np.mean(avg_HPF_FBR)+threshold*np.std(avg_HPF_FBR):
#             xco.append(Times_avg_HPF[it])
#             xco_array_HPF.append(xco)
#             xco = []
#         print(it)

#     T_overlap_LPF_HPF = 0
#     T_LPF = 0
#     for xco_LPF in xco_array_LPF:
#         T_LPF+=(xco_LPF[1]-xco_LPF[0])
#         for xco_HPF in xco_array_HPF:
#             if xco_HPF[0] <= xco_LPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_LPF[1] <= xco_HPF[1]:
#                 T_overlap_LPF_HPF+=(xco_LPF[1] - xco_LPF[0])
#             elif xco_HPF[0] <= xco_LPF[0] <= xco_HPF[1]:
#                 T_overlap_LPF_HPF+=(xco_HPF[1] - xco_LPF[0])
#             elif xco_HPF[0] <= xco_LPF[1] <= xco_HPF[1]:
#                 T_overlap_LPF_HPF+=(xco_LPF[1] - xco_HPF[0])
#             elif xco_LPF[0] <= xco_HPF[0] <= xco_LPF[1] and xco_LPF[0] <= xco_HPF[1] <= xco_LPF[1]:
#                 T_overlap_LPF_HPF+=(xco_HPF[1] - xco_HPF[0])
    

#     perc_overlap_LPF_HPF.append(round((T_overlap_LPF_HPF/T_LPF)*100,1))

#     T_overlap_HPF_LPF = 0
#     T_HPF = 0
#     for xco_HPF in xco_array_HPF:
#         T_HPF+=(xco_HPF[1]-xco_HPF[0])
#         for xco_LPF in xco_array_LPF:
#             if xco_LPF[0] <= xco_HPF[0] <= xco_LPF[1] and xco_LPF[0] <= xco_HPF[1] <= xco_LPF[1]:
#                 T_overlap_HPF_LPF+=(xco_HPF[1] - xco_HPF[0])
#             elif xco_LPF[0] <= xco_HPF[0] <= xco_LPF[1]:
#                 T_overlap_HPF_LPF+=(xco_LPF[1] - xco_HPF[0])
#             elif xco_LPF[0] <= xco_HPF[1] <= xco_LPF[1]:
#                 T_overlap_HPF_LPF+=(xco_HPF[1] - xco_LPF[0])
#             elif xco_HPF[0] <= xco_LPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_LPF[1] <= xco_HPF[1]:
#                 T_overlap_HPF_LPF+=(xco_LPF[1] - xco_LPF[0])

#     perc_overlap_HPF_LPF.append(round((T_overlap_HPF_LPF/T_HPF)*100,1))



#     T_overlap_BPF_HPF = 0
#     T_BPF = 0
#     for xco_BPF in xco_array_BPF:
#         T_BPF+=(xco_BPF[1]-xco_BPF[0])
#         for xco_HPF in xco_array_HPF:
#             if xco_HPF[0] <= xco_BPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_BPF[1] <= xco_HPF[1]:
#                 T_overlap_BPF_HPF+=(xco_BPF[1] - xco_BPF[0])
#             elif xco_HPF[0] <= xco_BPF[0] <= xco_HPF[1]:
#                 T_overlap_BPF_HPF+=(xco_HPF[1] - xco_BPF[0])
#             elif xco_HPF[0] <= xco_BPF[1] <= xco_HPF[1]:
#                 T_overlap_BPF_HPF+=(xco_BPF[1] - xco_HPF[0])
#             elif xco_BPF[0] <= xco_HPF[0] <= xco_BPF[1] and xco_BPF[0] <= xco_HPF[1] <= xco_BPF[1]:
#                 T_overlap_BPF_HPF+=(xco_HPF[1] - xco_HPF[0])

#     perc_overlap_BPF_HPF.append(round((T_overlap_BPF_HPF/T_BPF)*100,1))

#     T_overlap_HPF_BPF = 0
#     T_HPF = 0
#     for xco_HPF in xco_array_HPF:
#         T_HPF+=(xco_HPF[1]-xco_HPF[0])
#         for xco_BPF in xco_array_BPF:
#             if xco_BPF[0] <= xco_HPF[0] <= xco_BPF[1] and xco_BPF[0] <= xco_HPF[1] <= xco_BPF[1]:
#                 T_overlap_HPF_BPF+=(xco_HPF[1] - xco_HPF[0])
#             elif xco_BPF[0] <= xco_HPF[0] <= xco_BPF[1]:
#                 T_overlap_HPF_BPF+=(xco_BPF[1] - xco_HPF[0])
#             elif xco_BPF[0] <= xco_HPF[1] <= xco_BPF[1]:
#                 T_overlap_HPF_BPF+=(xco_HPF[1] - xco_BPF[0])
#             elif xco_HPF[0] <= xco_BPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_BPF[1] <= xco_HPF[1]:
#                 T_overlap_HPF_BPF+=(xco_BPF[1] - xco_BPF[0])

#     perc_overlap_HPF_BPF.append(round((T_overlap_HPF_BPF/T_HPF)*100,1))




#     fig,ax = plt.subplots(figsize=(14,8))
#     idx_min = np.searchsorted(Time_OF,np.min(Times_avg_HPF)); idx_max = np.searchsorted(Time_OF,np.max(Times_avg_HPF))
#     ax.plot(Time_OF[idx_min:idx_max],LPF_1_FBR[idx_min:idx_max],"-g")
#     ax.set_ylabel("LPF Bearing force magnitude [kN]")
#     ax.grid()
#     ax.axhline(y=np.mean(LPF_1_FBR)+threshold*np.std(LPF_1_FBR),linestyle="--",color="g")

#     for xco in xco_array_LPF:
#         square = patches.Rectangle((xco[0],np.min(LPF_1_FBR)), (xco[1]-xco[0]), (np.max(LPF_1_FBR)-np.min(LPF_1_FBR)), fill=True,color="g",alpha=0.1)
#         ax.add_patch(square)


#     ax2=ax.twinx()
#     ax2.plot(Times_avg_HPF,avg_HPF_FBR,"-b")
#     ax2.axhline(y=np.mean(avg_HPF_FBR)+threshold*np.std(avg_HPF_FBR),linestyle="--",color="b")

#     for xco in xco_array_HPF:
#         square = patches.Rectangle((xco[0],np.min(avg_HPF_FBR)), (xco[1]-xco[0]), (np.max(avg_HPF_FBR)-np.min(avg_HPF_FBR)), fill=True,color="b",alpha=0.1)
#         ax2.add_patch(square)

#     ax2.set_ylabel("Filtered (9s) HPF Bearing force magnitude [kN]")
#     fig.supxlabel("Time [s]")
#     fig.suptitle("Threshold T={}: mean(x)+T*std(x)\ncorrelation coefficient = {}\nPercentage overlap LPF to HPF = {}\nPercentage overlap HPF to LPF = {}".format(threshold,cc_LPF,perc_overlap_LPF_HPF[ix],perc_overlap_HPF_LPF[ix]))
#     plt.tight_layout()
#     plt.savefig(out_dir+"LPF_HPF_FB_T_{}.png".format(threshold))
#     plt.close(fig)


#     fig,ax = plt.subplots(figsize=(14,8))
#     ax.plot(Env_Times,Env_BPF_FBR,"-r")
#     ax.axhline(y=np.mean(Env_BPF_FBR)+threshold*np.std(Env_BPF_FBR),linestyle="--",color="r")

#     for xco in xco_array_BPF:
#         square = patches.Rectangle((xco[0],np.min(Env_BPF_FBR), ), (xco[1]-xco[0]), (np.max(Env_BPF_FBR)-np.min(Env_BPF_FBR)), fill=True,color="r",alpha=0.1)
#         ax.add_patch(square)

#     ax.set_ylabel("Envelope BPF Bearing force magnitude [kN]")
#     ax.grid()
#     ax2=ax.twinx()
#     ax2.plot(Times_avg_HPF,avg_HPF_FBR,"-b")
#     ax2.axhline(y=np.mean(avg_HPF_FBR)+threshold*np.std(avg_HPF_FBR),linestyle="--",color="b")

#     for xco in xco_array_HPF:
#         square = patches.Rectangle((xco[0],np.min(avg_HPF_FBR)), (xco[1]-xco[0]), (np.max(avg_HPF_FBR)-np.min(avg_HPF_FBR)), fill=True,color="b",alpha=0.1)
#         ax2.add_patch(square)

#     ax2.set_ylabel("Filtered (9s) HPF Bearing force magnitude [kN]")
#     fig.supxlabel("Time [s]")
#     fig.suptitle("Threshold T={}: mean(x)+T*std(x)\ncorrelation coefficient = {}\nPercentage overlap BPF to HPF = {}\nPercentage overlap HPF to BPF = {}".format(threshold,cc_BPF,perc_overlap_BPF_HPF[ix],perc_overlap_HPF_BPF[ix]))
#     plt.tight_layout()
#     plt.savefig(out_dir+"BPF_HPF_FB_T_{}.png".format(threshold))
#     plt.close(fig)

#     ix+=1



# plt.figure(figsize=(14,8))
# plt.plot(thresholds,perc_overlap_LPF_HPF,"-og",label="LPF to HPF")
# plt.plot(thresholds,perc_overlap_HPF_LPF,"-ob",label="HPF to LPF")
# plt.plot(thresholds,perc_overlap_BPF_HPF,"-or",label="BPF to HPF")
# plt.plot(thresholds,perc_overlap_HPF_BPF,"-om",label="HPF to BPF")
# plt.xlabel("Threshold $T$: $mean(x)+T*std(x)$ [kN]")
# plt.ylabel("Percentage overlap [%]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Threshold_percentage_overlap.png")
# plt.close(fig)






# #BPF calc
# dBPF_FBR = np.array(dt_calc(BPF_FBR,dt))
# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
# Env_BPF_FBR = []
# Env_Times = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR),2):
#     idx = zero_crossings_index_BPF_FBR[i]
#     Env_BPF_FBR.append(BPF_FBR[idx]); Env_Times.append(Time_OF[idx])

# f_BPF = interpolate.interp1d(Env_Times,Env_BPF_FBR)
# f_LPF = interpolate.interp1d(Time_OF,LPF_1_FBR)

# plt.rcParams['font.size'] = 16
# out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,abs(HPF_FBR),"-k")
# shift = 0
# #HPF calc
# abs_HPF_FBR = abs(HPF_FBR)
# windows = [1,5,9]
# for window in windows:
#     window_idx = np.searchsorted(Time_OF,Time_OF[0]+window)
#     if (window_idx % 2) != 0:
#         window_idx+=1
#     Times_avg_HPF = Time_OF[int(window_idx/2):-int(window_idx/2)]
#     avg_HPF_FBR = []
#     for i in np.arange(0,len(Time_OF)-window_idx):
#         avg_HPF_FBR.append(np.average(abs_HPF_FBR[i:i+window_idx]))
    
#     idx_min = np.searchsorted(Times_avg_HPF,np.min(Env_Times)); idx_max = np.searchsorted(Times_avg_HPF,np.max(Env_Times))
#     Env_BPF_FBR_interp = f_BPF(Times_avg_HPF[idx_min:idx_max])
#     cc_BPF = round(correlation_coef(Env_BPF_FBR_interp,avg_HPF_FBR[idx_min:idx_max]),3)

#     LPF_1_FBR_interp = f_LPF(Times_avg_HPF)
#     cc_LPF = round(correlation_coef(LPF_1_FBR_interp,avg_HPF_FBR),3)

#     plt.plot(Times_avg_HPF,np.add(avg_HPF_FBR,shift),label="window = {}s".format(window)+"\n$cc_{LPF}$"+"= {}".format(cc_LPF)+"\n$cc_{BPF}$"+"= {}".format(cc_BPF))

#     shift+=100


# plt.xlabel("Time [s]")
# plt.ylabel("HPF Bearing force magnitude [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"avg_HPF_FB.png")
# plt.close(fig)


#plotting cc against window size for LPF and env BPF with effectively filtered HPF
# #BPF calc
# dBPF_FBR = np.array(dt_calc(BPF_FBR,dt))
# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
# Env_BPF_FBR = []
# Env_Times = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR),2):
#     idx = zero_crossings_index_BPF_FBR[i]
#     Env_BPF_FBR.append(BPF_FBR[idx]); Env_Times.append(Time_OF[idx])

# f_BPF = interpolate.interp1d(Env_Times,Env_BPF_FBR)
# f_LPF = interpolate.interp1d(Time_OF,LPF_1_FBR)
# #HPF calc
# abs_HPF_FBR = abs(HPF_FBR)
# windows = np.arange(1,13,1)
# cc_BPF = []
# cc_LPF = []
# for window in windows:
#     window_idx = np.searchsorted(Time_OF,Time_OF[0]+window)
#     if (window_idx % 2) != 0:
#         window_idx+=1
#     Times_avg_HPF = Time_OF[int(window_idx/2):-int(window_idx/2)]
#     avg_HPF_FBR = []
#     for i in np.arange(0,len(Time_OF)-window_idx):
#         avg_HPF_FBR.append(np.average(abs_HPF_FBR[i:i+window_idx]))
    
#     idx_min = np.searchsorted(Times_avg_HPF,np.min(Env_Times)); idx_max = np.searchsorted(Times_avg_HPF,np.max(Env_Times))
#     Env_BPF_FBR_interp = f_BPF(Times_avg_HPF[idx_min:idx_max])
#     cc_BPF.append(correlation_coef(Env_BPF_FBR_interp,avg_HPF_FBR[idx_min:idx_max]))

#     LPF_1_FBR_interp = f_LPF(Times_avg_HPF)
#     cc_LPF.append(correlation_coef(LPF_1_FBR_interp,avg_HPF_FBR))

# plt.rcParams['font.size'] = 16
# out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
# fig = plt.figure(figsize=(14,8))
# plt.plot(windows,cc_BPF,"-r",label="(BPF $F_B$, avg(|$F_{B,HPF}$|)")
# plt.plot(windows,cc_LPF,"-b",label="(LPF $F_B$, avg(|$F_{B,HPF}$|)")
# plt.xlabel("Window size [s]")
# plt.ylabel("Correlation coefficient")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(out_dir+"cc_window_HPF_FB.png")
# plt.close(fig)





#Quantifying time period PDF's for LPF, BPF and HPF signals representing the overall envelopes of the respective signals
# #LPF calc
# LPF_FBR = hard_filter(FBR,0.1,dt,"lowpass")

# dLPF_FBR = np.array(dt_calc(LPF_FBR,dt))
# zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_FBR)))[0]

# dt_LPF = []
# Time_mag = []
# FBR_LPF = []
# for i in np.arange(0,len(zero_crossings_index_LPF_FBR)-2,2):
#     it_1 = zero_crossings_index_LPF_FBR[i]
#     it_2 = zero_crossings_index_LPF_FBR[i+1]
#     dt_LPF.append(Time_OF[it_2]-Time_OF[it_1])
#     Time_mag.append(Time_OF[it_1])
#     FBR_LPF.append(LPF_FBR[it_1])

# #BPF calc
# dBPF_FBR = np.array(dt_calc(BPF_FBR,dt))
# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
# Env_BPF_FBR = []
# Env_Times = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR),2):
#     idx = zero_crossings_index_BPF_FBR[i]
#     Env_BPF_FBR.append(BPF_FBR[idx]); Env_Times.append(Time_OF[idx])


# f = interpolate.interp1d(Env_Times,Env_BPF_FBR)
# Env_Times = np.linspace(Env_Times[0],Env_Times[-1],len(Time_OF))
# dt_Env = Env_Times[1] - Env_Times[0]
# Env_BPF_FBR = f(Env_Times)
# Env_BPF_FBR_LPF = hard_filter(Env_BPF_FBR,0.1,dt_Env,"lowpass")

# dBPF_FBR = np.array(dt_calc(Env_BPF_FBR_LPF,dt_Env))
# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
# dt_BPF = []
# Time_mag = []
# FBR_BPF = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-2,2):
#     it_1 = zero_crossings_index_BPF_FBR[i]
#     it_2 = zero_crossings_index_BPF_FBR[i+1]
#     dt_BPF.append(Env_Times[it_2]-Env_Times[it_1])
#     Time_mag.append(Env_Times[it_1])
#     FBR_BPF.append(Env_BPF_FBR_LPF[it_1])

# #HPF calc
# abs_HPF_FBR = abs(HPF_FBR)
# windows = [7]
# for window in windows:
#     window_idx = np.searchsorted(Time_OF,Time_OF[0]+window)
#     if (window_idx % 2) != 0:
#         window_idx+=1
#     Times_avg_HPF = Time_OF[int(window_idx/2):-int(window_idx/2)]
#     avg_HPF_FBR = []
#     for i in np.arange(0,len(Time_OF)-window_idx):
#         avg_HPF_FBR.append(np.average(abs_HPF_FBR[i:i+window_idx]))

# avg_HPF_FBR_LPF = hard_filter(avg_HPF_FBR,0.1,dt,"lowpass")


# dAvg_HPF_FBR = np.array(dt_calc(avg_HPF_FBR_LPF,dt))

# zero_crossings_index_Avg_HPF_FBR = np.where(np.diff(np.sign(dAvg_HPF_FBR)))[0]

# dt_HPF = []
# Time_mag = []
# FBR_HPF = []
# for i in np.arange(0,len(zero_crossings_index_Avg_HPF_FBR)-2,2):
#     it_1 = zero_crossings_index_Avg_HPF_FBR[i]
#     it_2 = zero_crossings_index_Avg_HPF_FBR[i+1]
#     dt_HPF.append(Times_avg_HPF[it_2]-Times_avg_HPF[it_1])
#     Time_mag.append(Times_avg_HPF[it_1])
#     FBR_HPF.append(avg_HPF_FBR_LPF[it_1])

# plt.rcParams['font.size'] = 16
# out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
# fig = plt.figure(figsize=(14,8))
# P,X = probability_dist(dt_LPF)
# plt.plot(X,P,"-g",label="LPF")
# P,X = probability_dist(dt_BPF)
# plt.plot(X,P,"-r",label="Envelope BPF")
# P,X = probability_dist(dt_HPF)
# plt.plot(X,P,"-b",label="Filtered HPF")
# plt.xlabel("dt [s]")
# plt.ylabel("Probability [-]")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(out_dir+"PDF_dt_LPF_Env_BPF_avg_HPF.png")
# plt.close()



# mean_LPF = []
# thresholds = np.linspace(np.mean(LPF_1_FBR),np.mean(LPF_1_FBR)+2*np.std(LPF_1_FBR),5)
# thresholds = [np.mean(LPF_1_FBR)]
# for threshold in thresholds:
#     LPF_1_FBR_threshold = []
#     for it in np.arange(0,len(LPF_1_FBR)):
#         if LPF_1_FBR[it] >= threshold:
#             LPF_1_FBR_threshold.append(LPF_1_FBR[it])
#         else:
#             LPF_1_FBR_threshold.append(0.0)



#     dLPF_FBR = np.array(dt_calc(LPF_1_FBR_threshold,dt))

#     zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_FBR)))[0]

#     dt_LPF = []
#     FBR_LPF = []
#     Time_mag  = []
#     for i in np.arange(0,len(zero_crossings_index_LPF_FBR)-1,1):
#         it_1 = zero_crossings_index_LPF_FBR[i]
#         it_2 = zero_crossings_index_LPF_FBR[i+1]
#         dt_LPF.append(Time_OF[it_2]-Time_OF[it_1])

#         Time_mag.append(Time_OF[it_1])
#         FBR_LPF.append(LPF_1_FBR_threshold[it_1])

    
#     mean_LPF.append(np.mean(dt_LPF))
#     fig = plt.figure(figsize=(14,8))
#     plt.plot(Time_OF,LPF_1_FBR,"-g")
#     plt.plot(Time_OF,LPF_1_FBR_threshold,"-r")
#     plt.scatter(Time_mag,FBR_LPF)
#     plt.show()

# fig = plt.figure(figsize=(14,8))
# plt.plot(thresholds,mean_LPF)
# plt.show()

# Env_BPF_FBR = []
# Env_Times = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR),2):
#     idx = zero_crossings_index_BPF_FBR[i]
#     Env_BPF_FBR.append(BPF_FBR[idx]); Env_Times.append(Time_OF[idx])

# plt.rcParams['font.size'] = 16
# out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
# f = interpolate.interp1d(Env_Times,Env_BPF_FBR)
# Env_Times = np.linspace(Env_Times[0],Env_Times[-1],len(Time_OF))
# dt_Env = Env_Times[1] - Env_Times[0]
# Env_BPF_FBR = f(Env_Times)
# Env_BPF_FBR_LPF = low_pass_filter(Env_BPF_FBR,0.3,dt_Env)
# f = interpolate.interp1d(Time_OF,LPF_1_FBR)
# LPF_1_FBR_interp = f(Env_Times)
# cc = round(correlation_coef(LPF_1_FBR_interp,Env_BPF_FBR_LPF),2)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Env_Times,Env_BPF_FBR_LPF,"-r")
# ax.set_ylabel("LPF Envelope BPF Bearing force [kN]")
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(Time_OF,LPF_1_FBR,"-g")
# ax2.set_ylabel("LPF Bearing force [kN]")
# fig.supxlabel("Time [s]")
# fig.suptitle("correlation coefficient = {}".format(cc))
# plt.tight_layout()
# plt.savefig(out_dir+"Env_BPF_LPF_FB.png")
# plt.close()

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,BPF_FBR,"-r",label="BPF")
# plt.plot(Env_Times,Env_BPF_FBR_LPF,"-b",label="LPF Envelope BPF")
# plt.grid()
# plt.xlabel("Time [s]")
# plt.ylabel("BPF Bearing force [kN]")
# plt.legend()
# plt.tight_layout()
# plt.savefig(out_dir+"Env_BPF.png")
# plt.close()

# plt.rcParams['font.size'] = 16
# out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
# f = interpolate.interp1d(Time_OF,LPF_1_FBR)
# abs_HPF_FBR = abs(HPF_FBR)
# ccs = []
# windows = [7]
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF,LPF_1_FBR,"-g")
# ax.axhline(y=np.mean(LPF_1_FBR)+2*np.std(LPF_1_FBR),linestyle="--",color="g")
# ax.set_ylabel("LPF Bearing force [kN]")
# ax2=ax.twinx()
# addition = 0
# for window in windows:
#     window_idx = np.searchsorted(Time_OF,Time_OF[0]+window)
#     if (window_idx % 2) != 0:
#         window_idx+=1
#     avg_HPF_FBR = []
#     for i in np.arange(0,len(Time_OF)-window_idx):
#         avg_HPF_FBR.append(np.average(abs_HPF_FBR[i:i+window_idx]))

#     LPF_1_FBR_interp = f(Time_OF[int(window_idx/2):-int(window_idx/2)])
#     cc = round(correlation_coef(LPF_1_FBR_interp,avg_HPF_FBR),2)
#     ccs.append(cc)
#     addition += 100
# avg_HPF_FBR_LPF = low_pass_filter(avg_HPF_FBR,0.3,dt)
# cc = round(correlation_coef(LPF_1_FBR_interp,avg_HPF_FBR_LPF),2)
# ax2.plot(Time_OF[int(window_idx/2):-int(window_idx/2)],avg_HPF_FBR_LPF,"-b")
# ax2.axhline(y=np.mean(avg_HPF_FBR)+2*np.std(avg_HPF_FBR),linestyle="--",color="b")
# ax2.set_ylabel("LPF Averaged over T=7s, absolute HPF Bearing force [kN]")
# fig.supxlabel("Time [s]")
# fig.suptitle("correlation coefficient = {}".format(cc))
# ax.grid()
# #plt.legend()
# plt.tight_layout()
# plt.savefig(out_dir+"avg_HPF_LPF_FB.png")
# plt.close()

# Times_avg_HPF = Time_OF[int(window_idx/2):-int(window_idx/2)]

# fig = plt.figure(figsize=(14,8))
# plt.yscale("log")
# P,X = probability_dist(LPF_1_FBR)
# plt.plot(X,P,"-g",label="LPF")
# P,X = probability_dist(avg_HPF_FBR)
# plt.plot(X,P,"-b",label="Avg abs HPF")
# plt.xlabel("Bearing force [kN]")
# plt.ylabel("log() probability [-]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"PDF_avg_HPF_LPF_FB.png")
# plt.close()

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,HPF_FBR,"-b",label="HPF")
# plt.plot(Time_OF[int(window_idx/2):-int(window_idx/2)],avg_HPF_FBR_LPF,"-r",label="Averaged HPF T = 7s")
# plt.xlabel("TIme [s]")
# plt.ylabel("HPF Bearing force [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Avg_HPF.png")
# plt.close()

# fig = plt.figure(figsize=(14,8))
# plt.plot(windows,ccs)
# plt.xlabel("Window size [s]")
# plt.ylabel("correlation coefficient (LPF $F_B$,avg(|$F_{B,HPF}$|)")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"cc_window_LPF_HPF_FB.png")
# plt.close()


# dLPF_FBR = np.array(dt_calc(LPF_1_FBR,dt))

# zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_FBR)))[0]

# DT_LPF = []
# for i in np.arange(0,len(zero_crossings_index_LPF_FBR)-2,2):
#     it_1 = zero_crossings_index_LPF_FBR[i]
#     it_2 = zero_crossings_index_LPF_FBR[i+1]
#     DT_LPF.append(Time_OF[it_2]-Time_OF[it_1])

# dEnv_BPF_FBR = np.array(dt_calc(Env_BPF_FBR_LPF,dt_Env))

# zero_crossings_index_Env_BPF_FBR = np.where(np.diff(np.sign(dEnv_BPF_FBR)))[0]

# DT_BPF = []
# for i in np.arange(0,len(zero_crossings_index_Env_BPF_FBR)-2,2):
#     it_1 = zero_crossings_index_Env_BPF_FBR[i]
#     it_2 = zero_crossings_index_Env_BPF_FBR[i+1]
#     DT_BPF.append(Time_OF[it_2]-Time_OF[it_1])

# dAvg_HPF_FBR = np.array(dt_calc(avg_HPF_FBR_LPF,dt))

# zero_crossings_index_Avg_HPF_FBR = np.where(np.diff(np.sign(dAvg_HPF_FBR)))[0]

# DT_HPF = []
# for i in np.arange(1,len(zero_crossings_index_Avg_HPF_FBR)-2,2):
#     it_1 = zero_crossings_index_Avg_HPF_FBR[i]
#     it_2 = zero_crossings_index_Avg_HPF_FBR[i+1]
#     DT_HPF.append(Times_avg_HPF[it_2]-Times_avg_HPF[it_1])


# fig = plt.figure(figsize=(14,8))
# P,X = probability_dist(DT_LPF)
# plt.plot(X,P,"-g",label="LPF")
# P,X = probability_dist(DT_BPF)
# plt.plot(X,P,"-r",label="Env BPF")
# P,X = probability_dist(DT_HPF)
# plt.plot(X,P,"-r",label="Avg HPF")
# plt.legend()
# plt.xlabel("dt [s]")
# plt.ylabel("Probability [-]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"PDF_dt_LPF_Env_BPF_avg_HPF.png")
# plt.close()


BPF_FBR = np.subtract(LPF_2_FBR,LPF_1_FBR)

FBR = low_pass_filter(FBR,40,dt)
dFBR = np.array(dt_calc(FBR,dt))
dBPF_FBR = np.array(dt_calc(BPF_FBR,dt))
dHPF_FBR = np.array(dt_calc(HPF_FBR,dt))
dLPF_2_FBR = np.array(dt_calc(LPF_2_FBR,dt))

zero_crossings_index_FBR = np.where(np.diff(np.sign(dFBR)))[0]
zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_2_FBR)))[0]
zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]


#FBR calc
dF_mag = []
for i in np.arange(0,len(zero_crossings_index_FBR)-1):

    it_1 = zero_crossings_index_FBR[i]
    it_2 = zero_crossings_index_FBR[i+1]

    dF_mag.append(FBR[it_2] - FBR[it_1])

threshold = 2*np.std(dF_mag)


#FBR calc
Time_mag = []
dF_FBR_mag = []
dF_BPF_mag = []
dF_HPF_mag = []
for i in np.arange(0,len(zero_crossings_index_FBR)-1):

    it_1 = zero_crossings_index_FBR[i]
    it_2 = zero_crossings_index_FBR[i+1]

    if abs(FBR[it_2]-FBR[it_1]) >= threshold:
        Time_mag.append(Time_OF[it_1])
        dF_FBR_mag.append(abs(FBR[it_2]-FBR[it_1]))
        dF_BPF_mag.append(abs(BPF_FBR[it_2]-BPF_FBR[it_1]))
        dF_HPF_mag.append(abs(HPF_FBR[it_2]-HPF_FBR[it_1]))

plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(14,8),sharex=True)
ax1.scatter(Time_mag,dF_FBR_mag)
ax1.grid()
ax1.set_title("$dF$ $F_{B,tot}$ threshold on $2\sigma$ $F_{B,tot}$ [kN]")
ax1_2 = ax1.twiny()
P,X=probability_dist(dF_FBR_mag)
print(moments(dF_FBR_mag))
ax1_2.plot(P,X,"-k")
ax1_2.axvline(x=0.0,color="k")
ax2.scatter(Time_mag,dF_BPF_mag)
ax2.grid()
ax2.set_title("$dF$ $F_{B,BPF}$ threshold on $2\sigma$ $F_{B,tot}$ [kN]")
ax2_2 = ax2.twiny()
P,X=probability_dist(dF_BPF_mag)
print(moments(dF_BPF_mag))
ax2_2.plot(P,X,"-k")
ax2_2.axvline(x=0.0,color="k")
ax3.scatter(Time_mag,dF_HPF_mag)
ax3.grid()
ax3.set_title("$dF$ $F_{B,HPF}$ threshold on $2\sigma$ $F_{B,tot}$ [kN]")
ax3_2 = ax3.twiny()
P,X=probability_dist(dF_HPF_mag)
print(moments(dF_HPF_mag))
ax3_2.plot(P,X,"-k")
ax3_2.axvline(x=0.0,color="k")
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"dF_FBR_all_freqs.png")
plt.close()



#FBR calc
Time_mag = []
dF_diff = []
dF_ratio = []
Time_mag_threshold = []
dF_diff_threshold = []
dF_ratio_threshold = []
for i in np.arange(0,len(zero_crossings_index_FBR)-1):

    it_1 = zero_crossings_index_FBR[i]
    it_2 = zero_crossings_index_FBR[i+1]

   
    Time_mag.append(Time_OF[it_1])
    dF_diff.append((abs(FBR[it_2] - FBR[it_1]) - abs(LPF_2_FBR[it_2] - LPF_2_FBR[it_1])))
    #dF_ratio.append((FBR[it_2] - FBR[it_1])/(LPF_2_FBR[it_2] - LPF_2_FBR[it_1]))

    if abs(FBR[it_2] - FBR[it_1])>=threshold:
        Time_mag_threshold.append(Time_OF[it_1])
        dF_diff_threshold.append((abs(FBR[it_2] - FBR[it_1]) - abs(LPF_2_FBR[it_2] - LPF_2_FBR[it_1])))
        #dF_ratio_threshold.append((FBR[it_2] - FBR[it_1])/(LPF_2_FBR[it_2] - LPF_2_FBR[it_1]))



plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
fig = plt.figure(figsize=(14,8))
plt.scatter(Time_mag,dF_diff)
plt.xlabel("Time [s]")
plt.ylabel("$(|dF_{total}| - |dF_{LPF+BPF}|)$")
plt.grid()
plt.savefig(out_dir+"dF_diff.png")
plt.close()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(dF_diff)
plt.yscale("log")
plt.plot(X,P,"-k")
plt.xlabel("$(|dF_{total}| - |dF_{LPF+BPF})|$")
plt.ylabel("Probability [-]")
plt.grid()
plt.savefig(out_dir+"PDF_dF_diff.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.scatter(Time_mag_threshold,dF_diff_threshold)
plt.xlabel("Time [s]")
plt.ylabel("$(|dF_{total}| - |dF_{LPF+BPF}|)$")
plt.grid()
plt.title("Threshold on $2\sigma$ $dF_{total}$")
plt.savefig(out_dir+"dF_diff_threshold.png")
plt.close()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(dF_diff_threshold)
plt.yscale("log")
plt.plot(X,P,"-k")
plt.xlabel("$(|dF_{total}| - |dF_{LPF+BPF}|)$")
plt.ylabel("Probability [-]")
plt.grid()
plt.title("Threshold on $2\sigma$ $dF_{total}$")
plt.savefig(out_dir+"PDF_dF_diff_threshold.png")
plt.close()


#LPF BPF FBR calc
Time_mag = []
dF_diff = []
dF_ratio = []
Time_mag_threshold = []
dF_diff_threshold = []
dF_ratio_threshold = []
for i in np.arange(0,len(zero_crossings_index_LPF_FBR)-1):

    it_1 = zero_crossings_index_LPF_FBR[i]
    it_2 = zero_crossings_index_LPF_FBR[i+1]

   
    Time_mag.append(Time_OF[it_1])
    dF_diff.append((abs(FBR[it_2] - FBR[it_1]) - abs(LPF_2_FBR[it_2] - LPF_2_FBR[it_1])))
    #dF_ratio.append((FBR[it_2] - FBR[it_1])/(LPF_2_FBR[it_2] - LPF_2_FBR[it_1]))

    if abs(FBR[it_2] - FBR[it_1])>=threshold:
        Time_mag_threshold.append(Time_OF[it_1])
        dF_diff_threshold.append((abs(FBR[it_2] - FBR[it_1]) - abs(LPF_2_FBR[it_2] - LPF_2_FBR[it_1])))
        #dF_ratio_threshold.append((FBR[it_2] - FBR[it_1])/(LPF_2_FBR[it_2] - LPF_2_FBR[it_1]))



plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
fig = plt.figure(figsize=(14,8))
plt.scatter(Time_mag,dF_diff)
plt.xlabel("Time [s]")
plt.ylabel("$(|dF_{total}| - |dF_{LPF+BPF}|)$")
plt.grid()
plt.savefig(out_dir+"dF_diff_2.png")
plt.close()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(dF_diff)
plt.yscale("log")
plt.plot(X,P,"-k")
plt.xlabel("$(|dF_{total}| - |dF_{LPF+BPF}|)$")
plt.ylabel("Probability [-]")
plt.grid()
plt.savefig(out_dir+"PDF_dF_diff_2.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.scatter(Time_mag_threshold,dF_diff_threshold)
plt.xlabel("Time [s]")
plt.ylabel("$(|dF_{total}| - |dF_{LPF+BPF})|$")
plt.grid()
plt.title("Threshold on $2\sigma$ $dF_{total}$")
plt.savefig(out_dir+"dF_diff_threshold_2.png")
plt.close()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(dF_diff_threshold)
plt.yscale("log")
plt.plot(X,P,"-k")
plt.xlabel("$(|dF_{total}| - |dF_{LPF+BPF}|)$")
plt.ylabel("Probability [-]")
plt.grid()
plt.title("Threshold on $2\sigma$ $dF_{total}$")
plt.savefig(out_dir+"PDF_dF_diff_threshold_2.png")
plt.close()



plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/FB_dF_all_times/"
times = np.arange(200,1220,20)
for j in np.arange(0,len(times)-1):
    it_1 = np.searchsorted(Time_OF,times[j])
    it_2 = np.searchsorted(Time_OF,times[j+1])

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[it_1:it_2],FBR[it_1:it_2],"-k")
    plt.plot(Time_OF[it_1:it_2],LPF_2_FBR[it_1:it_2],"-r")
    for i in np.arange(0,len(dF_mag)):
        if dF_mag[i] >= 2*np.std(dF_mag) and Time_OF[it_1] <=Time_mag[i] <= Time_OF[it_2]:
            plt.plot(Time_mag[i],FBR_p[i],"oy")
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Bearing force magnitude [kN]")
    plt.tight_layout()
    plt.savefig(out_dir+"{}_{}.png".format(times[j],times[j+1]))
    plt.close()

plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
ax1.scatter(Time_mag,dF_mag,s=1,color="b")
ax1.grid()
ax3=ax1.twiny()
P,X = probability_dist(dF_mag)
ax3.plot(P,X,"-k")
ax3.axvline(x=0.0,color="k")
ax1.set_title("$dF$ $F_B$ [kN]")
ax2.scatter(Time_mag,dt_mag,s=1,color="b")
ax2.grid()
ax4=ax2.twiny()
P,X = probability_dist(dt_mag)
ax4.plot(P,X,"-k")
ax4.axvline(x=0.0,color="k")
ax2.set_title("$dt$ $F_B$ [s]")
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"dF_dt_FBR.png")
plt.close()

print("dF_mag",moments(dF_mag))
print("dt mag",moments(dt_mag))


dF_mag_threshold = []
dt_mag_threshold = []
time_mag_threshold = []
for i in np.arange(0,len(dF_mag)):
    if dF_mag[i] >= 2*np.std(dF_mag):
        dF_mag_threshold.append(dF_mag[i]); dt_mag_threshold.append(dt_mag[i]);time_mag_threshold.append(Time_mag[i])
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
ax1.scatter(time_mag_threshold,dF_mag_threshold,s=2,color="b")
ax1.grid()
ax3=ax1.twiny()
P,X = probability_dist(dF_mag_threshold)
ax3.plot(P,X,"-k")
ax3.axvline(x=0.0,color="k")
ax1.set_title("$dF$ $F_B$ threshold on $2\sigma$ [kN]")
ax2.scatter(time_mag_threshold,dt_mag_threshold,s=2,color="b")
ax2.grid()
ax4=ax2.twiny()
P,X = probability_dist(dt_mag_threshold)
ax4.plot(P,X,"-k")
ax4.axvline(x=0.0,color="k")
ax2.set_title("$dt$ $F_B$ threshold on $2\sigma$ [s]")
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"dF_dt_FBR_threshold.png")
plt.close()

print("dF threshold",moments(dF_mag_threshold))
print("dt threshold",moments(dt_mag_threshold))


zero_crossings_index_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
#FBR BPF calc
dF_mag = []
dt_mag = []
FBR_p = []
Time_mag = []
for i in np.arange(0,len(zero_crossings_index_FBR)-1):

    it_1 = zero_crossings_index_FBR[i]
    it_2 = zero_crossings_index_FBR[i+1]

    Time_mag.append(Time_OF[it_1])

    dt_mag.append(Time_OF[it_2]-Time_OF[it_1])

    FBR_p.append(FBR[it_1])

    dF_mag.append(abs(FBR[it_2] - FBR[it_1]))

plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
ax1.scatter(Time_mag,dF_mag,s=1,color="b")
ax1.grid()
ax3=ax1.twiny()
P,X = probability_dist(dF_mag)
ax3.plot(P,X,"-k")
ax3.axvline(x=0.0,color="k")
ax1.set_title("BPF $dF$ $F_B$ [kN]")
ax2.scatter(Time_mag,dt_mag,s=1,color="b")
ax2.grid()
ax4=ax2.twiny()
P,X = probability_dist(dt_mag)
ax4.plot(P,X,"-k")
ax4.axvline(x=0.0,color="k")
ax2.set_title("BPF $dt$ $F_B$ [s]")
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"BPF_dF_dt_FBR.png")
plt.close()

print("BPF dF_mag",moments(dF_mag))
print("BPF dt mag",moments(dt_mag))


dF_mag_threshold = []
dt_mag_threshold = []
time_mag_threshold = []
for i in np.arange(0,len(dF_mag)):
    if dF_mag[i] >= 2*np.std(dF_mag):
        dF_mag_threshold.append(dF_mag[i]); dt_mag_threshold.append(dt_mag[i]);time_mag_threshold.append(Time_mag[i])
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
ax1.scatter(time_mag_threshold,dF_mag_threshold,s=2,color="b")
ax1.grid()
ax3=ax1.twiny()
P,X = probability_dist(dF_mag_threshold)
ax3.plot(P,X,"-k")
ax3.axvline(x=0.0,color="k")
ax1.set_title("BPF $dF$ $F_B$ threshold on $2\sigma$ [kN]")
ax2.scatter(time_mag_threshold,dt_mag_threshold,s=2,color="b")
ax2.grid()
ax4=ax2.twiny()
P,X = probability_dist(dt_mag_threshold)
ax4.plot(P,X,"-k")
ax4.axvline(x=0.0,color="k")
ax2.set_title("BPF $dt$ $F_B$ threshold on $2\sigma$ [s]")
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"BPF_dF_dt_FBR_threshold.png")
plt.close()

print("BPF dF threshold",moments(dF_mag_threshold))
print("BPF dt threshold",moments(dt_mag_threshold))


zero_crossings_index_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]


#HPF FBR calc
dF_mag = []
dt_mag = []
FBR_p = []
Time_mag = []
for i in np.arange(0,len(zero_crossings_index_FBR)-1):

    it_1 = zero_crossings_index_FBR[i]
    it_2 = zero_crossings_index_FBR[i+1]

    Time_mag.append(Time_OF[it_1])

    dt_mag.append(Time_OF[it_2]-Time_OF[it_1])

    FBR_p.append(FBR[it_1])

    dF_mag.append(abs(FBR[it_2] - FBR[it_1]))

plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
ax1.scatter(Time_mag,dF_mag,s=1,color="b")
ax1.grid()
ax3=ax1.twiny()
P,X = probability_dist(dF_mag)
ax3.plot(P,X,"-k")
ax3.axvline(x=0.0,color="k")
ax1.set_title("HPF $dF$ $F_B$ [kN]")
ax2.scatter(Time_mag,dt_mag,s=1,color="b")
ax2.grid()
ax4=ax2.twiny()
P,X = probability_dist(dt_mag)
ax4.plot(P,X,"-k")
ax4.axvline(x=0.0,color="k")
ax2.set_title("HPF $dt$ $F_B$ [s]")
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"HPF_dF_dt_FBR.png")
plt.close()

print("HPF dF_mag",moments(dF_mag))
print("HPF dt mag",moments(dt_mag))


dF_mag_threshold = []
dt_mag_threshold = []
time_mag_threshold = []
for i in np.arange(0,len(dF_mag)):
    if dF_mag[i] >= 2*np.std(dF_mag):
        dF_mag_threshold.append(dF_mag[i]); dt_mag_threshold.append(dt_mag[i]);time_mag_threshold.append(Time_mag[i])
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
ax1.scatter(time_mag_threshold,dF_mag_threshold,s=2,color="b")
ax1.grid()
ax3=ax1.twiny()
P,X = probability_dist(dF_mag_threshold)
ax3.plot(P,X,"-k")
ax3.axvline(x=0.0,color="k")
ax1.set_title("HPF $dF$ $F_B$ threshold on $2\sigma$ [kN]")
ax2.scatter(time_mag_threshold,dt_mag_threshold,s=2,color="b")
ax2.grid()
ax4=ax2.twiny()
P,X = probability_dist(dt_mag_threshold)
ax4.plot(P,X,"-k")
ax4.axvline(x=0.0,color="k")
ax2.set_title("HPF $dt$ $F_B$ threshold on $2\sigma$ [s]")
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"HPF_dF_dt_FBR_threshold.png")
plt.close()

print("HPF dF threshold",moments(dF_mag_threshold))
print("HPF dt threshold",moments(dt_mag_threshold))



zero_crossings_index_LPF_OOPBM = np.where(np.diff(np.sign(dLPF_OOPBM)))[0]
zero_crossings_index_BPF_OOPBM = np.where(np.diff(np.sign(dBPF_OOPBM)))[0]
zero_crossings_index_HPF_OOPBM = np.where(np.diff(np.sign(dHPF_OOPBM)))[0]


#LPF OOPBM calc
dF_mag_LPF = []
MR_LPF = []
Time_mag_LPF = []
for i in np.arange(0,len(zero_crossings_index_LPF_OOPBM)-1):

    it_1 = zero_crossings_index_LPF_OOPBM[i]
    it_2 = zero_crossings_index_LPF_OOPBM[i+1]

    Time_mag_LPF.append(Time_OF[it_1])

    MR_LPF.append(LPF_OOPBM[it_1])

    dF_mag_LPF.append(abs(LPF_OOPBM[it_2] - LPF_OOPBM[it_1]))

threshold = 2*np.std(dF_mag_LPF)


LPF_burst_array = []
burst = []
for i in np.arange(0,len(zero_crossings_index_LPF_OOPBM)-2):

    it_1 = zero_crossings_index_LPF_OOPBM[i]
    it_2 = zero_crossings_index_LPF_OOPBM[i+1]
    it_3 = zero_crossings_index_LPF_OOPBM[i+2]

    dF_1 = abs(LPF_OOPBM[it_2] - LPF_OOPBM[it_1])
    dF_2 = abs(LPF_OOPBM[it_3] - LPF_OOPBM[it_2])

    
    if dF_1 >= threshold and dF_2 >= threshold and len(burst)<2:
        burst.append(Time_OF[it_1]); burst.append(Time_OF[it_3])
    elif dF_1 >= threshold and dF_2 >= threshold and len(burst)==2:
        burst[1] = Time_OF[it_2]
    elif dF_1 >= threshold and dF_2 < threshold and len(burst) == 2:
        LPF_burst_array.append(burst); burst = []

LPF_burst_array = np.array(LPF_burst_array)

#BPF OOPBM calc
dF_mag_BPF = []
MR_BPF = []
Time_mag_BPF = []
for i in np.arange(0,len(zero_crossings_index_BPF_OOPBM)-1):

    it_1 = zero_crossings_index_BPF_OOPBM[i]
    it_2 = zero_crossings_index_BPF_OOPBM[i+1]

    Time_mag_BPF.append(Time_OF[it_1])

    MR_BPF.append(BPF_OOPBM[it_1])

    dF_mag_BPF.append(abs(BPF_OOPBM[it_2] - BPF_OOPBM[it_1]))


fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF,BPF_OOPBM,"-r")
for i in np.arange(0,len(dF_mag_BPF)):
    if dF_mag_BPF[i] >= 2*np.std(dF_mag_BPF):
        plt.plot(Time_mag_BPF[i],MR_BPF[i],"ob")
plt.grid()
plt.show()


#HPF OOPBM calc
dF_mag_HPF = []
MR_HPF = []
Time_mag_HPF = []
for i in np.arange(0,len(zero_crossings_index_HPF_OOPBM)-1):

    it_1 = zero_crossings_index_HPF_OOPBM[i]
    it_2 = zero_crossings_index_HPF_OOPBM[i+1]

    Time_mag_HPF.append(Time_OF[it_1])

    MR_HPF.append(HPF_OOPBM[it_1])

    dF_mag_HPF.append(abs(HPF_OOPBM[it_2] - HPF_OOPBM[it_1]))


threshold = 2*np.std(dF_mag_BPF)

burst_array = []
burst = []
for i in np.arange(0,len(zero_crossings_index_BPF_OOPBM)-2):

    it_1 = zero_crossings_index_BPF_OOPBM[i]
    it_2 = zero_crossings_index_BPF_OOPBM[i+1]
    it_3 = zero_crossings_index_BPF_OOPBM[i+2]

    dF_1 = abs(BPF_OOPBM[it_2] - BPF_OOPBM[it_1])
    dF_2 = abs(BPF_OOPBM[it_3] - BPF_OOPBM[it_2])

    
    if dF_1 >= threshold and dF_2 >= threshold and len(burst)<2:
        burst.append(Time_OF[it_1]); burst.append(Time_OF[it_3])
    elif dF_1 >= threshold and dF_2 >= threshold and len(burst)==2:
        burst[1] = Time_OF[it_2]
    elif dF_1 >= threshold and dF_2 < threshold and len(burst) == 2:
        burst_array.append(burst); burst = []

burst_array = np.array(burst_array)
threshold = 2*np.std(dF_mag_HPF)

k = 0; burst_times = []; burst_MR_HPF = []
for i in np.arange(0,len(zero_crossings_index_HPF_OOPBM)-1):

    it_1 = zero_crossings_index_HPF_OOPBM[i]
    it_2 = zero_crossings_index_HPF_OOPBM[i+1]


    if abs(HPF_OOPBM[it_2] - HPF_OOPBM[it_1]) >= threshold:
        Time_it_1 = Time_OF[it_1]
        for j in np.arange(0,len(burst_array)):
            if burst_array[j,0] <= Time_it_1 <= burst_array[j,1]:
                k+=1; burst_times.append(Time_it_1); burst_MR_HPF.append(HPF_OOPBM[it_1])
                break

m = 0
for i in np.arange(0,len(dF_mag_HPF)):
    if dF_mag_HPF[i] >= 2*np.std(dF_mag_HPF):
        m+=1






out_dir=in_dir+"three_frequency_analysis/OOPBM_analysis/"
plt.rcParams['font.size'] = 16

perc = round((k/m)*100,2)
fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF,BPF_OOPBM,"-r",label="BPF")
plt.plot(Time_OF,HPF_OOPBM,"-b",label="HPF")
for i in np.arange(0,len(dF_mag_BPF)):
    if dF_mag_BPF[i] >= 2*np.std(dF_mag_BPF):
        plt.plot(Time_mag_BPF[i],MR_BPF[i],"ob")
for i in np.arange(0,len(dF_mag_HPF)):
    if dF_mag_HPF[i] >= 2*np.std(dF_mag_HPF):
        plt.plot(Time_mag_HPF[i],MR_HPF[i],"or")
for burst in burst_array:
    plt.axvline(x=burst[0],color="k")
    plt.axvline(x=burst[1],color="k")
for i in np.arange(0,len(burst_times)):
    plt.plot(burst_times[i],burst_MR_HPF[i],"ok")

xl = [0,-1]; yl = [0,-1]
plt.plot(xl,yl,"ob",label="2$\sigma$ BPF")
plt.plot(xl,yl,"or",label="2$\sigma$ HPF")
plt.plot(xl,yl,"-k",label="burst event BPF")
plt.plot(xl,yl,"ok",label="2$\sigma$ HPF within BPF burst")
plt.xlim([150,1250])
plt.ylim([-2000,2000])

plt.xlabel("Time [s]")
plt.ylabel("Out-of-plane bending moment magnitude [kN-m]")
plt.title("percentage of 2$\sigma$ HPF jumps\nwithin bursts of BPF 2$\sigma$ events = {}".format(perc))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"BPF_HPF_bursting_events.png")
plt.close()

threshold = 2*np.std(dF_mag_BPF)
k = 0; burst_times = []; burst_MR_BPF = []
for i in np.arange(0,len(zero_crossings_index_BPF_OOPBM)-1):

    it_1 = zero_crossings_index_BPF_OOPBM[i]
    it_2 = zero_crossings_index_BPF_OOPBM[i+1]


    if abs(BPF_OOPBM[it_2] - BPF_OOPBM[it_1]) >= threshold:
        Time_it_1 = Time_OF[it_1]
        for j in np.arange(0,len(LPF_burst_array)):
            if LPF_burst_array[j,0] <= Time_it_1 <= LPF_burst_array[j,1]:
                k+=1; burst_times.append(Time_it_1); burst_MR_BPF.append(BPF_OOPBM[it_1])
                break

m = 0
for i in np.arange(0,len(dF_mag_BPF)):
    if dF_mag_BPF[i] >= 2*np.std(dF_mag_BPF):
        m+=1

perc = round((k/m)*100,2)
fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF,LPF_OOPBM,"-g",label="LPF")
plt.plot(Time_OF,BPF_OOPBM,"-r",label="BPF")
for i in np.arange(0,len(dF_mag_LPF)):
    if dF_mag_LPF[i] >= 2*np.std(dF_mag_LPF):
        plt.plot(Time_mag_LPF[i],MR_LPF[i],"om")
for i in np.arange(0,len(dF_mag_BPF)):
    if dF_mag_BPF[i] >= 2*np.std(dF_mag_BPF):
        plt.plot(Time_mag_BPF[i],MR_BPF[i],"ob")
for burst in LPF_burst_array:
    plt.axvline(x=burst[0],color="k")
    plt.axvline(x=burst[1],color="k")
for i in np.arange(0,len(burst_times)):
    plt.plot(burst_times[i],burst_MR_BPF[i],"ok")

xl = [0,-1]; yl = [0,-1]
plt.plot(xl,yl,"om",label="2$\sigma$ LPF")
plt.plot(xl,yl,"ob",label="2$\sigma$ BPF")
plt.plot(xl,yl,"-k",label="burst event LPF")
plt.plot(xl,yl,"ok",label="2$\sigma$ BPF within LPF burst")
plt.xlim([150,1250])
plt.ylim([-2000,4500])

plt.xlabel("Time [s]")
plt.ylabel("Out-of-plane bending moment magnitude [kN-m]")
plt.title("percentage of 2$\sigma$ BPF jumps\nwithin bursts of LPF 2$\sigma$ events = {}".format(perc))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"LPF_BPF_bursting_events.png")
plt.close()



threshold = 2*np.std(dF_mag_HPF)
k = 0; burst_times = []; burst_MR_HPF = []
for i in np.arange(0,len(zero_crossings_index_HPF_OOPBM)-1):

    it_1 = zero_crossings_index_HPF_OOPBM[i]
    it_2 = zero_crossings_index_HPF_OOPBM[i+1]


    if abs(HPF_OOPBM[it_2] - HPF_OOPBM[it_1]) >= threshold:
        Time_it_1 = Time_OF[it_1]
        for j in np.arange(0,len(LPF_burst_array)):
            if LPF_burst_array[j,0] <= Time_it_1 <= LPF_burst_array[j,1]:
                k+=1; burst_times.append(Time_it_1); burst_MR_HPF.append(HPF_OOPBM[it_1])
                break

m = 0
for i in np.arange(0,len(dF_mag_HPF)):
    if dF_mag_HPF[i] >= 2*np.std(dF_mag_HPF):
        m+=1

perc = round((k/m)*100,2)
fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF,LPF_OOPBM,"-g",label="LPF")
plt.plot(Time_OF,HPF_OOPBM,"-b",label="HPF")
for i in np.arange(0,len(dF_mag_LPF)):
    if dF_mag_LPF[i] >= 2*np.std(dF_mag_LPF):
        plt.plot(Time_mag_LPF[i],MR_LPF[i],"om")
for i in np.arange(0,len(dF_mag_HPF)):
    if dF_mag_HPF[i] >= 2*np.std(dF_mag_HPF):
        plt.plot(Time_mag_HPF[i],MR_HPF[i],"or")
for burst in LPF_burst_array:
    plt.axvline(x=burst[0],color="k")
    plt.axvline(x=burst[1],color="k")
for i in np.arange(0,len(burst_times)):
    plt.plot(burst_times[i],burst_MR_HPF[i],"ok")

xl = [0,-1]; yl = [0,-1]
plt.plot(xl,yl,"om",label="2$\sigma$ LPF")
plt.plot(xl,yl,"or",label="2$\sigma$ HPF")
plt.plot(xl,yl,"-k",label="burst event LPF")
plt.plot(xl,yl,"ok",label="2$\sigma$ HPF within LPF burst")
plt.xlim([150,1250])
plt.ylim([-2000,4500])

plt.xlabel("Time [s]")
plt.ylabel("Out-of-plane bending moment magnitude [kN-m]")
plt.title("percentage of 2$\sigma$ HPF jumps\nwithin bursts of LPF 2$\sigma$ events = {}".format(perc))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"LPF_HPF_bursting_events.png")
plt.close()



fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF,OOPBM,"-k",label="$Total$")
plt.plot(Time_OF,LPF_1_OOPBM,"-g",label="LPF")
plt.plot(Time_OF,np.subtract(BPF_OOPBM,1000),"-r",label="BPF -1000kN-m")
plt.plot(Time_OF,np.subtract(HPF_OOPBM,1000),"-b",label="HPF -1000kN-m")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Aerodynamic Out-of-plane\nbending moment magnitude [kN-m]")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"OOPBM_all_freqs.png")
plt.close()




zero_crossings_index_BPF_OOPBM = np.where(np.diff(np.sign(dBPF_OOPBM)))[0]

Env_BPF_OOPBM = []
Env_Times = []
for i in np.arange(0,len(zero_crossings_index_BPF_OOPBM),2):
    idx = zero_crossings_index_BPF_OOPBM[i]
    Env_BPF_OOPBM.append(BPF_OOPBM[idx]); Env_Times.append(Time_OF[idx])


#local variance calc
local_var_LPF = []
local_var_BPF = []
local_var_HPF = []
start_time_idx = np.searchsorted(Time_OF,10+Time_OF[0])
for i in np.arange(0,len(Time_OF)-start_time_idx,1):
    local_var_LPF.append(np.std(LPF_1_OOPBM[i:i+start_time_idx]))
    local_var_BPF.append(np.std(BPF_OOPBM[i:i+start_time_idx]))
    local_var_HPF.append(np.std(HPF_OOPBM[i:i+start_time_idx]))

f = interpolate.interp1d(Time_OF,LPF_1_OOPBM)
LPF_1_OOPBM_interp = f(Env_Times)
cc1 = round(correlation_coef(LPF_1_OOPBM_interp,Env_BPF_OOPBM),2)

idx = int(start_time_idx/2)
cc2 = round(correlation_coef(LPF_1_OOPBM[idx+1:-idx],local_var_HPF),2)
cc3 = round(correlation_coef(local_var_BPF,local_var_HPF),2)

fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF,np.add(LPF_1_OOPBM,400),"-g",label="LPF +400kN-m")
plt.plot(Env_Times,np.add(Env_BPF_OOPBM,50),"-r",label="Envelope BPF +50kN-m")
plt.plot(Time_OF[idx+1:-idx],local_var_HPF,"-b",label="Local std HPF")
plt.xlabel("Time [s]")
plt.ylabel("Out-of-plane bending moment magnitude [kN-m]")
plt.legend()
plt.grid()
plt.title("cc LPF BPF = {}, cc LPF HPF = {}, cc BPF HPF = {}".format(cc1,cc2,cc3))
plt.tight_layout()
plt.savefig(out_dir+"cc_all_freqs.png")
plt.close()




Time_shift_idx = np.searchsorted(Env_Times,Env_Times[0]+4.6)

Rotor_grads = df_OF.groups["Rotor_Gradients"]
drUx = np.array(Rotor_grads.variables["drUx"][Time_sampling_start_idx:])
f = interpolate.interp1d(Time_sampling,drUx)
drUx_interp = f(Env_Times)
drUx_interp_shifted = drUx_interp[:-Time_shift_idx]

b = Dataset(in_dir+"Dataset_2.nc")
Rotor_grads = b.groups["Rotor_Gradients"]
drUx_2 = np.array(Rotor_grads.variables["drUx"][Time_sampling_start_idx:])
f = interpolate.interp1d(Time_sampling,drUx_2)
drUx_2_interp = f(Env_Times)
drUx_2_interp_shifted = drUx_2_interp[:-Time_shift_idx]

Env_Times_shifted = Env_Times[:-Time_shift_idx]
Env_BPF_OOPBM_shifted = Env_BPF_OOPBM[Time_shift_idx:]

f = interpolate.interp1d(Time_OF,LPF_1_OOPBM)
LPF_1_OOPBM_interp = f(Env_Times)
print("LPF BPF env",correlation_coef(LPF_1_OOPBM_interp,Env_BPF_OOPBM))

idx = np.searchsorted(Env_Times,Env_Times[0]+10)
cc = []
for it in np.arange(0,len(Env_Times)-idx):
    cc.append(correlation_coef(Env_BPF_OOPBM[it:it+idx],LPF_1_OOPBM_interp[it:it+idx]))

fig,(ax,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax.plot(Env_Times,LPF_1_OOPBM_interp,"-b",label="LPF 0.3Hz $M_{H}$")
ax.plot(Env_Times,Env_BPF_OOPBM,"-r",label="BPF 0.3-0.9Hz $M_{H}$")
fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(LPF_1_OOPBM_interp,Env_BPF_OOPBM),2)))
fig.supxlabel("Time [s]")
ax.set_ylabel("Magnitude Aerodynamic\nRotor moment vector [kN-m]")
ax.legend()
ax.grid()
ax2.plot(Env_Times[int(idx/2):-int(idx/2)],cc,"-k")
ax2.set_ylabel("local correlation\ncoefficient T= 10s")
ax2.grid()
plt.tight_layout()
plt.show()

out_dir = in_dir+"three_frequency_analysis/OOPBM_analysis/"
cc = round(correlation_coef(Env_BPF_OOPBM_shifted,drUx_interp_shifted),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Env_Times_shifted,Env_BPF_OOPBM_shifted,"-r")
ax.set_ylabel("Envelope BPF OOPBM [kN-m]")
ax.grid()
ax2=ax.twinx()
ax2.plot(Env_Times_shifted,drUx_2_interp_shifted,"-b")
ax2.set_ylabel("Rotor average Magntiude velocity gradient [1/s]\n0-100% span")
fig.suptitle("correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"drUx_Env_BPF_OOPBM.png")
plt.close()

cc = round(correlation_coef(Env_BPF_OOPBM_shifted,drUx_2_interp_shifted),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Env_Times_shifted,Env_BPF_OOPBM_shifted,"-r")
ax.set_ylabel("Envelope BPF OOPBM [kN-m]")
ax.grid()
ax2=ax.twinx()
ax2.plot(Env_Times_shifted,drUx_2_interp_shifted,"-b")
ax2.set_ylabel("Rotor average Magntiude velocity gradient [1/s]\n70-80% span")
fig.suptitle("correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"drUx_Env_BPF_OOPBM_2.png")
plt.close()

out_dir = in_dir+"three_frequency_analysis/frequencies_all_times_3P_LPF_OOPBM/"
Times = np.arange(200,1300,100)
for i in np.arange(0,len(Times)-1):
    idx1 = np.searchsorted(Time_OF,Times[i]);idx2 = np.searchsorted(Time_OF,Times[i+1])
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[idx1:idx2],OOPBM[idx1:idx2],"-k",label="Total $M_{H}$")
    plt.plot(Time_OF[idx1:idx2],LPF_1_OOPBM[idx1:idx2],"-g",label="LPF 0.3Hz $M_{H}$")
    plt.plot(Time_OF[idx1:idx2],BPF_OOPBM[idx1:idx2],"-r",label="BPF 0.3-0.9Hz $M_{H}$")
    idx1 = np.searchsorted(Env_Times,Times[i]);idx2 = np.searchsorted(Env_Times,Times[i+1])
    plt.plot(Env_Times[idx1:idx2],Env_BPF_OOPBM[idx1:idx2],"--b",linewidth=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude Aerodynamic Rotor moment vector [kN-m]")
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"{}_{}.png".format(Times[i],Times[i+1]))
    plt.close()



out_dir = in_dir+"three_frequency_analysis/frequencies_all_times_HPF_3P_OOPBM/"
Times = np.arange(200,1300,100)
for i in np.arange(0,len(Times)-1):
    idx1 = np.searchsorted(Time_OF,Times[i]);idx2 = np.searchsorted(Time_OF,Times[i+1])
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[idx1:idx2],OOPBM[idx1:idx2],"-k",label="Total $M_{H}$")
    plt.plot(Time_OF[idx1:idx2],LPF_1_OOPBM[idx1:idx2],"-g",label="LPF 0.3Hz $M_{H}$")
    plt.plot(Time_OF[idx1:idx2],BPF_OOPBM[idx1:idx2],"-r",label="BPF 0.3-0.9Hz $M_{H}$")
    plt.plot(Time_OF[idx1:idx2],HPF_OOPBM[idx1:idx2]-1000,"-b",label="HPF 1.5Hz $M_{H}$\noffset: -1000kN-m")
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude Aerodynamic Rotor moment vector [kN-m]")
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"{}_{}.png".format(Times[i],Times[i+1]))
    plt.close()



f = interpolate.interp1d(Time_OF,LPF_1_OOPBM)
LPF_1_OOPBM_interp = f(Env_Times)
print("LPF BPF env",correlation_coef(LPF_1_OOPBM_interp,Env_BPF_OOPBM))

idx = np.searchsorted(Env_Times,Env_Times[0]+10)
cc = []
for it in np.arange(0,len(Env_Times)-idx):
    cc.append(correlation_coef(Env_BPF_OOPBM[it:it+idx],LPF_1_OOPBM_interp[it:it+idx]))

out_dir=in_dir+"three_frequency_analysis/OOPBM_analysis/"
plt.rcParams.update({'font.size': 18})


fig,(ax,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax.plot(Env_Times,LPF_1_OOPBM_interp,"-b",label="LPF 0.3Hz $M_{H}$")
ax.plot(Env_Times,Env_BPF_OOPBM,"-r",label="BPF 0.3-0.9Hz $M_{H}$")
fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(LPF_1_OOPBM_interp,Env_BPF_OOPBM),2)))
fig.supxlabel("Time [s]")
ax.set_ylabel("Magnitude Aerodynamic\nRotor moment vector [kN-m]")
ax.legend()
ax.grid()
ax2.plot(Env_Times[int(idx/2):-int(idx/2)],cc,"-k")
ax2.set_ylabel("local correlation\ncoefficient T= 10s")
ax2.grid()
plt.tight_layout()
plt.show()
# plt.savefig(out_dir+"local_cc_LPF_BPF.png")
# plt.close()

out_dir = in_dir+"three_frequency_analysis/OOPBM_analysis/"
LPF_1_OOPBM_interp_sampling = f(Time_sampling)
cc = round(correlation_coef(I,LPF_1_OOPBM_interp_sampling),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling,I,"-b")
ax.set_ylabel("Magnitude Asymmetry vector [$m^4/s$]")

ax2=ax.twinx()
ax2.plot(Env_Times,Env_BPF_OOPBM,"-r")
ax2.set_ylabel("Envelope BPF OOPBM [kN-m]")
ax.grid()
fig.suptitle("Correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"I_cc_BPF_OOPBM.png")
plt.close()



#local variance calc
local_var_LPF = []
local_var_BPF = []
local_var_HPF = []
start_time_idx = np.searchsorted(Time_OF,10+Time_OF[0])
for i in np.arange(0,len(Time_OF)-start_time_idx,1):
    local_var_LPF.append(np.std(LPF_1_OOPBM[i:i+start_time_idx]))
    local_var_BPF.append(np.std(BPF_OOPBM[i:i+start_time_idx]))
    local_var_HPF.append(np.std(HPF_OOPBM[i:i+start_time_idx]))

plt.rcParams['font.size'] = 12
fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF[:-start_time_idx],np.add(local_var_LPF,400),"-g",label="LPF (0.3Hz), offset +400kN-m")
plt.plot(Time_OF[:-start_time_idx],np.add(local_var_BPF,50),"-r",label="BPF (0.3-0.9Hz), offset +50kN-m")
plt.plot(Time_OF[:-start_time_idx],local_var_HPF,"-b",label="HPF (1.5Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Local standard deviation T=10s\nAerodynamic Rotor moment vector $M_{H}$ [kN-m]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"loc_var.png")
plt.close()

print(correlation_coef(local_var_LPF,local_var_BPF))
print(correlation_coef(local_var_LPF,local_var_HPF))
print(correlation_coef(local_var_BPF,local_var_HPF))

fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF,np.add(LPF_1_OOPBM,400),"-g",label="LPF $M_H$ (0.3Hz), offset +400kN-m")
plt.plot(Env_Times,np.add(Env_BPF_OOPBM,50),"-r",label="Env: BPF (0.3-0.9Hz), offset +50kN-m")
plt.plot(Time_OF[:-start_time_idx],local_var_HPF,"-b",label="Local std: HPF (1.5Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Local standard deviation T=10s\nAerodynamic Rotor moment vector $M_{H}$ [kN-m]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"loc_var_rev2.png")
plt.close()


print(correlation_coef(LPF_1_OOPBM_interp,Env_BPF_OOPBM))
print(correlation_coef(LPF_1_OOPBM[:-start_time_idx],local_var_HPF))
print(correlation_coef(local_var_BPF,local_var_HPF))


out_dir = in_dir+"three_frequency_analysis/OOPBM_analysis/"
start_time_sampling_idx = np.searchsorted(Time_sampling,10+Time_sampling[0])
I_cut = I[:-start_time_sampling_idx]

f = interpolate.interp1d(Time_OF[:-start_time_idx],local_var_HPF)
local_var_HPF_interp = f(Time_sampling[:-start_time_sampling_idx])

cc = round(correlation_coef(I_cut,local_var_HPF_interp),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling,I,"-b")
ax.set_ylabel("Magnitude Asymmetry vector [$m^4/s$]")

ax2=ax.twinx()
ax2.plot(Time_OF[:-start_time_idx],local_var_HPF,"-r")
ax2.set_ylabel("local standard deviation HPF OOPBM T = 10s [kN-m]")
ax.grid()
fig.suptitle("Correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"I_cc_HPF_OOPBM.png")
plt.close()


zero_crossings_index_HPF_OOPBM = np.where(np.diff(np.sign(dHPF_OOPBM)))[0]

Env_HPF_OOPBM = []
Env_Times = []
for i in np.arange(0,len(zero_crossings_index_HPF_OOPBM),2):
    idx = zero_crossings_index_HPF_OOPBM[i]
    Env_HPF_OOPBM.append(HPF_OOPBM[idx]); Env_Times.append(Time_OF[idx])


out_dir = in_dir+"three_frequency_analysis/HPF_env_all_times/"
Times = np.arange(200,1250,50)
for i in np.arange(0,len(Times)-1):
    idx1 = np.searchsorted(Time_OF,Times[i]);idx2 = np.searchsorted(Time_OF,Times[i+1])
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[idx1:idx2],HPF_OOPBM[idx1:idx2],"-r",label="HPF 1.5Hz $M_{H}$")
    idx1 = np.searchsorted(Env_Times,Times[i]);idx2 = np.searchsorted(Env_Times,Times[i+1])
    plt.plot(Env_Times[idx1:idx2],Env_HPF_OOPBM[idx1:idx2],"--b",linewidth=0.7,label="Envelope HPF $M_H$")
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude Aerodynamic Rotor moment vector [kN-m]")
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"{}_{}.png".format(Times[i],Times[i+1]))
    plt.close()


f = interpolate.interp1d(Time_OF,LPF_1_OOPBM)
LPF_1_OOPBM_interp = f(Env_Times)

idx = np.searchsorted(Env_Times,Env_Times[0]+10)
cc = []
for it in np.arange(0,len(Env_Times)-idx):
    cc.append(correlation_coef(Env_HPF_OOPBM[it:it+idx],LPF_1_OOPBM_interp[it:it+idx]))



fig,(ax,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax.plot(Env_Times,LPF_1_OOPBM_interp,"-b",label="LPF 0.3Hz $M_{H}$")
ax.plot(Env_Times,Env_HPF_OOPBM,"-r",label="HPF 1.5Hz $M_{H}$")
fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(LPF_1_OOPBM_interp,Env_HPF_OOPBM),2)))
fig.supxlabel("Time [s]")
ax.set_ylabel("Magnitude Aerodynamic\nRotor moment vector [kN-m]")
ax.legend()
ax.grid()
ax2.plot(Env_Times[idx:],cc,"-k")
ax2.set_ylabel("local correlation\ncoefficient T= 10s")
ax2.grid()
plt.tight_layout()
plt.savefig(out_dir+"LPF_HPF.png")
plt.close()



out_dir = in_dir+"three_frequency_analysis/frequencies_all_times_BPF_HPF_loc_std/"
Times = np.arange(200,1300,100)
for i in np.arange(0,len(Times)-1):
    idx1 = np.searchsorted(Time_OF,Times[i]);idx2 = np.searchsorted(Time_OF,Times[i+1])
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[idx1:idx2],np.subtract(HPF_OOPBM[idx1:idx2],600),"-b",label="HPF 1.5Hz $M_{H}$\noffset: -600kN-m") 
    plt.plot(Time_OF[idx1:idx2],BPF_OOPBM[idx1:idx2],"-r",label="BPF 0.3-0.9Hz $M_{H}$")
    if i == 9:
        idx2 = len(Time_OF)-start_time_idx
        plt.plot(np.add(Time_OF[idx1:idx2],5),np.subtract(local_var_HPF[idx1:idx2],600),"-k",label="loc std HPF 1.5Hz $M_H$\noffset -600kN-m")
        plt.plot(np.add(Time_OF[idx1:idx2],5),local_var_BPF[idx1:idx2],"-k",label="loc std BPF 0.3-0.9Hz $M_H$")
    else:
        plt.plot(np.add(Time_OF[idx1:idx2],5),np.subtract(local_var_HPF[idx1:idx2],600),"-k",label="loc std HPF 1.5Hz $M_H$\noffset -600kN-m")
        plt.plot(np.add(Time_OF[idx1:idx2],5),local_var_BPF[idx1:idx2],"-k",label="loc std BPF 0.3-0.9Hz $M_H$")
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude Aerodynamic Rotor moment vector [kN-m]")
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"{}_{}.png".format(Times[i],Times[i+1]))
    plt.close()


#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))




frq,PSD = temporal_spectra(FBR,dt,Var="FBR")

# out_dir=in_dir+"peak_peak_analysis/"

# fig = plt.figure(figsize=(14,8))
# plt.loglog(frq,PSD,"-r",label="Magnitude Main Bearing force vector [kN]")

dFBR_dt = np.array(dt_calc(FBR,dt))

Theta_FB = np.degrees(np.arctan2(FBz,FBy))
Theta_FB = np.array(theta_360(Theta_FB))

# out_dir=in_dir+"peak_peak_directional_analysis/"
# plt.rcParams['font.size'] = 16
# fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
# ax1.plot(Time_OF,Theta_FB,"-b")
# ax1.grid()

for it in Time_steps[:-1]:
    if FBz[it] > 0 and FBy[it] > 0 and FBy[it+1] > 0 and FBz[it+1] < 0:
        Theta_FB[it+1:]=Theta_FB[it+1:]-360
    elif FBz[it] < 0 and FBy[it] > 0 and FBy[it+1] > 0 and FBz[it+1] > 0:
        Theta_FB[it+1:]=Theta_FB[it+1:]+360   


# ax2.plot(Time_OF,Theta_FB,"-b")
# ax2.grid()

# fig.supxlabel("Time [s]")
# fig.supylabel("Polar position [deg]")

# plt.tight_layout()
# plt.show()

dTheta_FB_dt = np.array(dt_calc_theta(Theta_FB,dt))


#Total radial bearing force FBR inc weight
L1 = 1.912; L2 = 2.09

#Total bearing force
FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR_weight = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

# fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
# ax1.plot(Time_OF,FBR_weight,"-k")

dFBR_weight_dt = np.array(dt_calc(FBR_weight,dt))

Theta_FB_weight = np.degrees(np.arctan2(FBz,FBy))
Theta_FB_weight = np.array(theta_360(Theta_FB_weight))

for it in Time_steps[:-1]:
    if FBz[it] > 0 and FBy[it] > 0 and FBy[it+1] > 0 and FBz[it+1] < 0:
        Theta_FB_weight[it+1:]=Theta_FB_weight[it+1:]-360
    elif FBz[it] < 0 and FBy[it] > 0 and FBy[it+1] > 0 and FBz[it+1] > 0:
        Theta_FB_weight[it+1:]=Theta_FB_weight[it+1:]+360  

dTheta_FB_weight_dt = np.array(dt_calc_theta(Theta_FB_weight,dt))


FB_weight = np.sqrt(np.square(LSShftFxa)+np.square(FBy)+np.square(FBz))
# out_dir = in_dir+"peak_peak_analysis_weight/"
# plt.rcParams.update({'font.size': 18})
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,FB_weight,"-k",label="$F_B$")
# plt.plot(Time_OF,FBR_weight,"-b",label="$F_{B_R}$")
# plt.plot(Time_OF,LSShftFxa,"-r",label="$F_{B_x}$")
# plt.legend()
# plt.title("correlation coefficient $F_B, \, FBR$ = {}\ncorrelation coefficient $F_B, \, FBx = {}$".format(round(correlation_coef(FB_weight,FBR_weight),2),round(correlation_coef(FB_weight,LSShftFxa),2)))
# plt.xlabel("Time [s]")
# plt.ylabel("Magnitude Main Bearing force vector components [kN]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"FB_FBR_FBx.png")
# plt.close()


#Filtering FBR aero
LPF_1_FBR = low_pass_filter(FBR,0.3,dt)
LPF_2_FBR = low_pass_filter(FBR,0.9,dt)
LPF_3_FBR = low_pass_filter(FBR,1.5,dt)

HPF_FBR = np.subtract(FBR,LPF_3_FBR)
HPF_FBR = np.array(low_pass_filter(HPF_FBR,40,dt))
BPF_FBR = np.subtract(LPF_2_FBR,LPF_1_FBR)
dBPF_FBR = np.array(dt_calc(BPF_FBR,dt))
dHPF_FBR = np.array(dt_calc(HPF_FBR,dt))


plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/FBR_analysis/"
fig = plt.figure(figsize=(14,8))
P,X=probability_dist(FBR)
plt.plot(X,P,"-k",label="Total")
P,X=probability_dist(LPF_1_FBR)
plt.plot(X,P,"-g",label="LPF")
P,X=probability_dist(BPF_FBR)
plt.plot(X,P,"-r",label="BPF")
P,X=probability_dist(HPF_FBR)
plt.plot(X,P,"-b",label="HPF")
plt.xlabel("Bearing force magntiude [kN]")
plt.ylabel("Probability [-]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"PDF_FBR.png")
plt.close()

fig = plt.figure(figsize=(14,8))
P,X=probability_dist(FBR)
plt.plot(X,P,"-k",label="Total")
P,X=probability_dist(LPF_1_FBR)
plt.plot(X,P,"-g",label="LPF")
P,X=probability_dist(BPF_FBR)
plt.plot(X,P,"-r",label="BPF")
P,X=probability_dist(HPF_FBR)
plt.plot(X,P,"-b",label="HPF")
plt.xlabel("Bearing force magntiude [kN]")
plt.ylabel("Probability [-]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"PDF_FBR.png")
plt.close()



# idx1 = np.searchsorted(Time_OF,210);idx2 = np.searchsorted(Time_OF,230)
# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF[idx1:idx2],FBR[idx1:idx2],"-k",label="Total $F_{B_R}$")
# plt.plot(Time_OF[idx1:idx2],LPF_1_FBR[idx1:idx2],"-g",label="LPF 0.3Hz $F_{B_R}$")
# plt.plot(Time_OF[idx1:idx2],HPF_FBR[idx1:idx2]-600,"-b",label="HPF 1.5Hz $F_{B_R}$\noffset: -600kN")
# plt.plot(Time_OF[idx1:idx2],BPF_FBR[idx1:idx2],"-r",label="BPF 0.3-0.9Hz $F_{B_R}$")
# plt.xlabel("Time [s]")
# plt.ylabel("Magnitude Main Bearing force vector [kN]")
# plt.legend(loc="upper right")
# plt.grid()
# plt.tight_layout()
# plt.show()

#local variance calc
local_var_LPF = []
local_var_BPF = []
local_var_HPF = []
start_time_idx = np.searchsorted(Time_OF,20+Time_OF[0])
for i in np.arange(0,len(Time_OF)-start_time_idx,1):
    local_var_LPF.append(np.std(LPF_1_FBR[i:i+start_time_idx]))
    local_var_BPF.append(np.std(BPF_FBR[i:i+start_time_idx]))
    local_var_HPF.append(np.std(HPF_FBR[i:i+start_time_idx]))

plt.rcParams['font.size'] = 12
fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF[:-start_time_idx],np.add(local_var_LPF,200),"-g",label="LPF (0.3Hz), offset +200kN")
plt.plot(Time_OF[:-start_time_idx],np.add(local_var_BPF,50),"-r",label="BPF (0.3-0.9Hz), offset +50kN")
plt.plot(Time_OF[:-start_time_idx],local_var_HPF,"-b",label="HPF (1.5Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Local standard deviation T=20s\nMagnitude Main bearing force vector $F_{B_R}$ [kN]")
plt.grid()
plt.legend()
plt.tight_layout()

print(correlation_coef(local_var_LPF,local_var_BPF))
print(correlation_coef(local_var_LPF,local_var_HPF))
print(correlation_coef(local_var_BPF,local_var_HPF))

fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF[:-start_time_idx],LPF_1_FBR[:-start_time_idx],"-g",label="LPF (0.3Hz) total signal")
plt.plot(Time_OF[:-start_time_idx],np.add(local_var_BPF,50),"-r",label="BPF (0.3-0.9Hz), offset +50kN")
plt.plot(Time_OF[:-start_time_idx],local_var_HPF,"-b",label="HPF (1.5Hz)")
plt.xlabel("Time [s]")
plt.ylabel("Local standard deviation T=20s\nMagnitude Main bearing force vector $F_{B_R}$ [kN]")
plt.grid()
plt.legend()
plt.tight_layout()

print(correlation_coef(LPF_1_FBR[:-start_time_idx],local_var_BPF))
print(correlation_coef(LPF_1_FBR[:-start_time_idx],local_var_HPF))
print(correlation_coef(local_var_BPF,local_var_HPF))

plt.show()


print("Total LPF",correlation_coef(FBR,LPF_1_FBR))
print("Total BPF",correlation_coef(FBR,BPF_FBR))
print("Total HPF",correlation_coef(FBR,HPF_FBR))

print("LPF BPF",correlation_coef(LPF_1_FBR,BPF_FBR))
print("LPF HPF",correlation_coef(LPF_1_FBR,HPF_FBR))

print("BPF HPF",correlation_coef(BPF_FBR,HPF_FBR))


LPF_1_Theta_FBR = low_pass_filter(Theta_FB,0.3,dt)
LPF_2_Theta_FBR = low_pass_filter(Theta_FB,0.9,dt)
LPF_3_Theta_FBR = low_pass_filter(Theta_FB,1.5,dt)

HPF_Theta_FB = np.subtract(Theta_FB,LPF_3_Theta_FBR)
BPF_Theta_FB = np.subtract(LPF_2_Theta_FBR,LPF_1_Theta_FBR)

zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]

BPF_FBR_2 = []
Times_2 = []
for i in np.arange(0,len(zero_crossings_index_BPF_FBR),2):
    idx = zero_crossings_index_BPF_FBR[i]
    BPF_FBR_2.append(BPF_FBR[idx]); Times_2.append(Time_OF[idx])

f = interpolate.interp1d(Time_OF,LPF_1_FBR)
LPF_1_FBR_interp = f(Times_2)
print("LPF BPF env",correlation_coef(LPF_1_FBR_interp,BPF_FBR_2))

f = interpolate.interp1d(Time_OF,HPF_FBR)
HPF_FBR_interp = f(Times_2)
print("BPF env HPF",correlation_coef(BPF_FBR_2,HPF_FBR_interp))

out_dir = in_dir+"three_frequency_analysis/"
Times = np.arange(200,1300,100)
for i in np.arange(0,len(Times)-1):
    idx1 = np.searchsorted(Time_OF,Times[i]);idx2 = np.searchsorted(Time_OF,Times[i+1])
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[idx1:idx2],FBR[idx1:idx2],"-k",label="Total $F_{B_R}$")
    plt.plot(Time_OF[idx1:idx2],LPF_1_FBR[idx1:idx2],"-g",label="LPF 0.3Hz $F_{B_R}$")
    plt.plot(Time_OF[idx1:idx2],BPF_FBR[idx1:idx2],"-r",label="BPF 0.3-0.9Hz $F_{B_R}$")
    idx1 = np.searchsorted(Times_2,Times[i]);idx2 = np.searchsorted(Times_2,Times[i+1])
    plt.plot(Times_2[idx1:idx2],BPF_FBR_2[idx1:idx2],"--b",linewidth=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude Main Bearing force vector [kN]")
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"{}_{}.png".format(Times[i],Times[i+1]))
    plt.close()



out_dir = in_dir+"peak_peak_analysis/frequencies_all_times_HPF_3P/"
Times = np.arange(200,1300,100)
for i in np.arange(0,len(Times)-1):
    idx1 = np.searchsorted(Time_OF,Times[i]);idx2 = np.searchsorted(Time_OF,Times[i+1])
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[idx1:idx2],FBR[idx1:idx2],"-k",label="Total $F_{B_R}$")
    plt.plot(Time_OF[idx1:idx2],HPF_FBR[idx1:idx2]-600,"-b",label="HPF 1.5Hz $F_{B_R}$\noffset: -600kN")
    plt.plot(Time_OF[idx1:idx2],BPF_FBR[idx1:idx2],"-r",label="BPF 0.3-0.9Hz $F_{B_R}$")
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude Main Bearing force vector [kN]")
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"{}_{}.png".format(Times[i],Times[i+1]))
    plt.close()

zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]

BPF_FBR_2 = []
Times_2 = []
for i in np.arange(0,len(zero_crossings_index_BPF_FBR),2):
    idx = zero_crossings_index_BPF_FBR[i]
    BPF_FBR_2.append(BPF_FBR[idx]); Times_2.append(Time_OF[idx])

# plt.plot(Time_OF,LPF_1_FBR,"-b")
# plt.plot(Time_OF,BPF_FBR,"-r")
# plt.plot(Times_2,BPF_FBR_2,"-k")

f = interpolate.interp1d(Time_OF,LPF_1_FBR)
LPF_1_FBR_interp = f(Times_2)

idx = np.searchsorted(Times_2,Times_2[0]+20)
cc = []
for it in np.arange(0,len(Times_2)-idx):
    cc.append(correlation_coef(BPF_FBR_2[it:it+idx],LPF_1_FBR_interp[it:it+idx]))

out_dir = in_dir+"peak_peak_analysis/"
plt.rcParams.update({'font.size': 18})

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,BPF_FBR,"-r")
# plt.plot(Times_2,BPF_FBR_2,"-k")
# plt.xlabel("Time [s]")
# plt.ylabel("Magnitude Main Bearing force vector [kN]")
# plt.xlim([200,300])
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"BPF.png")
# plt.close()

fig,(ax,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax.plot(Times_2,LPF_1_FBR_interp,"-b",label="LPF 0.3Hz $F_{B_R}$")
ax.plot(Times_2,BPF_FBR_2,"-r",label="BPF 0.3-0.9Hz $F_{B_R}$")
fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(LPF_1_FBR_interp,BPF_FBR_2),2)))
fig.supxlabel("Time [s]")
ax.set_ylabel("Magnitude Main Bearing\nforce vector [kN]")
ax.legend()
ax.grid()
ax2.plot(Times_2[idx:],cc,"-k")
ax2.set_ylabel("local correlation\ncoefficient T= 20s")
ax2.grid()
plt.tight_layout()
plt.savefig(out_dir+"LPF_BPF.png")
plt.close()


# zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]

# HPF_FBR_2 = []
# Times_2 = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR),2):
#     idx = zero_crossings_index_HPF_FBR[i]
#     HPF_FBR_2.append(HPF_FBR[idx]); Times_2.append(Time_OF[idx])

# # plt.plot(Time_OF,LPF_1_FBR,"-b")
# # plt.plot(Time_OF,BPF_FBR,"-r")
# # plt.plot(Times_2,BPF_FBR_2,"-k")

# f = interpolate.interp1d(Time_OF,LPF_1_FBR)
# LPF_1_FBR_interp = f(Times_2)

# idx = np.searchsorted(Times_2,Times_2[0]+20)
# cc = []
# for it in np.arange(0,len(Times_2)-idx):
#     cc.append(correlation_coef(HPF_FBR_2[it:it+idx],LPF_1_FBR_interp[it:it+idx]))

# out_dir = in_dir+"peak_peak_analysis/"
# plt.rcParams.update({'font.size': 18})

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,HPF_FBR,"-r")
# plt.plot(Times_2,HPF_FBR_2,"-k")
# plt.xlabel("Time [s]")
# plt.ylabel("Magnitude Main Bearing force vector [kN]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"HPF.png")
# plt.close()

# fig,(ax,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
# ax.plot(Times_2,LPF_1_FBR_interp,"-b",label="LPF 0.3Hz $F_{B_R}$")
# ax.plot(Times_2,HPF_FBR_2,"-r",label="HPF 1.5Hz $F_{B_R}$")
# fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(LPF_1_FBR_interp,HPF_FBR_2),2)))
# fig.supxlabel("Time [s]")
# ax.set_ylabel("Magnitude Main Bearing\nforce vector [kN]")
# ax.legend()
# ax.grid()
# ax2.plot(Times_2[idx:],cc,"-k")
# ax2.set_ylabel("local correlation\ncoefficient T= 20s")
# ax2.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"LPF_HPF.png")
# plt.close()


# a = Dataset(in_dir+"Asymmetry_Dataset.nc")

# Time = np.array(a.variables["time"])
# Time = Time - Time[0]
# dt = Time[1] - Time[0]
# Time_steps = np.arange(0,len(Time))

# Time_start = 200; Time_start_idx = np.searchsorted(Time,Time_start)
# Time = Time[Time_start_idx:]

# A_high = np.array(a.variables["Area_high"][Time_start_idx:])
# A_low = np.array(a.variables["Area_low"][Time_start_idx:])

# fig,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(14,8),sharex=True)
# ax.plot(Times_2,LPF_1_FBR_interp,"-b",label="LPF 0.3Hz $F_{B_R}$")
# ax.plot(Times_2,BPF_FBR_2,"-r",label="BPF 0.3-0.9Hz $F_{B_R}$")
# fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(LPF_1_FBR_interp,BPF_FBR_2),2)))
# fig.supxlabel("Time [s]")
# ax.set_ylabel("Magnitude Main Bearing\nforce vector [kN]")
# ax.legend()
# ax.grid()
# ax2.plot(Times_2[7:-6],cc,"-k")
# ax2.set_ylabel("local correlation\ncoefficient T= 20s")
# ax2.grid()
# ax3.plot(Time,A_high,"-r")
# ax3.plot(Time,A_low,"-b")
# ax3.set_ylabel("Area [$m^2$]")
# ax3.grid()
# plt.tight_layout()
# plt.show()

# plt.plot(Time_OF,LPF_1_FBR/1079,"-g",label="LPF $F_B$")
# plt.plot(Time_OF,BPF_FBR/1079,"-r",label="BPF $F_B$")
# plt.plot(Time_OF,HPF_FBR/1079,"-b",label="HPF $F_B$")
# plt.ylabel("Magnitude Main Bearing force vector\nNormalized on rotor weight (1079kN) [-]")
# plt.xlabel("Time [s]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

dBPF_Theta_FB = np.array(dt_calc_theta(BPF_Theta_FB,dt))
dHPF_Theta_FB = np.array(dt_calc_theta(HPF_Theta_FB,dt))

# out_dir=in_dir+"peak_peak_analysis/"
# plt.rcParams['font.size'] = 16
# P,X=probability_dist(dBPF_Theta_FB)
# fig = plt.figure(figsize=(14,8))
# plt.plot(X,P,"k")
# plt.axvline(x=np.mean(dBPF_Theta_FB),linestyle="--",color="k")
# plt.axvline(x=np.mean(dBPF_Theta_FB)+np.std(dBPF_Theta_FB),linestyle="-.",color="k")
# plt.axvline(x=np.mean(dBPF_Theta_FB)-np.std(dBPF_Theta_FB),linestyle="-.",color="k")
# plt.xlabel("Band pass 0.3-0.9Hz filtered Angular velocity Main Bearing force vector [1/s]")
# plt.ylabel("Probability [-]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"PDF_dBPF_Theta_BPF.png")
# plt.close()

# P,X=probability_dist(dHPF_Theta_FB)
# fig = plt.figure(figsize=(14,8))
# plt.plot(X,P,"k")
# plt.axvline(x=np.mean(dHPF_Theta_FB),linestyle="--",color="k")
# plt.axvline(x=np.mean(dHPF_Theta_FB)+np.std(dHPF_Theta_FB),linestyle="-.",color="k")
# plt.axvline(x=np.mean(dHPF_Theta_FB)-np.std(dHPF_Theta_FB),linestyle="-.",color="k")
# plt.xlabel("High pass 1.5Hz filtered Angular velocity Main Bearing force vector [1/s]")
# plt.ylabel("Probability [-]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"PDF_dHPF_Theta_BPF.png")
# plt.close()


#Filtering FBR inc weight
LPF_1_FBR_weight = low_pass_filter(FBR_weight,0.3,dt)
LPF_2_FBR_weight = low_pass_filter(FBR_weight,0.9,dt)
LPF_3_FBR_weight = low_pass_filter(FBR_weight,1.5,dt)

HPF_FBR_weight = np.subtract(FBR_weight,LPF_3_FBR_weight)
HPF_FBR_weight = np.array(low_pass_filter(HPF_FBR_weight,40,dt))
BPF_FBR_weight = np.subtract(LPF_2_FBR_weight,LPF_1_FBR_weight)
dBPF_FBR_weight = np.array(dt_calc(BPF_FBR_weight,dt))
dHPF_FBR_weight = np.array(dt_calc(HPF_FBR_weight,dt))


LPF_1_Theta_FBR = low_pass_filter(Theta_FB_weight,0.3,dt)
LPF_2_Theta_FBR = low_pass_filter(Theta_FB_weight,0.9,dt)
LPF_3_Theta_FBR = low_pass_filter(Theta_FB_weight,1.5,dt)

HPF_Theta_FB_weight = np.subtract(Theta_FB_weight,LPF_3_Theta_FBR)
BPF_Theta_FB_weight = np.subtract(LPF_2_Theta_FBR,LPF_1_Theta_FBR)

# ax1.plot(Time_OF,LPF_1_FBR_weight,"-g")
# ax2.plot(Time_OF,BPF_FBR_weight,"-r")
# ax2.plot(Time_OF,HPF_FBR_weight,"-b")
# plt.show()



#Filtering MR
#OOPBM
MR = np.sqrt(np.add(np.square(RtAeroMys), np.square(RtAeroMzs)))

frq,PSD = temporal_spectra(MR,dt,Var="MR")
#plt.loglog(frq,PSD,"-b",label="Magnitude OOPBM vector [kN-m]")

LPF_1_MR = low_pass_filter(MR,0.3,dt)
LPF_2_MR = low_pass_filter(MR,0.9,dt)
LPF_3_MR = low_pass_filter(MR,1.5,dt)

HPF_MR = np.subtract(MR,LPF_3_MR)
HPF_MR = np.array(low_pass_filter(HPF_MR,40,dt))
BPF_MR = np.subtract(LPF_2_MR,LPF_1_MR)
dBPF_MR = np.array(dt_calc(BPF_MR,dt))
dHPF_MR = np.array(dt_calc(HPF_MR,dt))



# out_dir = in_dir+"peak_peak_analysis/"



# plt.rcParams['font.size'] = 16
# LPF_1_MR = np.subtract(LPF_1_MR,np.mean(LPF_1_MR))
# LPF_1_FBR = np.subtract(LPF_1_FBR,np.mean(LPF_1_FBR))


# fig=plt.figure(figsize=(14,8))
# ax=fig.add_subplot(111, label="1")
# ax2=fig.add_subplot(111, label="2", frame_on=False)

# ax.plot(Time_OF, LPF_1_MR, "-r",label="OOPBM vector [kN-m]")
# ax.plot(Time_OF,LPF_1_FBR,"-b",label="$F_B$ vector [kN]")
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Fluctuations LPF 0.3Hz Magnitude of vector")
# fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(LPF_1_MR,LPF_1_FBR),2)))
# ax.legend(loc="upper left",fontsize=12)
# ax.grid()
# ax.tick_params(axis='x')
# ax.tick_params(axis='y')

# P,X=probability_dist(LPF_1_FBR)
# ax2.plot(P, X,"-k")
# ax2.xaxis.tick_top()
# ax2.yaxis.tick_right()
# ax2.set_ylabel('Fluctuations LPF 0.3Hz Magnitude $F_B$ vector [kN]') 
# ax2.set_xlabel('Probability [-]')     
# ax2.grid()  
# ax2.xaxis.set_label_position('top') 
# ax2.yaxis.set_label_position('right') 
# ax2.tick_params(axis='x')
# ax2.tick_params(axis='y')
# ax2.invert_xaxis()
# plt.tight_layout()
# plt.savefig(out_dir+"cc_FBR_MR_LPF_1.png")
# plt.close()

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(MR,dt,Var="OOPBM")
# plt.loglog(frq,PSD,"-r")
# frq,PSD = temporal_spectra(FBR,dt,Var="FBR")
# plt.loglog(frq,PSD,"-b")
# plt.xlabel("frequency [Hz]")
# plt.ylabel("PSD")
# plt.legend(["Magnitude Out-of-plane bending moment vector[kN-m]","Magnitude Main Bearing force vector [kN]"])
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"spectra_FBR_MR.png")
# plt.close()


# fig=plt.figure(figsize=(14,8))
# ax=fig.add_subplot(111, label="1")
# ax2=fig.add_subplot(111, label="2", frame_on=False)

# ax.plot(Time_OF, BPF_MR, "-r",label="OOPBM vector [kN-m]")
# ax.plot(Time_OF,BPF_FBR,"-b",label="$F_B$ vector [kN]")
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Fluctuations BPF 0.3-0.9Hz Magnitude of vector")
# fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(BPF_MR,BPF_FBR),2)))
# ax.legend(loc="upper left",fontsize=12)
# ax.grid()
# ax.tick_params(axis='x')
# ax.tick_params(axis='y')

# P,X=probability_dist(BPF_FBR)
# ax2.plot(P, X,"-k")
# ax2.xaxis.tick_top()
# ax2.yaxis.tick_right()
# ax2.set_ylabel('Fluctuations BPF 0.3-0.9Hz Magnitude $F_B$ vector [kN]') 
# ax2.set_xlabel('Probability [-]')     
# ax2.grid()  
# ax2.xaxis.set_label_position('top') 
# ax2.yaxis.set_label_position('right') 
# ax2.tick_params(axis='x')
# ax2.tick_params(axis='y')
# ax2.invert_xaxis()
# plt.tight_layout()
# plt.savefig(out_dir+"cc_FBR_MR_BPF.png")
# plt.close()

# fig=plt.figure(figsize=(14,8))
# ax=fig.add_subplot(111, label="1")
# ax2=fig.add_subplot(111, label="2", frame_on=False)

# ax.plot(Time_OF, HPF_MR, "-r",label="OOPBM vector [kN-m]")
# ax.plot(Time_OF,HPF_FBR,"-b",label="$F_B$ vector [kN]")
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Fluctuations HPF 1.5Hz Magnitude of vector")
# fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(HPF_MR,HPF_FBR),2)))
# ax.legend(loc="upper left",fontsize=12)
# ax.grid()
# ax.tick_params(axis='x')
# ax.tick_params(axis='y')

# P,X=probability_dist(HPF_FBR)
# ax2.plot(P, X,"-k")
# ax2.xaxis.tick_top()
# ax2.yaxis.tick_right()
# ax2.set_ylabel('Fluctuations HPF 1.5Hz Magnitude $F_B$ vector [kN]') 
# ax2.set_xlabel('Probability [-]')     
# ax2.grid()  
# ax2.xaxis.set_label_position('top') 
# ax2.yaxis.set_label_position('right') 
# ax2.tick_params(axis='x')
# ax2.tick_params(axis='y')
# ax2.invert_xaxis()
# plt.tight_layout()
# plt.savefig(out_dir+"cc_FBR_MR_HPF.png")
# plt.close()

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF[:2600],HPF_FBR[:2600],"-b")
# plt.xlabel("Time [s]")
# plt.ylabel("Fluctuations HPF 1.5Hz Magnitude $F_B$ vector [kN]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"HPF_FBR_short.png")
# plt.close()


#print(moments(LPF_1_FBR),moments(BPF_FBR),moments(HPF_FBR))


#Asymmetry
group = df_OF.groups["63.0"]
Iy = np.array(group.variables["Iy"])
Iz = -np.array(group.variables["Iz"])

dt_sampling = Time_sampling[1] - Time_sampling[0]

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
frq,PSD = temporal_spectra(I,dt_sampling,Var="I")
# plt.loglog(frq,PSD,"-g",label="Magnitude Asymmetry vector [$m^4/s$]")

# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(out_dir+"spectra_FBR_MR_I.png")
# plt.close()

f = interpolate.interp1d(Time_sampling,Iy)
Iy = f(Time_OF)



f = interpolate.interp1d(Time_sampling,Iz)
Iz = f(Time_OF)

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
Theta_I = np.degrees(np.arctan2(Iz,Iy))
Theta_I = theta_360(Theta_I)
Theta_I = np.radians(np.array(Theta_I))

LPF_1_I = low_pass_filter(I,0.3,dt)
LPF_2_I = low_pass_filter(I,0.9,dt)
LPF_3_I = low_pass_filter(I,1.5,dt)

HPF_I = np.subtract(I,LPF_3_I)
BPF_I = np.subtract(LPF_2_I,LPF_1_I)

LPF_1_Theta_I = low_pass_filter(Theta_I,0.3,dt)
LPF_2_Theta_I = low_pass_filter(Theta_I,0.9,dt)
LPF_3_Theta_I = low_pass_filter(Theta_I,1.5,dt)

HPF_Theta_I = np.subtract(Theta_I,LPF_3_Theta_I)
BPF_Theta_I = np.subtract(LPF_2_Theta_I,LPF_1_Theta_I)


# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
# zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]

# arr = []
# inner_arr = [0,0]
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-1):
#     idx = zero_crossings_index_BPF_FBR[i]
#     idx_step = zero_crossings_index_BPF_FBR[i+1]
#     if inner_arr == [0,0] and abs(BPF_FBR[idx]) >= 1.5*np.std(BPF_FBR):
#         inner_arr[0] = idx

#     if inner_arr[0] != 0 and inner_arr[1] == 0 and abs(BPF_FBR[idx_step]) < 1.5*np.std(BPF_FBR):
#         inner_arr[1] = idx

#     if inner_arr[0] != 0 and inner_arr[1] != 0:
#         arr.append(inner_arr)
#         inner_arr = [0,0]


# i = 0
# for inner_arr in arr:
#     print(Time_OF[inner_arr[0]],Time_OF[inner_arr[1]])
#     for idx in np.arange(inner_arr[0],inner_arr[1]+1):
#         if idx in zero_crossings_index_HPF_FBR and abs(HPF_FBR[idx]) >= 1.5*np.std(HPF_FBR):
#             i+=1


#     print(i)
# print(i)

# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
# zero_crossings_index_BPF_Theta_FB = np.where(np.diff(np.sign(dBPF_Theta_FB)))[0]
# i = 0
# for idx in zero_crossings_index_BPF_FBR:
#     if idx in zero_crossings_index_BPF_Theta_FB:
#         i+=1

# print(i)

# zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]
# zero_crossings_index_HPF_Theta_FB = np.where(np.diff(np.sign(dHPF_Theta_FB)))[0]
# i = 0
# for idx in zero_crossings_index_HPF_FBR:
#     if idx in zero_crossings_index_HPF_Theta_FB:
#         i+=1

# print(i)

if dF_F_analysis == True:

    out_dir = in_dir+"peak_peak_analysis_weight/"

    #Magnitude
    zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR_weight)))[0]
    Time_zero_crossings_BPF_FBR = Time_OF[zero_crossings_index_BPF_FBR]
    FBR_zero_crossings_BPF_FBR = BPF_FBR_weight[zero_crossings_index_BPF_FBR]
    dFBR_zero_crossings_BPF_FBR = dBPF_FBR_weight[zero_crossings_index_BPF_FBR]

    zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR_weight)))[0]
    Time_zero_crossings_HPF_FBR = Time_OF[zero_crossings_index_HPF_FBR]
    FBR_zero_crossings_HPF_FBR = HPF_FBR_weight[zero_crossings_index_HPF_FBR]
    dFBR_zero_crossings_HPF_FBR = dHPF_FBR_weight[zero_crossings_index_HPF_FBR]

    # fig = plt.figure(figsize=(14,8))
    # plt.plot(Time_OF,BPF_FBR,"-b",label="Fluctuations BPF Magnitude $F_B$ vector [kN]")
    # plt.plot(Time_OF[:-1],dBPF_FBR,"-k",label="Derivative Fluctuations BPF Magnitude $F_B$ vector [kN/s]")
    # plt.plot(Time_zero_crossings_BPF_FBR,FBR_zero_crossings_BPF_FBR,"or",label="peaks in $F_B$")
    # plt.plot(Time_zero_crossings_BPF_FBR,dFBR_zero_crossings_BPF_FBR,"og",label="zero crossings")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Fluctuations BPF Magnitude $F_B$ vector [kN]")
    # plt.legend()
    # plt.grid()
    # plt.xlim([200,220])
    # plt.ylim([-3000,3000])
    # plt.title("20s period")
    # plt.tight_layout()
    # plt.savefig(out_dir+"BPF_FBR_short.png")
    # plt.close()


    #BPF calc
    dF_mag_BPF = []
    FBR_mag_BPF = []
    FB_mag_BPF = []
    dt_mag_BPF = []
    dTheta_mag_BPF = []
    Time_mag_BPF = []
    for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-1):

        it_1 = zero_crossings_index_BPF_FBR[i]
        it_2 = zero_crossings_index_BPF_FBR[i+1]

        Time_mag_BPF.append(Time_OF[it_1])

        dF_mag_BPF.append(BPF_FBR_weight[it_2] - BPF_FBR_weight[it_1])

        dt_mag_BPF.append(Time_OF[it_2] - Time_OF[it_1])

        dTheta_mag_BPF.append(BPF_Theta_FB_weight[it_2]-BPF_Theta_FB_weight[it_1])

        FBR_mag_BPF.append(FBR_weight[it_1])

        FB_mag_BPF.append(FB_weight[it_1])


    a = Dataset(in_dir+"Asymmetry_Dataset.nc")

    Time = np.array(a.variables["time"])
    Time = Time - Time[0]
    dt = Time[1] - Time[0]
    Time_steps = np.arange(0,len(Time))

    Time_start = 200; Time_start_idx = np.searchsorted(Time,Time_start)
    Time = Time[Time_start_idx:]
    Time_steps = np.arange(0,len(Time))
    cutoff = 1/60

    Ux_high = np.array(a.variables["Ux_high"][Time_start_idx:])
    Ux_low = np.array(a.variables["Ux_low"][Time_start_idx:])
    Ux_int = np.array(a.variables["Ux_int"][Time_start_idx:])

    # #remove all zeros from ux_low and time
    
    # fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)

    # ax1.plot(Time,Ux_int,"-b")
    # ax1.axhline(y=np.mean(Ux_low),linestyle="--",color="r")
    # ax1.set_ylabel("Average streamwise velocity [m/s]",fontsize=14)
    # ax1.set_xlabel("Time [s]",fontsize=16)
    # ax1.grid()

    # ax2.plot(Time_mag_BPF,dF_mag_BPF,"-k")
    # ax2.grid()

    # plt.tight_layout()
    # plt.show()

    plt.rcParams.update({'font.size': 18})


    fig = plt.figure(figsize=(14,8))
    plt.scatter(dt_mag_BPF,dF_mag_BPF,c=FBR_mag_BPF,cmap="viridis")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN]$")
    plt.colorbar()
    plt.title("BPF $F_{B_R}$ inc weight")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_dt_BPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(dTheta_mag_BPF,dF_mag_BPF,c=FBR_mag_BPF,cmap="viridis")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.colorbar()
    plt.title("BPF $F_{B_R}$ inc weight")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_dtheta_BPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(FBR_mag_BPF,dF_mag_BPF)
    plt.xlabel("Initial $F_{B_R}$ inc weight [kN]")
    plt.ylabel("$dF$ [kN]")
    plt.title("BPF $F_{B_R}$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_F_BPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(dt_mag_BPF,np.true_divide(dF_mag_BPF,1079),c=FB_mag_BPF,cmap="viridis")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("Normalized $dF [-]$")
    plt.colorbar()
    plt.title("BPF $F_B$ inc weight\nNormalized on rotor weight (1079kN)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_dt_BPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(dTheta_mag_BPF,np.true_divide(dF_mag_BPF,1079),c=FB_mag_BPF,cmap="viridis")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("Normalized $dF [-]$")
    plt.colorbar()
    plt.title("BPF $F_B$ inc weight\nNormalized on rotor weight (1079kN)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_dtheta_BPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(FB_mag_BPF,np.true_divide(dF_mag_BPF,1079))
    plt.xlabel("Initial $F_B$ inc weight [kN]")
    plt.ylabel("Normalized $dF$ [-]")
    plt.title("BPF $F_B$\nNormalized on rotor weight (1079kN)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_F_BPF.png")
    plt.close()

    dF_mag_BPF_threshold = []
    dt_mag_BPF_threshold = []
    dTheta_mag_BPF_threshold = []
    Time_mag_BPF_threshold = []
    FBR_mag_BPF_threshold = []
    for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-1):

        it_1 = zero_crossings_index_BPF_FBR[i]
        it_2 = zero_crossings_index_BPF_FBR[i+1]

        dF = BPF_FBR_weight[it_2] - BPF_FBR_weight[it_1]

        dt = Time_OF[it_2] - Time_OF[it_1]

        dTheta = BPF_Theta_FB_weight[it_2]-BPF_Theta_FB_weight[it_1]

        FBR_it = FBR_weight[it_1]

        if FBR_it > np.mean(FBR_weight)+1.5*np.std(FBR_weight) or FBR_it < np.mean(FBR_weight)-1.5*np.std(FBR_weight):

            Time_mag_BPF_threshold.append(Time_OF[it_1])

            dF_mag_BPF_threshold.append(dF); dt_mag_BPF_threshold.append(dt); dTheta_mag_BPF_threshold.append(dTheta); FBR_mag_BPF_threshold.append(FBR_it)


    fig = plt.figure(figsize=(14,8))
    plt.scatter(dt_mag_BPF_threshold,dF_mag_BPF_threshold,c=FBR_mag_BPF_threshold,cmap="viridis")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN]$")
    plt.title("BPF $F_{B_R}$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_dt_BPF_threshold.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(dTheta_mag_BPF_threshold,dF_mag_BPF_threshold,c=FBR_mag_BPF_threshold,cmap="viridis")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("BPF $F_{B_R}$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_dtheta_BPF_threshold.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(FBR_mag_BPF_threshold,dF_mag_BPF_threshold)
    plt.xlabel("Initial $F_{B_R}$ inc weight [kN]")
    plt.ylabel("$dF$ [kN]")
    plt.title("BPF $F_{B_R}$ inc weight threshold on 1.5x standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_F_BPF_threshold.png")
    plt.close()

    dF_mag_BPF_threshold = []
    dt_mag_BPF_threshold = []
    dTheta_mag_BPF_threshold = []
    Time_mag_BPF_threshold = []
    FB_mag_BPF_threshold = []
    for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-1):

        it_1 = zero_crossings_index_BPF_FBR[i]
        it_2 = zero_crossings_index_BPF_FBR[i+1]

        dF = BPF_FBR_weight[it_2] - BPF_FBR_weight[it_1]

        dt = Time_OF[it_2] - Time_OF[it_1]

        dTheta = BPF_Theta_FB_weight[it_2]-BPF_Theta_FB_weight[it_1]

        FB_it = FB_weight[it_1]

        if FB_it > np.mean(FB_weight)+1.5*np.std(FB_weight) or FB_it < np.mean(FB_weight)-1.5*np.std(FB_weight):

            Time_mag_BPF_threshold.append(Time_OF[it_1])

            dF_mag_BPF_threshold.append(dF); dt_mag_BPF_threshold.append(dt); dTheta_mag_BPF_threshold.append(dTheta); FB_mag_BPF_threshold.append(FB_it)


    fig = plt.figure(figsize=(14,8))
    plt.scatter(dt_mag_BPF_threshold,dF_mag_BPF_threshold,c=FB_mag_BPF_threshold,cmap="viridis")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN]$")
    plt.title("BPF $F_B$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_dt_BPF_threshold.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(dTheta_mag_BPF_threshold,dF_mag_BPF_threshold,c=FB_mag_BPF_threshold,cmap="viridis")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("BPF $F_B$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_dtheta_BPF_threshold.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(FB_mag_BPF_threshold,dF_mag_BPF_threshold)
    plt.xlabel("Initial $F_B$ inc weight [kN]")
    plt.ylabel("$dF$ [kN]")
    plt.title("BPF $F_B$ inc weight threshold on 1.5x standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_F_BPF_threshold.png")
    plt.close()

    #HPF calc
    dF_mag_HPF = []
    dt_mag_HPF = []
    dTheta_mag_HPF = []
    Time_mag_HPF = []
    FBR_mag_HPF = []
    FB_mag_HPF = []
    for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

        it_1 = zero_crossings_index_HPF_FBR[i]
        it_2 = zero_crossings_index_HPF_FBR[i+1]

        Time_mag_HPF.append(Time_OF[it_1])

        dF_mag_HPF.append(HPF_FBR[it_2] - HPF_FBR[it_1])

        dt_mag_HPF.append(Time_OF[it_2] - Time_OF[it_1])

        dTheta_mag_HPF.append(HPF_Theta_FB[it_2]-HPF_Theta_FB[it_1])

        FBR_mag_HPF.append(FBR_weight[it_1])

        FB_mag_HPF.append(FB_weight[it_1])


    
    # out_dir = in_dir+"peak_peak_analysis/dF_all_times/"
    # Times = np.arange(200,1220,20)
    # for i in np.arange(0,len(Times)-1):
    #     fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
    #     ax1.plot(Time_mag_BPF,dF_mag_BPF)
    #     ax1.axhline(y=1.5*np.std(dF_mag_BPF),linestyle="--",color="k")
    #     ax1.axhline(y=-1.5*np.std(dF_mag_BPF),linestyle="--",color="k")
    #     ax2.plot(Time_mag_HPF,dF_mag_HPF)
    #     ax2.axhline(y=1.5*np.std(dF_mag_HPF),linestyle="--",color="k")
    #     ax2.axhline(y=-1.5*np.std(dF_mag_HPF),linestyle="--",color="k")
    #     fig.supxlabel("Times [s]")
    #     ax1.set_ylabel("dF BPF Magnitude\nAerodynamic Main bearing force vector [kN]",fontsize=12)
    #     ax1.grid()
    #     ax1.set_xlim([Times[i],Times[i+1]])
    #     ax2.set_ylabel("dF HPF Magnitude\nAerodynamic Main bearing force vector [kN]",fontsize=12)
    #     ax2.grid()
    #     ax2.set_xlim([Times[i],Times[i+1]])
    #     plt.tight_layout()
    #     plt.savefig(out_dir+"{}_{}.png".format(Times[i],Times[i+1]))
    #     plt.close()


    fig = plt.figure(figsize=(14,8))
    plt.scatter(dt_mag_HPF,dF_mag_HPF,c=FBR_mag_HPF,cmap="viridis")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN]$")
    plt.title("HPF $F_{B_R}$ inc weight")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_dt_HPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(dTheta_mag_HPF,dF_mag_HPF,c=FBR_mag_HPF,cmap="viridis")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("HPF $F_{B_R}$ inc weight")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_dtheta_HPF.png")
    plt.close()


    fig = plt.figure(figsize=(14,8))
    plt.scatter(FBR_mag_HPF,dF_mag_HPF)
    plt.xlabel("Initial $F_{B_R}$ inc weight [kN]")
    plt.ylabel("$dF$ [kN]")
    plt.title("HPF $F_{B_R}$ inc weight")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_F_HPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(dt_mag_HPF,np.true_divide(dF_mag_HPF,1079),c=FB_mag_HPF,cmap="viridis")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("Normalized $dF [-]$")
    plt.title("HPF $F_B$ inc weight\nNormalized on rotor weight (1079kN)")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_dt_HPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(dTheta_mag_HPF,np.true_divide(dF_mag_HPF,1079),c=FB_mag_HPF,cmap="viridis")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("Normalized $dF [-]$")
    plt.title("HPF $F_B$ inc weight\nNormalized on rotor weight (1079kN)")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_dtheta_HPF.png")
    plt.close()


    fig = plt.figure(figsize=(14,8))
    plt.scatter(FB_mag_HPF,np.true_divide(dF_mag_HPF,1079))
    plt.xlabel("Initial $F_B$ inc weight [kN]")
    plt.ylabel("Normalized $dF$ [-]")
    plt.title("HPF $F_B$ inc weight\nNormalized on rotor weight (1079kN)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_F_HPF.png")
    plt.close()

    #HPF threshold calc
    dF_mag_HPF_threshold = []
    dt_mag_HPF_threshold = []
    dTheta_mag_HPF_threshold = []
    Time_mag_HPF_threshold = []
    FBR_mag_HPF_threshold = []
    for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

        it_1 = zero_crossings_index_HPF_FBR[i]
        it_2 = zero_crossings_index_HPF_FBR[i+1]

        dF = HPF_FBR_weight[it_1] - HPF_FBR_weight[it_2]

        dt = Time_OF[it_2] - Time_OF[it_1]

        dTheta = HPF_Theta_FB_weight[it_2]-HPF_Theta_FB_weight[it_1]

        FBR_it = FBR_weight[it_1]

        if FBR_it > np.mean(FBR_weight)+1.5*np.std(FBR_weight) or FBR_it < np.mean(FBR_weight)-1.5*np.std(FBR_weight):

            Time_mag_HPF_threshold.append(Time_OF[it_1])

            dF_mag_HPF_threshold.append(dF); dt_mag_HPF_threshold.append(dt); dTheta_mag_HPF_threshold.append(dTheta); FBR_mag_HPF_threshold.append(FBR_it)


    fig = plt.figure(figsize=(14,8))
    plt.scatter(dt_mag_HPF_threshold,dF_mag_HPF_threshold,c=FBR_mag_HPF_threshold,cmap="viridis")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN]$")
    plt.title("HPF $F_{B_R}$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_dt_HPF_threshold.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(dTheta_mag_HPF_threshold,dF_mag_HPF_threshold,c=FBR_mag_HPF_threshold,cmap="viridis")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("HPF $F_{B_R}$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_dtheta_HPF_threshold.png")
    plt.close()


    fig = plt.figure(figsize=(14,8))
    plt.scatter(FBR_mag_HPF_threshold,dF_mag_HPF_threshold)
    plt.xlabel("Initial $F_{B_R}$ inc weight [kN]")
    plt.ylabel("$dF$ [kN]")
    plt.title("HPF $F_{B_R}$ inc weight threshold on 1.5x standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_dF_F_HPF_threshold.png")
    plt.close()

    dF_mag_HPF_threshold = []
    dt_mag_HPF_threshold = []
    dTheta_mag_HPF_threshold = []
    Time_mag_HPF_threshold = []
    FB_mag_HPF_threshold = []
    for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

        it_1 = zero_crossings_index_HPF_FBR[i]
        it_2 = zero_crossings_index_HPF_FBR[i+1]

        dF = HPF_FBR_weight[it_1] - HPF_FBR_weight[it_2]

        dt = Time_OF[it_2] - Time_OF[it_1]

        dTheta = HPF_Theta_FB_weight[it_2]-HPF_Theta_FB_weight[it_1]

        FB_it = FB_weight[it_1]

        if FB_it > np.mean(FB_weight)+1.5*np.std(FB_weight) or FB_it < np.mean(FB_weight)-1.5*np.std(FB_weight):

            Time_mag_HPF_threshold.append(Time_OF[it_1])

            dF_mag_HPF_threshold.append(dF); dt_mag_HPF_threshold.append(dt); dTheta_mag_HPF_threshold.append(dTheta); FB_mag_HPF_threshold.append(FB_it)


    fig = plt.figure(figsize=(14,8))
    plt.scatter(dt_mag_HPF_threshold,dF_mag_HPF_threshold,c=FB_mag_HPF_threshold,cmap="viridis")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN]$")
    plt.title("HPF $F_B$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_dt_HPF_threshold.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(dTheta_mag_HPF_threshold,np.true_divide(dF_mag_HPF_threshold,1079),c=FB_mag_HPF_threshold,cmap="viridis")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("Normalized $dF [kN]$")
    plt.title("HPF $F_B$ Threshold on 1.5x standard deviation\nNormalized on rotor weight (1079kN)")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_dtheta_HPF_threshold.png")
    plt.close()


    fig = plt.figure(figsize=(14,8))
    plt.scatter(FB_mag_HPF_threshold,np.true_divide(dF_mag_HPF_threshold,1079))
    plt.xlabel("Initial $F_B$ inc weight [kN]")
    plt.ylabel("Normalized $dF$ [-]")
    plt.title("HPF $F_B$ inc weight threshold on 1.5x standard deviation\nNormalized on rotor weight (1079kN)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FB_dF_F_HPF_threshold.png")
    plt.close()


if dF_dt_analysis == True:

    out_dir = in_dir+"peak_peak_analysis/"

    #Magnitude
    zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
    Time_zero_crossings_BPF_FBR = Time_OF[zero_crossings_index_BPF_FBR]
    FBR_zero_crossings_BPF_FBR = BPF_FBR[zero_crossings_index_BPF_FBR]
    dFBR_zero_crossings_BPF_FBR = dBPF_FBR[zero_crossings_index_BPF_FBR]

    zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]
    Time_zero_crossings_HPF_FBR = Time_OF[zero_crossings_index_HPF_FBR]
    FBR_zero_crossings_HPF_FBR = HPF_FBR[zero_crossings_index_HPF_FBR]
    dFBR_zero_crossings_HPF_FBR = dHPF_FBR[zero_crossings_index_HPF_FBR]

    # fig = plt.figure(figsize=(14,8))
    # plt.plot(Time_OF,BPF_FBR,"-b",label="Fluctuations BPF Magnitude $F_B$ vector [kN]")
    # plt.plot(Time_OF[:-1],dBPF_FBR,"-k",label="Derivative Fluctuations BPF Magnitude $F_B$ vector [kN/s]")
    # plt.plot(Time_zero_crossings_BPF_FBR,FBR_zero_crossings_BPF_FBR,"or",label="peaks in $F_B$")
    # plt.plot(Time_zero_crossings_BPF_FBR,dFBR_zero_crossings_BPF_FBR,"og",label="zero crossings")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Fluctuations BPF Magnitude $F_B$ vector [kN]")
    # plt.legend()
    # plt.grid()
    # plt.xlim([200,220])
    # plt.ylim([-3000,3000])
    # plt.title("20s period")
    # plt.tight_layout()
    # plt.savefig(out_dir+"BPF_FBR_short.png")
    # plt.close()


    #BPF calc
    dF_mag_BPF = []
    dt_mag_BPF = []
    dTheta_mag_BPF = []
    Time_mag_BPF = []
    for i in np.arange(1,len(zero_crossings_index_BPF_FBR)-1):

        it_0 = zero_crossings_index_BPF_FBR[i-1]
        it_1 = zero_crossings_index_BPF_FBR[i]
        it_2 = zero_crossings_index_BPF_FBR[i+1]

        Time_mag_BPF.append(Time_OF[it_1])

        dF_left = abs(BPF_FBR[it_1] - BPF_FBR[it_0])
        dF_right = abs(BPF_FBR[it_1] - BPF_FBR[it_2])

        dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

        if dF_left > dF_right:
            dF_mag_BPF.append(dF_left); dt_mag_BPF.append(dt_left); dTheta_mag_BPF.append(abs(BPF_Theta_FB[it_1]-BPF_Theta_FB[it_0]))
        else:
            dF_mag_BPF.append(dF_right); dt_mag_BPF.append(dt_right); dTheta_mag_BPF.append(abs(BPF_Theta_FB[it_2]-BPF_Theta_FB[it_1]))


    print(len(Time_mag_BPF))
    plt.rcParams.update({'font.size': 18})

    fig = plt.figure(figsize=(14,8))
    plt.plot(dt_mag_BPF,dF_mag_BPF,"ob")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN]$")
    plt.title("BPF $F_B$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_BPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(dTheta_mag_BPF,dF_mag_BPF,"ob")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("BPF $F_B$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dtheta_BPF.png")
    plt.close()

    fig=plt.figure(figsize=(26,15),constrained_layout=True)
    ax=fig.add_subplot(311, label="1")
    ax2=fig.add_subplot(311, label="2", frame_on=False)
    ax3=fig.add_subplot(312)
    ax4=fig.add_subplot(312,frame_on=False)
    ax5=fig.add_subplot(313)
    ax6=fig.add_subplot(313,frame_on=False)

    ax.plot(Time_mag_BPF, np.true_divide(dF_mag_BPF,1079), "ob")
    ax.set_title("$dF$ BPF 0.3-0.9Hz Magnitude of $F_B$ vector\nNormalized on rotor weight (1079kN) [-]",pad=75)
    ax.grid()
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(np.true_divide(dF_mag_BPF,1079))
    ax2.plot(P, X,"-k")
    ax2.axvline(x=0.0,color="k")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()   
    ax2.grid()  
    ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.invert_xaxis()

    ax3.plot(Time_mag_BPF,dt_mag_BPF,"ob")
    ax3.set_title("$dt$ BPF 0.3-0.9Hz Magnitude of $F_B$ vector [s]",pad=50)
    ax.grid()
    ax3.tick_params(axis='x')
    ax3.tick_params(axis='y')

    P,X=probability_dist(dt_mag_BPF)
    ax4.plot(P, X,"-k")
    ax4.axvline(x=0.0,color="k")
    ax4.xaxis.tick_top()
    ax4.yaxis.tick_right()
    ax4.grid()  
    ax4.xaxis.set_label_position('top') 
    ax4.yaxis.set_label_position('right') 
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')
    ax4.invert_xaxis()


    ax5.plot(Time_mag_BPF,dTheta_mag_BPF,"ob")
    ax5.set_title("$d\\theta$ BPF 0.3-0.9Hz Magnitude of $F_B$ vector [deg]",pad=50)
    ax5.grid()
    ax5.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dTheta_mag_BPF)
    ax6.plot(P, X,"-k")
    ax6.axvline(x=0.0,color="k")
    ax6.xaxis.tick_top()
    ax6.yaxis.tick_right()   
    ax6.grid()  
    ax6.xaxis.set_label_position('top') 
    ax6.yaxis.set_label_position('right') 
    ax6.tick_params(axis='x')
    ax6.tick_params(axis='y')
    ax6.invert_xaxis()
    fig.supxlabel("Time [s]")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(out_dir+"dF_dt_dtheta_time_BPF.png")
    plt.close()

    print(moments(np.true_divide(dF_mag_BPF,1079)),moments(dt_mag_BPF),moments(dTheta_mag_BPF))

    #BPF threshold calc
    perc_BPF = []
    for phi in np.linspace(np.mean(dF_mag_BPF),np.mean(dF_mag_BPF)+2*np.std(dF_mag_BPF),10):
        Time_mag_BPF_threshold = []
        for i in np.arange(1,len(zero_crossings_index_BPF_FBR)-1):

            it_0 = zero_crossings_index_BPF_FBR[i-1]
            it_1 = zero_crossings_index_BPF_FBR[i]
            it_2 = zero_crossings_index_BPF_FBR[i+1]

            dF_left = abs(BPF_FBR[it_1] - BPF_FBR[it_0])
            dF_right = abs(BPF_FBR[it_1] - BPF_FBR[it_2])

            dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

            if dF_left > phi or dF_right > phi:
                Time_mag_BPF_threshold.append(Time_OF[it_1])


        perc_BPF.append((len(Time_mag_BPF_threshold)/len(Time_mag_BPF))*100)

    dF_mag_BPF_threshold = []
    dt_mag_BPF_threshold = []
    dTheta_mag_BPF_threshold = []
    Time_mag_BPF_threshold = []
    for i in np.arange(1,len(zero_crossings_index_BPF_FBR)-1):

        it_0 = zero_crossings_index_BPF_FBR[i-1]
        it_1 = zero_crossings_index_BPF_FBR[i]
        it_2 = zero_crossings_index_BPF_FBR[i+1]

        dF_left = abs(BPF_FBR[it_1] - BPF_FBR[it_0])
        dF_right = abs(BPF_FBR[it_1] - BPF_FBR[it_2])

        dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

        if dF_left > np.mean(dF_mag_BPF)+1.5*np.std(dF_mag_BPF) or dF_right > np.mean(dF_mag_BPF)+1.5*np.std(dF_mag_BPF):

            Time_mag_BPF_threshold.append(Time_OF[it_1])

            if dF_left > dF_right:
                dF_mag_BPF_threshold.append(dF_left/1079); dt_mag_BPF_threshold.append(dt_left); dTheta_mag_BPF_threshold.append(abs(BPF_Theta_FB[it_1]-BPF_Theta_FB[it_0]))
            else:
                dF_mag_BPF_threshold.append(dF_right/1079); dt_mag_BPF_threshold.append(dt_right); dTheta_mag_BPF_threshold.append(abs(BPF_Theta_FB[it_2]-BPF_Theta_FB[it_1]))

    print((len(Time_mag_BPF_threshold)/len(Time_mag_BPF))*100)
    fig = plt.figure(figsize=(14,8))
    plt.plot(dt_mag_BPF_threshold,dF_mag_BPF_threshold,"ob")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN]$")
    plt.title("BPF $F_B$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_BPF_threshold.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(dTheta_mag_BPF_threshold,dF_mag_BPF_threshold,"ob")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("BPF $F_B$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dtheta_BPF_threshold.png")
    plt.close()

    fig=plt.figure(figsize=(26,15))
    ax=fig.add_subplot(311, label="1")
    ax2=fig.add_subplot(311, label="2", frame_on=False)
    ax3=fig.add_subplot(312)
    ax4=fig.add_subplot(312,frame_on=False)
    ax5=fig.add_subplot(313)
    ax6=fig.add_subplot(313,frame_on=False)

    ax.plot(Time_mag_BPF_threshold, dF_mag_BPF_threshold, "ob")
    ax.set_title("$dF$ BPF 0.3-0.9Hz Magnitude of $F_B$ vector\nNormalized on rotor weight (1079kN) [-]",pad=75)
    ax.grid()
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dF_mag_BPF_threshold)
    ax2.plot(P, X,"-k")
    ax2.axvline(x=0.0,color="k")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.grid()  
    ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.invert_xaxis()

    ax3.plot(Time_mag_BPF_threshold,dt_mag_BPF_threshold,"ob")
    ax3.set_title("$dt$ BPF 0.3-0.9Hz Magnitude of $F_B$ vector [s]",pad=50)
    ax.grid()
    ax3.tick_params(axis='x')
    ax3.tick_params(axis='y')

    P,X=probability_dist(dt_mag_BPF_threshold)
    ax4.plot(P, X,"-k")
    ax4.axvline(x=0.0,color="k")
    ax4.xaxis.tick_top()
    ax4.yaxis.tick_right()     
    ax4.grid()  
    ax4.xaxis.set_label_position('top') 
    ax4.yaxis.set_label_position('right') 
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')
    ax4.invert_xaxis()


    ax5.plot(Time_mag_BPF_threshold,dTheta_mag_BPF_threshold,"ob")
    ax5.set_title("$d\\theta$ BPF 0.3-0.9Hz Magnitude of $F_B$ vector [deg]",pad=50)
    ax5.grid()
    ax5.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dTheta_mag_BPF_threshold)
    ax6.plot(P, X,"-k")
    ax6.axvline(x=0.0,color="k")
    ax6.xaxis.tick_top()
    ax6.yaxis.tick_right() 
    ax6.grid()  
    ax6.xaxis.set_label_position('top') 
    ax6.yaxis.set_label_position('right') 
    ax6.tick_params(axis='x')
    ax6.tick_params(axis='y')
    ax6.invert_xaxis()
    fig.supxlabel("Time [s]")
    fig.suptitle("Threshold on 1.5x standard deviation $dF$")
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_dtheta_BPF_threshold_time.png")
    plt.close()

    print(moments(dF_mag_BPF_threshold),moments(dt_mag_BPF_threshold),moments(dTheta_mag_BPF_threshold))

    #HPF calc
    dF_mag_HPF = []
    dt_mag_HPF = []
    dTheta_mag_HPF = []
    Time_mag_HPF = []
    for i in np.arange(1,len(zero_crossings_index_HPF_FBR)-1):

        it_0 = zero_crossings_index_HPF_FBR[i-1]
        it_1 = zero_crossings_index_HPF_FBR[i]
        it_2 = zero_crossings_index_HPF_FBR[i+1]

        Time_mag_HPF.append(Time_OF[it_1])

        dF_left = abs(HPF_FBR[it_1] - HPF_FBR[it_0])
        dF_right = abs(HPF_FBR[it_1] - HPF_FBR[it_2])

        dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

        if dF_left > dF_right:
            dF_mag_HPF.append(dF_left); dt_mag_HPF.append(dt_left); dTheta_mag_HPF.append(abs(HPF_Theta_FB[it_1]-HPF_Theta_FB[it_0]))
        else:
            dF_mag_HPF.append(dF_right); dt_mag_HPF.append(dt_right); dTheta_mag_HPF.append(abs(HPF_Theta_FB[it_2]-HPF_Theta_FB[it_1]))


    print(len(Time_mag_HPF))

    fig = plt.figure(figsize=(14,8))
    plt.plot(dt_mag_HPF,dF_mag_HPF,"ob")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN]$")
    plt.title("HPF $F_B$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_HPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(dTheta_mag_HPF,dF_mag_HPF,"ob")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("HPF $F_B$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dtheta_HPF.png")
    plt.close()


    fig=plt.figure(figsize=(26,15))
    ax=fig.add_subplot(311, label="1")
    ax2=fig.add_subplot(311, label="2", frame_on=False)
    ax3=fig.add_subplot(312)
    ax4=fig.add_subplot(312,frame_on=False)
    ax5=fig.add_subplot(313)
    ax6=fig.add_subplot(313,frame_on=False)

    ax.plot(Time_mag_HPF, np.true_divide(dF_mag_HPF,1079), "ob")
    ax.set_title("$dF$ HPF 1.5Hz Magnitude of $F_B$ vector\nNormalized on rotor weight (1079kN) [-]",pad=75)
    ax.grid()
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(np.true_divide(dF_mag_HPF,1079))
    ax2.plot(P, X,"-k")
    ax2.axvline(x=0.0,color="k")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.grid()  
    ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.invert_xaxis()

    ax3.plot(Time_mag_HPF,dt_mag_HPF,"ob")
    ax3.set_title("$dt$ HPF 1.5Hz Magnitude of $F_B$ vector [s]",pad=50)
    ax3.grid()
    ax3.tick_params(axis='x')
    ax3.tick_params(axis='y')

    P,X=probability_dist(dt_mag_HPF)
    ax4.plot(P, X,"-k")
    ax4.axvline(x=0.0,color="k")
    ax4.xaxis.tick_top()
    ax4.yaxis.tick_right()   
    ax4.grid()  
    ax4.xaxis.set_label_position('top') 
    ax4.yaxis.set_label_position('right') 
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')
    ax4.invert_xaxis()


    ax5.plot(Time_mag_HPF,dTheta_mag_HPF,"ob")
    ax5.set_title("$d\\theta$ HPF 1.5Hz Magnitude of $F_B$ vector [deg]",pad=50)
    ax5.grid()
    ax5.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dTheta_mag_HPF)
    ax6.plot(P, X,"-k")
    ax6.axvline(x=0.0,color="k")
    ax6.xaxis.tick_top()
    ax6.yaxis.tick_right()     
    ax6.grid()  
    ax6.xaxis.set_label_position('top') 
    ax6.yaxis.set_label_position('right') 
    ax6.tick_params(axis='x')
    ax6.tick_params(axis='y')
    ax6.invert_xaxis()
    fig.supxlabel("Time [s]")
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_dtheta_time_HPF.png")
    plt.close()

    print(moments(np.true_divide(dF_mag_HPF,1079)),moments(dt_mag_HPF),moments(dTheta_mag_HPF))

    #HPF threshold calc
    perc_HPF = []
    for phi in np.linspace(np.mean(dF_mag_HPF),np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF),10):
        Time_mag_HPF_threshold = []
        for i in np.arange(1,len(zero_crossings_index_HPF_FBR)-1):

            it_0 = zero_crossings_index_HPF_FBR[i-1]
            it_1 = zero_crossings_index_HPF_FBR[i]
            it_2 = zero_crossings_index_HPF_FBR[i+1]

            dF_left = abs(HPF_FBR[it_1] - HPF_FBR[it_0])
            dF_right = abs(HPF_FBR[it_1] - HPF_FBR[it_2])

            dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

            if dF_left > phi or dF_right > phi:
                Time_mag_HPF_threshold.append(Time_OF[it_1])


        perc_HPF.append((len(Time_mag_HPF_threshold)/len(Time_mag_HPF))*100)

    X = np.linspace(np.mean(dF_mag_BPF),np.mean(dF_mag_BPF)+2*np.std(dF_mag_BPF),10)
    fig = plt.figure(figsize=(14,8))
    plt.plot(X,perc_BPF,"-ob",label="BPF")
    plt.xlabel("Threshold value $F_B$ [kN]")
    plt.ylabel("Percentage of peaks in signal [%]")

    X = np.linspace(np.mean(dF_mag_HPF),np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF),10)
    plt.plot(X,perc_HPF,"-+r",label="HPF signal")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"perc_FBR_BPF_HPF.png")
    plt.close()

    dF_mag_HPF_threshold = []
    dt_mag_HPF_threshold = []
    dTheta_mag_HPF_threshold = []
    Time_mag_HPF_threshold = []
    for i in np.arange(1,len(zero_crossings_index_HPF_FBR)-1):

        it_0 = zero_crossings_index_HPF_FBR[i-1]
        it_1 = zero_crossings_index_HPF_FBR[i]
        it_2 = zero_crossings_index_HPF_FBR[i+1]

        dF_left = abs(HPF_FBR[it_1] - HPF_FBR[it_0])
        dF_right = abs(HPF_FBR[it_1] - HPF_FBR[it_2])

        dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

        if dF_left > np.mean(dF_mag_HPF)+1.5*np.std(dF_mag_HPF) or dF_right > np.mean(dF_mag_HPF)+1.5*np.std(dF_mag_HPF):

            Time_mag_HPF_threshold.append(Time_OF[it_1])

            if dF_left > dF_right:
                dF_mag_HPF_threshold.append(dF_left/1079); dt_mag_HPF_threshold.append(dt_left); dTheta_mag_HPF_threshold.append(abs(HPF_Theta_FB[it_1]-HPF_Theta_FB[it_0]))
            else:
                dF_mag_HPF_threshold.append(dF_right/1079); dt_mag_HPF_threshold.append(dt_right); dTheta_mag_HPF_threshold.append(abs(HPF_Theta_FB[it_2]-HPF_Theta_FB[it_1]))

    print((len(Time_mag_HPF_threshold)/len(Time_mag_HPF))*100)

    fig = plt.figure(figsize=(14,8))
    plt.plot(dt_mag_HPF_threshold,dF_mag_HPF_threshold,"ob")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN]$")
    plt.title("HPF $F_B$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_HPF_threshold.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(dTheta_mag_HPF_threshold,dF_mag_HPF_threshold,"ob")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("HPF $F_B$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dtheta_HPF_threshold.png")
    plt.close()


    fig=plt.figure(figsize=(26,15))
    ax=fig.add_subplot(311, label="1")
    ax2=fig.add_subplot(311, label="2", frame_on=False)
    ax3=fig.add_subplot(312)
    ax4=fig.add_subplot(312,frame_on=False)
    ax5=fig.add_subplot(313)
    ax6=fig.add_subplot(313,frame_on=False)

    ax.plot(Time_mag_HPF_threshold, dF_mag_HPF_threshold, "ob")
    ax.set_title("$dF$ HPF 1.5Hz Magnitude of $F_B$ vector\nNormalized on rotor weight (1079kN) [-]",pad=75)
    ax.grid()
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dF_mag_HPF_threshold)
    ax2.plot(P, X,"-k")
    ax2.axvline(x=0.0,color="k")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()    
    ax2.grid()  
    ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.invert_xaxis()

    ax3.plot(Time_mag_HPF_threshold,dt_mag_HPF_threshold,"ob")
    ax3.set_title("$dt$ HPF 1.5Hz Magnitude of $F_B$ vector [s]",pad=50)
    ax.grid()
    ax3.tick_params(axis='x')
    ax3.tick_params(axis='y')

    P,X=probability_dist(dt_mag_HPF_threshold)
    ax4.plot(P, X,"-k")
    ax4.axvline(x=0.0,color="k")
    ax4.xaxis.tick_top()
    ax4.yaxis.tick_right()     
    ax4.grid()  
    ax4.xaxis.set_label_position('top') 
    ax4.yaxis.set_label_position('right') 
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')
    ax4.invert_xaxis()


    ax5.plot(Time_mag_HPF_threshold,dTheta_mag_HPF_threshold,"ob")
    ax5.set_title("$d\\theta$ HPF 1.5Hz Magnitude of $F_B$ vector [deg]",pad=50)
    ax5.grid()
    ax5.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dTheta_mag_HPF_threshold)
    ax6.plot(P, X,"-k")
    ax6.axvline(x=0.0,color="k")
    ax6.xaxis.tick_top()
    ax6.yaxis.tick_right()   
    ax6.grid()  
    ax6.xaxis.set_label_position('top') 
    ax6.yaxis.set_label_position('right') 
    ax6.tick_params(axis='x')
    ax6.tick_params(axis='y')
    ax6.invert_xaxis()
    fig.supxlabel("Time [s]")
    fig.suptitle("Threshold on 1.5x standard deviation $dF$")
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_dtheta_threshold_time_HPF.png")
    plt.close()

    print(moments(dF_mag_HPF_threshold),moments(dt_mag_HPF_threshold),moments(dTheta_mag_HPF_threshold))


if dtheta_dt_analysis == True:

    out_dir = in_dir+"peak_peak_directional_analysis/"

    #Direction
    zero_crossings_index_BPF_Theta = np.where(np.diff(np.sign(dBPF_Theta_FB)))[0]
    Time_zero_crossings_BPF_Theta = Time_OF[zero_crossings_index_BPF_Theta]
    FBR_zero_crossings_BPF_Theta = BPF_Theta_FB[zero_crossings_index_BPF_Theta]
    dFBR_zero_crossings_BPF_Theta = dBPF_Theta_FB[zero_crossings_index_BPF_Theta]

    zero_crossings_index_HPF_Theta = np.where(np.diff(np.sign(dHPF_Theta_FB)))[0]
    Time_zero_crossings_HPF_Theta = Time_OF[zero_crossings_index_HPF_Theta]
    FBR_zero_crossings_HPF_Theta = HPF_Theta_FB[zero_crossings_index_HPF_Theta]
    dFBR_zero_crossings_HPF_Theta = dHPF_Theta_FB[zero_crossings_index_HPF_Theta]

    # fig = plt.figure(figsize=(14,8))
    # plt.plot(Time_OF,BPF_FBR,"-b",label="Fluctuations BPF Magnitude $F_B$ vector [kN]")
    # plt.plot(Time_OF[:-1],dBPF_FBR,"-k",label="Derivative Fluctuations BPF Magnitude $F_B$ vector [kN/s]")
    # plt.plot(Time_zero_crossings_BPF_FBR,FBR_zero_crossings_BPF_FBR,"or",label="peaks in $F_B$")
    # plt.plot(Time_zero_crossings_BPF_FBR,dFBR_zero_crossings_BPF_FBR,"og",label="zero crossings")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Fluctuations BPF Magnitude $F_B$ vector [kN]")
    # plt.legend()
    # plt.grid()
    # plt.xlim([200,220])
    # plt.ylim([-3000,3000])
    # plt.title("20s period")
    # plt.tight_layout()
    # plt.savefig(out_dir+"BPF_FBR_short.png")
    # plt.close()


    #BPF calc
    dTheta_dir_BPF = []
    dt_dir_BPF = []
    dF_dir_BPF = []
    Time_dir_BPF = []
    for i in np.arange(1,len(zero_crossings_index_BPF_Theta)-1):

        it_0 = zero_crossings_index_BPF_Theta[i-1]
        it_1 = zero_crossings_index_BPF_Theta[i]
        it_2 = zero_crossings_index_BPF_Theta[i+1]

        Time_dir_BPF.append(Time_OF[it_1])

        dF_left = abs(BPF_Theta_FB[it_1] - BPF_Theta_FB[it_0])
        dF_right = abs(BPF_Theta_FB[it_1] - BPF_Theta_FB[it_2])

        dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

        if dF_left > dF_right:
            dTheta_dir_BPF.append(dF_left); dt_dir_BPF.append(dt_left); dF_dir_BPF.append(abs(BPF_FBR[it_1]-BPF_FBR[it_0]))
        else:
            dTheta_dir_BPF.append(dF_right); dt_dir_BPF.append(dt_right); dF_dir_BPF.append(abs(BPF_FBR[it_2]-BPF_FBR[it_1]))


    print(len(Time_dir_BPF))
    plt.rcParams.update({'font.size': 18})

    fig = plt.figure(figsize=(14,8))
    plt.plot(dt_dir_BPF,dTheta_dir_BPF,"ob")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$d$\\theta$ [kN]$")
    plt.title("BPF $F_B$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dtheta_dt_BPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(dTheta_dir_BPF,dF_dir_BPF,"ob")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("BPF $F_B$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dtheta_BPF.png")
    plt.close()


    fig=plt.figure(figsize=(26,15))
    ax=fig.add_subplot(311, label="1")
    ax2=fig.add_subplot(311, label="2", frame_on=False)
    ax3=fig.add_subplot(312)
    ax4=fig.add_subplot(312,frame_on=False)
    ax5=fig.add_subplot(313)
    ax6=fig.add_subplot(313,frame_on=False)

    ax.plot(Time_dir_BPF, dTheta_dir_BPF, "ob")
    ax.set_title("$d\\theta$ BPF 0.3-0.9Hz Direction of $F_B$ vector [deg]",pad=50)
    ax.grid()
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dTheta_dir_BPF)
    ax2.plot(P, X,"-k")
    ax2.axvline(x=0.0,color="k")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.grid()  
    ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.invert_xaxis()

    ax3.plot(Time_dir_BPF,dt_dir_BPF,"ob")
    ax3.set_title("$dt$ BPF 0.3-0.9Hz Direction of $F_B$ vector [s]",pad=50)
    ax.grid()
    ax3.tick_params(axis='x')
    ax3.tick_params(axis='y')

    P,X=probability_dist(dt_dir_BPF)
    ax4.axvline(x=0.0,color="k")
    ax4.plot(P, X,"-k")
    ax4.xaxis.tick_top()
    ax4.yaxis.tick_right()
    ax4.grid()  
    ax4.xaxis.set_label_position('top') 
    ax4.yaxis.set_label_position('right') 
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')
    ax4.invert_xaxis()


    ax5.plot(Time_dir_BPF,dF_dir_BPF,"ob")
    ax5.set_title("dF BPF 0.3-0.9Hz Direction of $F_B$ vector [kN]",pad=50)
    ax5.grid()
    ax5.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dF_dir_BPF)
    ax6.plot(P, X,"-k")
    ax6.axvline(x=0.0,color="k")
    ax6.xaxis.tick_top()
    ax6.yaxis.tick_right()  
    ax6.grid()  
    ax6.xaxis.set_label_position('top') 
    ax6.yaxis.set_label_position('right') 
    ax6.tick_params(axis='x')
    ax6.tick_params(axis='y')
    ax6.invert_xaxis()
    fig.supxlabel("Time [s]")
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_dtheta_time_BPF.png")
    plt.close()

    print(moments(dTheta_dir_BPF),moments(dt_dir_BPF),moments(dF_dir_BPF))

    #BPF threshold calc
    perc_BPF = []
    for phi in np.linspace(np.mean(dTheta_dir_BPF),np.mean(dTheta_dir_BPF)+2*np.std(dTheta_dir_BPF),10):
        Time_dir_BPF_threshold = []
        for i in np.arange(1,len(zero_crossings_index_BPF_Theta)-1):

            it_0 = zero_crossings_index_BPF_Theta[i-1]
            it_1 = zero_crossings_index_BPF_Theta[i]
            it_2 = zero_crossings_index_BPF_Theta[i+1]

            dF_left = abs(BPF_Theta_FB[it_1] - BPF_Theta_FB[it_0])
            dF_right = abs(BPF_Theta_FB[it_1] - BPF_Theta_FB[it_2])

            dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

            if dF_left > phi or dF_right > phi:
                Time_dir_BPF_threshold.append(Time_OF[it_1])


        perc_BPF.append((len(Time_dir_BPF_threshold)/len(Time_dir_BPF))*100)

    dTheta_dir_BPF_threshold = []
    dt_dir_BPF_threshold = []
    dF_dir_BPF_threshold = []
    Time_dir_BPF_threshold = []
    for i in np.arange(1,len(zero_crossings_index_BPF_Theta)-1):

        it_0 = zero_crossings_index_BPF_Theta[i-1]
        it_1 = zero_crossings_index_BPF_Theta[i]
        it_2 = zero_crossings_index_BPF_Theta[i+1]

        dF_left = abs(BPF_Theta_FB[it_1] - BPF_Theta_FB[it_0])
        dF_right = abs(BPF_Theta_FB[it_1] - BPF_Theta_FB[it_2])

        dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

        if dF_left > np.mean(dTheta_dir_BPF)+1.5*np.std(dTheta_dir_BPF) or dF_right > np.mean(dTheta_dir_BPF)+1.5*np.std(dTheta_dir_BPF):

            Time_dir_BPF_threshold.append(Time_OF[it_1])

            if dF_left > dF_right:
                dTheta_dir_BPF_threshold.append(dF_left); dt_dir_BPF_threshold.append(dt_left); dF_dir_BPF_threshold.append(abs(BPF_FBR[it_1]-BPF_FBR[it_0]))
            else:
                dTheta_dir_BPF_threshold.append(dF_right); dt_dir_BPF_threshold.append(dt_right); dF_dir_BPF_threshold.append(abs(BPF_FBR[it_2]-BPF_FBR[it_1]))

    print((len(Time_dir_BPF_threshold)/len(Time_dir_BPF))*100)

    fig = plt.figure(figsize=(14,8))
    plt.plot(dt_dir_BPF_threshold,dTheta_dir_BPF_threshold,"ob")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$d\\theta$ [kN]")
    plt.title("BPF $F_B$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dTheta_dt_BPF_threshold.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(dTheta_dir_BPF_threshold,dF_dir_BPF_threshold,"ob")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("BPF $F_B$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dtheta_BPF_threshold.png")
    plt.close()


    fig=plt.figure(figsize=(26,15))
    ax=fig.add_subplot(311, label="1")
    ax2=fig.add_subplot(311, label="2", frame_on=False)
    ax3=fig.add_subplot(312)
    ax4=fig.add_subplot(312,frame_on=False)
    ax5=fig.add_subplot(313)
    ax6=fig.add_subplot(313,frame_on=False)

    ax.plot(Time_dir_BPF_threshold, dTheta_dir_BPF_threshold, "ob")
    ax.set_title("$d\\theta$ BPF 0.3-0.9Hz Direction of $F_B$ vector [deg]",pad=50)
    ax.grid()
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dTheta_dir_BPF_threshold)
    ax2.plot(P, X,"-k")
    ax2.axvline(x=0.0,color="k")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()   
    ax2.grid()  
    ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.invert_xaxis()

    ax3.plot(Time_dir_BPF_threshold,dt_dir_BPF_threshold,"ob")
    ax3.set_title("$dt$ BPF 0.3-0.9Hz Direction of $F_B$ vector [s]",pad=50)
    ax.grid()
    ax3.tick_params(axis='x')
    ax3.tick_params(axis='y')

    P,X=probability_dist(dt_dir_BPF_threshold)
    ax4.plot(P, X,"-k")
    ax4.axvline(x=0.0,color="k")
    ax4.xaxis.tick_top()
    ax4.yaxis.tick_right()   
    ax4.grid()  
    ax4.xaxis.set_label_position('top') 
    ax4.yaxis.set_label_position('right') 
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')
    ax4.invert_xaxis()


    ax5.plot(Time_dir_BPF_threshold,dF_dir_BPF_threshold,"ob")
    ax5.set_title("$dF$ BPF 0.3-0.9Hz Direction of $F_B$ vector [kN]",pad=50)
    ax5.grid()
    ax5.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dF_dir_BPF_threshold)
    ax6.plot(P, X,"-k")
    ax6.axvline(x=0.0,color="k")
    ax6.xaxis.tick_top()
    ax6.yaxis.tick_right()    
    ax6.grid()  
    ax6.xaxis.set_label_position('top') 
    ax6.yaxis.set_label_position('right') 
    ax6.tick_params(axis='x')
    ax6.tick_params(axis='y')
    ax6.invert_xaxis()
    fig.supxlabel("Time [s]")
    fig.suptitle("Threshold on 1.5x standard deviation $d\\theta$")
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_dtheta_BPF_threshold_time.png")
    plt.close()

    print(moments(dTheta_dir_BPF_threshold),moments(dt_dir_BPF_threshold),moments(dF_dir_BPF_threshold))




    #HPF calc
    dTheta_dir_HPF = []
    dt_dir_HPF = []
    dF_dir_HPF = []
    Time_dir_HPF = []
    for i in np.arange(1,len(zero_crossings_index_HPF_Theta)-1):

        it_0 = zero_crossings_index_HPF_Theta[i-1]
        it_1 = zero_crossings_index_HPF_Theta[i]
        it_2 = zero_crossings_index_HPF_Theta[i+1]

        Time_dir_HPF.append(Time_OF[it_1])

        dF_left = abs(HPF_Theta_FB[it_1] - HPF_Theta_FB[it_0])
        dF_right = abs(HPF_Theta_FB[it_1] - HPF_Theta_FB[it_2])

        dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

        if dF_left > dF_right:
            dTheta_dir_HPF.append(dF_left); dt_dir_HPF.append(dt_left); dF_dir_HPF.append(abs(HPF_FBR[it_1]-HPF_FBR[it_0]))
        else:
            dTheta_dir_HPF.append(dF_right); dt_dir_HPF.append(dt_right); dF_dir_HPF.append(abs(HPF_FBR[it_2]-HPF_FBR[it_1]))


    print(len(Time_dir_HPF))

    fig = plt.figure(figsize=(14,8))
    plt.plot(dt_dir_HPF,dTheta_dir_HPF,"ob")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$d\\theta$ [deg]")
    plt.title("HPF $F_B$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dTheta_dt_HPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(dTheta_dir_HPF,dF_dir_HPF,"ob")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("HPF $F_B$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dtheta_HPF.png")
    plt.close()


    fig=plt.figure(figsize=(26,15))
    ax=fig.add_subplot(311, label="1")
    ax2=fig.add_subplot(311, label="2", frame_on=False)
    ax3=fig.add_subplot(312)
    ax4=fig.add_subplot(312,frame_on=False)
    ax5=fig.add_subplot(313)
    ax6=fig.add_subplot(313,frame_on=False)

    ax.plot(Time_dir_HPF, dTheta_dir_HPF, "ob")
    ax.set_title("$d\\theta$ HPF 1.5Hz Direction of $F_B$ vector [deg]",pad=50)
    ax.grid()
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dTheta_dir_HPF)
    ax2.plot(P, X,"-k")
    ax2.axvline(x=0.0,color="k")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()    
    ax2.grid()  
    ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.invert_xaxis()

    ax3.plot(Time_dir_HPF,dt_dir_HPF,"ob")
    ax3.set_title("$dt$ HPF 1.5Hz Direction of $F_B$ vector [s]",pad=50)
    ax.grid()
    ax3.tick_params(axis='x')
    ax3.tick_params(axis='y')

    P,X=probability_dist(dt_dir_HPF)
    ax4.plot(P, X,"-k")
    ax4.axvline(x=0.0,color="k")
    ax4.xaxis.tick_top()
    ax4.yaxis.tick_right()    
    ax4.grid()  
    ax4.xaxis.set_label_position('top') 
    ax4.yaxis.set_label_position('right') 
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')
    ax4.invert_xaxis()


    ax5.plot(Time_dir_HPF,dF_dir_HPF,"ob")
    ax5.set_title("$dF$ HPF 1.5Hz Direction of $F_B$ vector [kN]",pad=50)
    ax5.grid()
    ax5.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dF_dir_HPF)
    ax6.plot(P, X,"-k")
    ax6.axvline(x=0.0,color="k")
    ax6.xaxis.tick_top()
    ax6.yaxis.tick_right()   
    ax6.grid()  
    ax6.xaxis.set_label_position('top') 
    ax6.yaxis.set_label_position('right') 
    ax6.tick_params(axis='x')
    ax6.tick_params(axis='y')
    ax6.invert_xaxis()
    fig.supxlabel("Time [s]")
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_dtheta_time_HPF.png")
    plt.close()

    print(moments(dTheta_dir_HPF),moments(dt_dir_HPF),moments(dF_dir_HPF))

    #HPF threshold calc
    perc_HPF = []
    for phi in np.linspace(np.mean(dTheta_dir_HPF),np.mean(dTheta_dir_HPF)+2*np.std(dTheta_dir_HPF),10):
        Time_dir_HPF_threshold = []
        for i in np.arange(1,len(zero_crossings_index_HPF_Theta)-1):

            it_0 = zero_crossings_index_HPF_Theta[i-1]
            it_1 = zero_crossings_index_HPF_Theta[i]
            it_2 = zero_crossings_index_HPF_Theta[i+1]

            dF_left = abs(HPF_Theta_FB[it_1] - HPF_Theta_FB[it_0])
            dF_right = abs(HPF_Theta_FB[it_1] - HPF_Theta_FB[it_2])

            dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

            if dF_left > phi or dF_right > phi:
                Time_dir_HPF_threshold.append(Time_OF[it_1])


        perc_HPF.append((len(Time_dir_HPF_threshold)/len(Time_dir_HPF))*100)

    X = np.linspace(np.mean(dTheta_dir_BPF),np.mean(dTheta_dir_BPF)+2*np.std(dTheta_dir_BPF),10)
    fig = plt.figure(figsize=(14,8))
    plt.plot(X,perc_BPF,"-ob",label="BPF")

    X = np.linspace(np.mean(dTheta_dir_HPF),np.mean(dTheta_dir_HPF)+2*np.std(dTheta_dir_HPF),10)
    plt.plot(X,perc_HPF,"-+r",label="HPF")
    plt.xlabel("Threshold value $F_B$ [deg]")
    plt.ylabel("Percentage of peaks in signal [%]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"perc_FBR_BPF_HPF.png")
    plt.close()

    dTheta_dir_HPF_threshold = []
    dt_dir_HPF_threshold = []
    dF_dir_HPF_threshold = []
    Time_dir_HPF_threshold = []
    for i in np.arange(1,len(zero_crossings_index_HPF_Theta)-1):

        it_0 = zero_crossings_index_HPF_Theta[i-1]
        it_1 = zero_crossings_index_HPF_Theta[i]
        it_2 = zero_crossings_index_HPF_Theta[i+1]

        dF_left = abs(HPF_Theta_FB[it_1] - HPF_Theta_FB[it_0])
        dF_right = abs(HPF_Theta_FB[it_1] - HPF_Theta_FB[it_2])

        dt_left = Time_OF[it_1] - Time_OF[it_0]; dt_right = Time_OF[it_2] - Time_OF[it_1]

        if dF_left > np.mean(dTheta_dir_HPF)+1.5*np.std(dTheta_dir_HPF) or dF_right > np.mean(dTheta_dir_HPF)+1.5*np.std(dTheta_dir_HPF):

            Time_dir_HPF_threshold.append(Time_OF[it_1])

            if dF_left > dF_right:
                dTheta_dir_HPF_threshold.append(dF_left); dt_dir_HPF_threshold.append(dt_left); dF_dir_HPF_threshold.append(abs(HPF_FBR[it_1]-HPF_FBR[it_0]))
            else:
                dTheta_dir_HPF_threshold.append(dF_right); dt_dir_HPF_threshold.append(dt_right); dF_dir_HPF_threshold.append(abs(HPF_FBR[it_2]-HPF_FBR[it_1]))

    print((len(Time_dir_HPF_threshold)/len(Time_dir_HPF))*100)

    fig = plt.figure(figsize=(14,8))
    plt.plot(dt_dir_HPF_threshold,dTheta_dir_HPF_threshold,"ob")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$d\\theta$ [deg]")
    plt.title("HPF $F_B$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dTheta_dt_HPF_threshold.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(dTheta_dir_HPF_threshold,dF_dir_HPF_threshold,"ob")
    plt.xlabel("$d\\theta$ [deg]")
    plt.ylabel("$dF [kN]$")
    plt.title("HPF $F_B$ Threshold on 1.5x standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dtheta_HPF_threshold.png")
    plt.close()


    fig=plt.figure(figsize=(26,15))
    ax=fig.add_subplot(311, label="1")
    ax2=fig.add_subplot(311, label="2", frame_on=False)
    ax3=fig.add_subplot(312)
    ax4=fig.add_subplot(312,frame_on=False)
    ax5=fig.add_subplot(313)
    ax6=fig.add_subplot(313,frame_on=False)

    ax.plot(Time_dir_HPF_threshold, dTheta_dir_HPF_threshold, "ob")
    ax.set_title("$d\\theta$ HPF 1.5Hz Direction of $F_B$ vector [deg]",pad=50)
    ax.grid()
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dTheta_dir_HPF_threshold)
    ax2.plot(P, X,"-k")
    ax2.axvline(x=0.0,color="k")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()  
    ax2.grid()  
    ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.invert_xaxis()

    ax3.plot(Time_dir_HPF_threshold,dt_dir_HPF_threshold,"ob")
    ax3.set_title("$dt$ HPF 1.5Hz Direction of $F_B$ vector [s]",pad=50)
    ax.grid()
    ax3.tick_params(axis='x')
    ax3.tick_params(axis='y')

    P,X=probability_dist(dt_dir_HPF_threshold)
    ax4.plot(P, X,"-k")
    ax4.axvline(x=0.0,color="k")
    ax4.xaxis.tick_top()
    ax4.yaxis.tick_right()   
    ax4.grid()  
    ax4.xaxis.set_label_position('top') 
    ax4.yaxis.set_label_position('right') 
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')
    ax4.invert_xaxis()


    ax5.plot(Time_dir_HPF_threshold,dF_dir_HPF_threshold,"ob")
    ax5.set_title("$dF$ HPF 1.5Hz Direction of $F_B$ vector [kN]",pad=50)
    ax5.grid()
    ax5.tick_params(axis='x')
    ax.tick_params(axis='y')

    P,X=probability_dist(dF_dir_HPF_threshold)
    ax6.plot(P, X,"-k")
    ax6.axvline(x=0.0,color="k")
    ax6.xaxis.tick_top()
    ax6.yaxis.tick_right()     
    ax6.grid()  
    ax6.xaxis.set_label_position('top') 
    ax6.yaxis.set_label_position('right') 
    ax6.tick_params(axis='x')
    ax6.tick_params(axis='y')
    ax6.invert_xaxis()
    fig.supxlabel("Time [s]")
    fig.suptitle("Threshold on 1.5x standard deviation $d\\theta$")
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_dtheta_threshold_time_HPF.png")
    plt.close()

    print(moments(dTheta_dir_HPF_threshold),moments(dt_dir_HPF_threshold),moments(dF_dir_HPF_threshold))