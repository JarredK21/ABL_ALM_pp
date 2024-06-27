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
    for i in np.arange(0,len(Time_OF)-1):
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


FBR_analysis = False
time_scale_analysis = False
plot_all_FBR_times = False
peak_to_peak_analysis = False
peak_to_peak_directional_analysis = False
peak_to_peak_weighted_analysis = False
dF_dt_analysis = False
dtheta_dt_analysis = False
dF_F_analysis = True

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df_OF = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(df_OF.variables["time_OF"])
Time_sampling = np.array(df_OF.variables["time_sampling"])
dt = Time_OF[1] - Time_OF[0]

Time_start = 200; Time_end = Time_sampling[-1]
Time_start_idx = np.searchsorted(Time_OF,Time_start); Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

Time_steps = np.arange(0,len(Time_OF))

Azimuth = np.radians(np.array(df_OF.variables["Azimuth"][Time_start_idx:Time_end_idx]))

RtAeroFyh = np.array(df_OF.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
RtAeroFzh = np.array(df_OF.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000


RtAeroMyh = np.array(df_OF.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
RtAeroMzh = np.array(df_OF.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000


LSSTipMys = np.array(df_OF.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(df_OF.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])

LSShftFxa = np.array(df_OF.variables["LSShftFxa"][Time_start_idx:Time_end_idx])
LSShftFys = np.array(df_OF.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(df_OF.variables["LSShftFzs"][Time_start_idx:Time_end_idx])

#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,FBR/1079,"-k",label="Total $F_B$")


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


LPF_1_Theta_FBR = low_pass_filter(Theta_FB,0.3,dt)
LPF_2_Theta_FBR = low_pass_filter(Theta_FB,0.9,dt)
LPF_3_Theta_FBR = low_pass_filter(Theta_FB,1.5,dt)

HPF_Theta_FB = np.subtract(Theta_FB,LPF_3_Theta_FBR)
BPF_Theta_FB = np.subtract(LPF_2_Theta_FBR,LPF_1_Theta_FBR)

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