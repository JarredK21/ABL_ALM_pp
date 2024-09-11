import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy import interpolate
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter,filtfilt


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


def low_pass_filter(signal,dt,cutoff):  
    
    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360


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


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(Time_OF)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt




in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

out_dir = in_dir + "Asymmetry_analysis/"

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["Time_OF"])
Time_sampling = np.array(a.variables["Time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 200
Time_end = Time_sampling[-1]

dt = Time_OF[1] - Time_OF[0]
dt_sampling = Time_sampling[1] - Time_sampling[0]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

OF_Vars = a.groups["OpenFAST_Variables"]

LSSTipMys = np.array(OF_Vars.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(OF_Vars.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])

LSShftFys = np.array(OF_Vars.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(OF_Vars.variables["LSShftFzs"][Time_start_idx:Time_end_idx])

Azimuth = np.radians(np.array(OF_Vars.variables["Azimuth"][Time_start_idx:Time_end_idx]))

RtAeroFyh = np.array(OF_Vars.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
RtAeroFzh = np.array(OF_Vars.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)


RtAeroMyh = np.array(OF_Vars.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
RtAeroMzh = np.array(OF_Vars.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)


MR = np.sqrt(np.add(np.square(LSSTipMys),np.square(LSSTipMzs)))
Theta_MR = np.degrees(np.arctan2(-RtAeroMys,RtAeroMzs))
Theta_MR = theta_360(Theta_MR)
Theta_MR = np.radians(np.array(Theta_MR))

L1 = 1.912; L2 = 2.09

FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)
FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))


Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = -(Aero_FBMy + Aero_FBFy); Aero_FBz = -(Aero_FBMz + Aero_FBFz)
Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))

#Filtering FBR aero
LPF_1_FBR = low_pass_filter(Aero_FBR,dt,0.3)
LPF_2_FBR = low_pass_filter(Aero_FBR,dt,0.9)
LPF_3_FBR = low_pass_filter(Aero_FBR,dt,1.5)
HPF_FBR = np.subtract(Aero_FBR,LPF_3_FBR)
HPF_FBR = np.array(low_pass_filter(HPF_FBR,dt,40))
BPF_FBR = np.subtract(LPF_2_FBR,LPF_1_FBR)

dLPF_FBR = np.array(dt_calc(LPF_1_FBR,dt))
dBPF_FBR = np.array(dt_calc(BPF_FBR,dt))
dHPF_FBR = np.array(dt_calc(HPF_FBR,dt))

zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_FBR)))[0]
zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]
zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]

Env_BPF_FBR = []
Env_Times = []
for i in np.arange(0,len(zero_crossings_index_BPF_FBR),2):
    idx = zero_crossings_index_BPF_FBR[i]
    Env_BPF_FBR.append(BPF_FBR[idx]); Env_Times.append(Time_OF[idx])


#LPF OOPBM calc
dF_mag_LPF = []
FBR_LPF = []
Time_mag_LPF = []
for i in np.arange(0,len(zero_crossings_index_LPF_FBR)-1):

    it_1 = zero_crossings_index_LPF_FBR[i]
    it_2 = zero_crossings_index_LPF_FBR[i+1]

    Time_mag_LPF.append(Time_OF[it_1])

    FBR_LPF.append(LPF_1_FBR[it_1])

    dF_mag_LPF.append(abs(LPF_1_FBR[it_2] - LPF_1_FBR[it_1]))

threshold = 2*np.std(dF_mag_LPF)


# #LPF FBR calc
# Time_LPF = []
# FBR_LPF = []
# for it in np.arange(0,len(Time_OF)):
#     if LPF_1_FBR[it] >= np.mean(LPF_1_FBR)+np.std(LPF_1_FBR):
#         Time_LPF.append(Time_OF[it])
#         FBR_LPF.append(LPF_1_FBR[it])

# #BPF FBR calc
# dF_BPF = []
# Time_BPF = []
# FBR_BPF = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-1):

#     it_1 = zero_crossings_index_BPF_FBR[i]
#     it_2 = zero_crossings_index_BPF_FBR[i+1]

#     Time_BPF.append(Time_OF[it_1])
#     dF_BPF.append(abs(BPF_FBR[it_2] - BPF_FBR[it_1]))
#     FBR_BPF.append(BPF_FBR[it_1])

# #HPF FBR calc
# dF_HPF = []
# Time_HPF = []
# FBR_HPF = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

#     it_1 = zero_crossings_index_HPF_FBR[i]
#     it_2 = zero_crossings_index_HPF_FBR[i+1]

#     Time_HPF.append(Time_OF[it_1])
#     dF_HPF.append(abs(HPF_FBR[it_2] - HPF_FBR[it_1]))
#     FBR_HPF.append(HPF_FBR[it_1])


# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,Aero_FBR,"-k")
# plt.grid()


# fig=plt.figure(figsize=(14,8))
# plt.plot(Time_OF,LPF_1_FBR,"-g")
# plt.plot(Time_OF,BPF_FBR,"-r")
# plt.plot(Time_OF,HPF_FBR,"-b")


# FBR_threshold = []
# T_threshold = []
# for i in np.arange(0,len(dF_BPF)):
#     if abs(dF_BPF[i]) >= np.mean(dF_BPF)+2*np.std(dF_BPF):
#         FBR_threshold.append(FBR_BPF[i]); T_threshold.append(Time_BPF[i])
# plt.scatter(T_threshold,FBR_threshold,color="red")

# FBR_threshold = []
# T_threshold = []
# for i in np.arange(0,len(dF_HPF)):
#     if abs(dF_HPF[i]) >= np.mean(dF_HPF)+2*np.std(dF_HPF):
#         FBR_threshold.append(FBR_HPF[i]); T_threshold.append(Time_HPF[i])
# plt.scatter(T_threshold,FBR_threshold,color="blue")

# print("line 255")

# plt.scatter(Time_LPF,FBR_LPF,color="green")

# plt.grid()
# plt.show()


# Rotor_avg_vars = a.groups["Rotor_Avg_Variables"]
# Rotor_avg_vars_63 = Rotor_avg_vars.groups["63.0"]

Time_start_idx = np.searchsorted(Time_sampling,Time_start)
Time_end_idx = np.searchsorted(Time_sampling,Time_end)

Time_sampling = Time_sampling[Time_start_idx:Time_end_idx]
# Iy = np.array(Rotor_avg_vars_63.variables["Iy"][Time_start_idx:Time_end_idx])
# Iz = -np.array(Rotor_avg_vars_63.variables["Iz"][Time_start_idx:Time_end_idx])

# I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
# LPF_I = low_pass_filter(I,dt_sampling,0.3)


# f = interpolate.interp1d(Time_OF,LPF_Aero_FBR)
# LPF_Aero_FBR_interp = f(np.linspace(Time_OF[0],Time_OF[-1],len(Time_sampling)))


IHL_Variables = a.groups["IHL_Variables"]

Area_high = np.array(IHL_Variables.variables["Area_high"][Time_start_idx:Time_end_idx])
Area_low = np.array(IHL_Variables.variables["Area_low"][Time_start_idx:Time_end_idx])
Area_int = np.array(IHL_Variables.variables["Area_int"][Time_start_idx:Time_end_idx])

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_sampling,Area_high,"-r")
# ax.plot(Time_sampling,Area_low,"-b")
# ax2=ax.twinx()
# ax2.plot(Time_OF,LPF_1_FBR,"-k")

# for i in np.arange(0,len(dF_mag_LPF)):
#     if dF_mag_LPF[i] >= threshold:
#         ax2.plot(Time_mag_LPF[i],FBR_LPF[i],"ok")
# plt.show()

Ux_high = np.array(IHL_Variables.variables["Ux_high"][Time_start_idx:Time_end_idx])
Ux_low = np.array(IHL_Variables.variables["Ux_low"][Time_start_idx:Time_end_idx])
Ux_int = np.array(IHL_Variables.variables["Ux_int"][Time_start_idx:Time_end_idx])


# fig,(ax,ax3) = plt.subplots(2,1,figsize=(14,8))
# ax.plot(Time_mag_BPF[:-Time_shift_idx],dF_mag_BPF[Time_shift_idx:],"-k")
# ax.axhline(y=2*np.std(dF_mag_BPF),linestyle="--",color="k")
# ax.axhline(y=-2*np.std(dF_mag_BPF),linestyle="--",color="k")
# # for i in np.arange(0,len(dF_mag_BPF)):
# #     if abs(dF_mag_BPF[i]) >= np.mean(dF_mag_BPF)+2*np.std(dF_mag_BPF):
# #         ax.plot(Time_mag_BPF[i],dF_mag_BPF[i],"ok")
# ax2=ax.twinx()
# time_shift_idx = np.searchsorted(Time_sampling,Time_sampling[0]+4.6)
# ax2.plot(Time_sampling[:-time_shift_idx], Area_high[:-time_shift_idx],"-r")
# ax2.plot(Time_sampling[:-time_shift_idx],Area_low[:-time_shift_idx],"-b")
# ax.grid()
# ax3.plot(Time_mag_BPF[:-Time_shift_idx],dF_mag_BPF[Time_shift_idx:],"-k")
# ax3.axhline(y=2*np.std(dF_mag_BPF),linestyle="--",color="k")
# ax3.axhline(y=-2*np.std(dF_mag_BPF),linestyle="--",color="k")
# ax4=ax3.twinx()

Iy_high = np.array(IHL_Variables.variables["Iy_high"][Time_start_idx:Time_end_idx])
Iy_low = np.array(IHL_Variables.variables["Iy_low"][Time_start_idx:Time_end_idx])
Iy_int = np.array(IHL_Variables.variables["Iy_int"][Time_start_idx:Time_end_idx])

Iz_high = -np.array(IHL_Variables.variables["Iz_high"][Time_start_idx:Time_end_idx])
Iz_low = -np.array(IHL_Variables.variables["Iz_low"][Time_start_idx:Time_end_idx])
Iz_int = -np.array(IHL_Variables.variables["Iz_int"][Time_start_idx:Time_end_idx])

I_high = np.sqrt(np.add(np.square(Iy_high),np.square(Iz_high)))
I_high_LPF = low_pass_filter(I_high,dt_sampling,0.3)
I_low = np.sqrt(np.add(np.square(Iy_low),np.square(Iz_low)))
I_low_LPF = low_pass_filter(I_low,dt_sampling,0.3)
I_int = np.sqrt(np.add(np.square(Iy_int),np.square(Iz_int)))
I_int_LPF = low_pass_filter(I_int,dt_sampling,0.3)


drUx_high = np.array(IHL_Variables.variables["drUx_high"][Time_start_idx:Time_end_idx])
drUx_high_LPF = low_pass_filter(drUx_high,dt_sampling,0.3)
drUx_low = np.array(IHL_Variables.variables["drUx_low"][Time_start_idx:Time_end_idx])
drUx_low_LPF = low_pass_filter(drUx_low,dt_sampling,0.3)
drUx_int = np.array(IHL_Variables.variables["drUx_int"][Time_start_idx:Time_end_idx])
drUx_int_LPF = low_pass_filter(drUx_int,dt_sampling,0.3)

#BPF FBR calc
dF_mag_BPF = []
Time_mag_BPF = []
FBR_mag_BPF = []
for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-1):

    it_1 = zero_crossings_index_BPF_FBR[i]
    it_2 = zero_crossings_index_BPF_FBR[i+1]

    dF_mag_BPF.append(abs(BPF_FBR[it_2] - BPF_FBR[it_1]))
    Time_mag_BPF.append(Time_OF[it_1])
    FBR_mag_BPF.append(BPF_FBR[it_1])

threshold = 2*np.std(dF_mag_BPF)

fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF,BPF_FBR,"-k")
ax.set_ylabel("BPF FB")
for i in np.arange(0,len(dF_mag_BPF)):
    if abs(dF_mag_BPF[i]) >= threshold:
        ax.plot(Time_mag_BPF[i],FBR_mag_BPF[i],"ob")
ax2=ax.twinx()
ax2.plot(Time_sampling,Area_high,"-r",label="High")
ax2.plot(Time_sampling,Area_low,"-b",label="low")
ax2.set_ylabel("Area")
ax.grid()
plt.tight_layout()


fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF,BPF_FBR,"-k")
ax.set_ylabel("BPF FB")
for i in np.arange(0,len(dF_mag_BPF)):
    if abs(dF_mag_BPF[i]) >= threshold:
        ax.plot(Time_mag_BPF[i],FBR_mag_BPF[i],"ob")
ax2=ax.twinx()
ax2.plot(Time_sampling,Iy_high,"-r",label="High")
ax2.plot(Time_sampling,Iy_low,"-b",label="low")
ax2.set_ylabel("Iy")
ax.grid()
plt.tight_layout()

fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF,BPF_FBR,"-k")
ax.set_ylabel("BPF FB")
for i in np.arange(0,len(dF_mag_BPF)):
    if abs(dF_mag_BPF[i]) >= threshold:
        ax.plot(Time_mag_BPF[i],FBR_mag_BPF[i],"ob")
ax2=ax.twinx()
ax2.plot(Time_sampling,Iz_high,"-r",label="High")
ax2.plot(Time_sampling,Iz_low,"-b",label="low")
ax2.set_ylabel("Iz")
ax.grid()
plt.tight_layout()

fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF,BPF_FBR,"-k")
ax.set_ylabel("BPF FB")
for i in np.arange(0,len(dF_mag_BPF)):
    if abs(dF_mag_BPF[i]) >= threshold:
        ax.plot(Time_mag_BPF[i],FBR_mag_BPF[i],"ob")
ax2=ax.twinx()
ax2.plot(Time_sampling,I_high,"-r",label="High")
ax2.plot(Time_sampling,I_low,"-b",label="low")
ax2.set_ylabel("I")
ax.grid()
plt.tight_layout()


#BPF FBR calc
Area_high_threshold = []; Area_low_threshold = []
Ux_high_threshold = []; Ux_low_threshold = []
Iy_high_threshold = []; Iy_low_threshold = []
Iz_high_threshold = []; Iz_low_threshold = []
I_high_threshold = []; I_low_threshold = []
drUx_high_threshold = []; drUx_low_threshold = []
start_time_idx = np.searchsorted(Time_OF[zero_crossings_index_BPF_FBR],Time_OF[0]+4.6)
for i in np.arange(start_time_idx,len(zero_crossings_index_BPF_FBR)-1):

    it_1 = zero_crossings_index_BPF_FBR[i]
    it_2 = zero_crossings_index_BPF_FBR[i+1]

    if abs(BPF_FBR[it_2] - BPF_FBR[it_1]) >= threshold:
        idx = np.searchsorted(Time_sampling,Time_OF[it_1]-4.6)
        Area_high_threshold.append(Area_high[idx]); Area_low_threshold.append(Area_low[idx])
        Ux_high_threshold.append(Ux_high[idx]); Ux_low_threshold.append(Ux_low[idx])
        Iy_high_threshold.append(Iy_high[idx]); Iy_low_threshold.append(Iy_low[idx])
        Iz_high_threshold.append(Iz_high[idx]); Iz_low_threshold.append(Iz_low[idx])
        I_high_threshold.append(I_high[idx]); I_low_threshold.append(I_low[idx])
        drUx_high_threshold.append(drUx_high[idx]); drUx_low_threshold.append(drUx_low[idx])


fig = plt.figure(figsize=(14,8))
P,X = probability_dist(Area_high)
plt.plot(X,P,label="Area high")
P,X = probability_dist(Area_low)
plt.plot(X,P,label="Area low")
P,X = probability_dist(Area_high_threshold)
plt.plot(X,P,label="Area high $2\sigma$")
P,X = probability_dist(Area_low_threshold)
plt.plot(X,P,label="Area low $2\sigma$")
plt.xlabel("Area in rotor disk [$m^2$]")
plt.ylabel("Probability [-]")
plt.grid()
plt.legend()
plt.tight_layout()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(Ux_high)
plt.plot(X,P,label="Ux high")
P,X = probability_dist(Ux_low)
plt.plot(X,P,label="Ux low")
P,X = probability_dist(Ux_high_threshold)
plt.plot(X,P,label="Ux high $2\sigma$")
P,X = probability_dist(Ux_low_threshold)
plt.plot(X,P,label="Ux low $2\sigma$")
plt.xlabel("Average velocity [m/s]")
plt.ylabel("Probability [-]")
plt.grid()
plt.legend()
plt.tight_layout()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(Iy_high)
plt.plot(X,P,label="Iy high")
P,X = probability_dist(Iy_low)
plt.plot(X,P,label="Iy low")
P,X = probability_dist(Iy_high_threshold)
plt.plot(X,P,label="Iy high $2\sigma$")
P,X = probability_dist(Iy_low_threshold)
plt.plot(X,P,label="Iy low $2\sigma$")
plt.xlabel("Asymmetry vector y component [$m^4/s$]")
plt.ylabel("Probability [-]")
plt.grid()
plt.legend()
plt.tight_layout()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(Iz_high)
plt.plot(X,P,label="Iz high")
P,X = probability_dist(Iz_low)
plt.plot(X,P,label="Iz low")
P,X = probability_dist(Iz_high_threshold)
plt.plot(X,P,label="Iz high $2\sigma$")
P,X = probability_dist(Iz_low_threshold)
plt.plot(X,P,label="Iz low $2\sigma$")
plt.xlabel("Asymmetry vector z component [$m^4/s$]")
plt.ylabel("Probability [-]")
plt.grid()
plt.legend()
plt.tight_layout()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(I_high)
plt.plot(X,P,label="I high")
P,X = probability_dist(I_low)
plt.plot(X,P,label="I low")
P,X = probability_dist(I_high_threshold)
plt.plot(X,P,label="I high $2\sigma$")
P,X = probability_dist(I_low_threshold)
plt.plot(X,P,label="I low $2\sigma$")
plt.xlabel("Asymmetry vector magnitude [$m^4/s$]")
plt.ylabel("Probability [-]")
plt.grid()
plt.legend()
plt.tight_layout()


fig = plt.figure(figsize=(14,8))
P,X = probability_dist(drUx_high)
plt.plot(X,P,label="dUx/dr high")
P,X = probability_dist(drUx_low)
plt.plot(X,P,label="dUx/dr low")
P,X = probability_dist(drUx_high_threshold)
plt.plot(X,P,label="dUx/dr high $2\sigma$")
P,X = probability_dist(drUx_low_threshold)
plt.plot(X,P,label="dUx/dr low $2\sigma$")
plt.xlabel("Average velocity gradient [1/s]")
plt.ylabel("Probability [-]")
plt.grid()
plt.legend()
plt.tight_layout()

plt.show()



#Time_shift_idx = np.searchsorted(Time_mag_BPF,Time_mag_BPF[0]+4.6)


f = interpolate.interp1d(Time_OF,LPF_1_FBR)
LPF_1_FBR_interp = f(Time_sampling)
time_shift = 4.78
time_shift_idx = np.searchsorted(Time_sampling,Time_sampling[0]+time_shift)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling[:-time_shift_idx],LPF_1_FBR_interp[time_shift_idx:],"-r")
ax2=ax.twinx()
ax2.plot(Time_sampling[:-time_shift_idx],drUx_low_LPF[:-time_shift_idx],"-b")
ax.grid()



Rotor_gradients = a.groups["Rotor_Gradients"]
drUx = np.array(Rotor_gradients.variables["drUx"][Time_start_idx:Time_end_idx])


fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling[:-time_shift_idx],LPF_1_FBR_interp[time_shift_idx:],"-r")
ax2=ax.twinx()
ax2.plot(Time_sampling[:-time_shift_idx],drUx[:-time_shift_idx],"-b")
ax.grid()
ax2.set_ylabel("drUx total")
print(correlation_coef(LPF_1_FBR_interp[time_shift_idx:],drUx[:-time_shift_idx]))

f = interpolate.interp1d(Time_OF,BPF_FBR)
BPF_FBR_interp = f(Time_sampling)

fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling[:-time_shift_idx],BPF_FBR_interp[time_shift_idx:],"-r")
ax2=ax.twinx()
ax2.plot(Time_sampling[:-time_shift_idx],drUx[:-time_shift_idx],"-b")
ax.grid()
ax2.set_ylabel("drUx total")
print(correlation_coef(BPF_FBR_interp[time_shift_idx:],drUx[:-time_shift_idx]))

time_shift = 4.78
time_shift_idx = np.searchsorted(Time_OF,Time_OF[0]+time_shift)

fig,ax = plt.subplots(figsize=(14,8))
ax2=ax.twinx()
ax2.plot(Time_OF[:-time_shift_idx],BPF_FBR[time_shift_idx:],"-k")
time_shift_idx = np.searchsorted(Time_sampling,Time_sampling[0]+time_shift)
ax.plot(Time_sampling[:-time_shift_idx],drUx_high_LPF[:-time_shift_idx],"-r",label="High")
ax.plot(Time_sampling[:-time_shift_idx],drUx_low_LPF[:-time_shift_idx],"-b",label="Low")
ax.plot(Time_sampling[:-time_shift_idx],drUx_int_LPF[:-time_shift_idx],"-g",label="Int")
ax.legend()
ax.grid()

out_dir=in_dir+"three_frequency_analysis/OOPBM_analysis/"
plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(14,8))
P,X = probability_dist(drUx_high_LPF)
plt.plot(X,P,"-r",label="High")
P,X = probability_dist(drUx_low_LPF)
plt.plot(X,P,"-b",label="Low")
P,X = probability_dist(drUx_int_LPF)
plt.plot(X,P,"-g",label="Int")
plt.legend()
plt.grid()
plt.xlabel("Average gradient magnitude [1/s]")
plt.ylabel("Probability [-]")
plt.savefig(out_dir+"PDF_drUx_IHL.png")
plt.close()



#local variance calc
local_var_HPF = []
start_time_idx = np.searchsorted(Time_OF,10+Time_OF[0])
for i in np.arange(0,len(Time_OF)-start_time_idx,1):
    local_var_HPF.append(np.std(HPF_FBR[i:i+start_time_idx]))

local_var_HPF_LPF = low_pass_filter(local_var_HPF,dt,0.3)

idx = int(start_time_idx/2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF[idx+1:-(idx+time_shift_idx)],local_var_HPF_LPF[time_shift_idx:],"-r")
ax2=ax.twinx()
ax2.plot(Time_sampling[:-time_shift_idx],drUx_low_LPF[:-time_shift_idx],"-b")
ax.grid()

# f = interpolate.interp1d(Time_sampl,drUx_low_LPF)
# drUx_low_LPF_interp = f(Env_Times)
# drUx_low_interp_shifted = drUx_low_LPF_interp[:-time_shift_idx]

# Env_BPF_FBR_shifted = Env_BPF_FBR[time_shift_idx:]

# print(correlation_coef(Env_BPF_FBR_shifted,drUx_low_interp_shifted))



fig,ax = plt.subplots(figsize=(14,8))
ax2=ax.twinx()
ax2.plot(Time_sampling[:-time_shift_idx],drUx_low_LPF[:-time_shift_idx],"-b")
time_shift_idx = np.searchsorted(Env_Times,Env_Times[0]+time_shift)
ax.plot(Env_Times[:-time_shift_idx],Env_BPF_FBR[time_shift_idx:],"-r")
ax.grid()

f = interpolate.interp1d(Time_sampling,drUx_low_LPF)
drUx_low_LPF_interp = f(Env_Times)
drUx_low_interp_shifted = drUx_low_LPF_interp[:-time_shift_idx]

Env_BPF_FBR_shifted = Env_BPF_FBR[time_shift_idx:]

print(correlation_coef(Env_BPF_FBR_shifted,drUx_low_interp_shifted))


plt.show()

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_sampling,Area_high)
# plt.ylabel("Area high")
# plt.grid()

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_sampling,Area_low)
# plt.ylabel("Area low")
# plt.grid()

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_sampling,Area_int)
# plt.ylabel("Area int")
# plt.grid()

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_sampling,I_high_LPF)
# plt.ylabel("I high LPF")
# plt.grid()

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_sampling,I_low_LPF)
# plt.ylabel("I low LPF")
# plt.grid()

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_sampling,I_int_LPF)
# plt.ylabel("I int LPF")
# plt.grid()

# print(correlation_coef(I_high_LPF,LPF_I))
# print(correlation_coef(I_low_LPF,LPF_I))
# print(correlation_coef(I_int_LPF,LPF_I))

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_sampling,LPF_I)
# plt.ylabel("I LPF")
# plt.grid()