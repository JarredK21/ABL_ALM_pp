import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import statistics
from scipy.signal import butter,filtfilt
from scipy import interpolate



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



def probability_dist(y):

    mu = np.mean(y)
    var = np.var(y)
    sd = np.std(y)
    print(mu)
    print(sd)
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



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Asymmetry_Dataset.nc")

Time = np.array(a.variables["time"])
Time = Time - Time[0]
dt = Time[1] - Time[0]

Time_start = 200; Time_start_idx = np.searchsorted(Time,Time_start)
Time = Time[Time_start_idx:]
Time_steps = np.arange(0,len(Time))

A_high = np.array(a.variables["Area_high"][Time_start_idx:])
A_low = np.array(a.variables["Area_low"][Time_start_idx:])
A_int = np.array(a.variables["Area_int"][Time_start_idx:])


Iy_high = np.array(a.variables["Iy_high"][Time_start_idx:])
Iy_low = np.array(a.variables["Iy_low"][Time_start_idx:])
Iy_int = np.array(a.variables["Iy_int"][Time_start_idx:])

Iz_high = np.array(a.variables["Iz_high"][Time_start_idx:])
Iz_low = np.array(a.variables["Iz_low"][Time_start_idx:])
Iz_int = np.array(a.variables["Iz_int"][Time_start_idx:])

I_high = np.sqrt(np.add(np.square(Iy_high),np.square(Iz_high)))
I_low = np.sqrt(np.add(np.square(Iy_low),np.square(Iz_low)))
I_int = np.sqrt(np.add(np.square(Iy_int),np.square(Iz_int)))

Theta_high = np.degrees(np.arctan2(Iz_high,Iy_high))
Theta_high = np.array(theta_360(Theta_high))

Theta_low = np.degrees(np.arctan2(Iz_low,Iy_low))
Theta_low = np.array(theta_360(Theta_low))

Theta_int = np.degrees(np.arctan2(Iz_int,Iy_int))
Theta_int = np.array(theta_360(Theta_int))

Iy = np.array(a.variables["Iy"][Time_start_idx:])
Iz = -np.array(a.variables["Iz"][Time_start_idx:])

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

Theta = np.degrees(np.arctan2(Iz,Iy))
Theta = np.array(theta_360(Theta))

DTheta_dt = np.subtract(Theta[1:],Theta[:-1])/dt

Delta_I_high_low = np.subtract(I_high,I_low)
Delta_I_high_int = np.subtract(I_high,I_int)
Delta_I_low_int = np.subtract(I_low,I_int)

Delta_Theta_high_low = np.subtract(Theta_high,Theta_low)
Delta_Theta_high_int = np.subtract(Theta_high,Theta_int)
Delta_Theta_low_int = np.subtract(Theta_low,Theta_int)

Delta_Theta_high_low[Delta_Theta_high_low<0]+=360
Delta_Theta_high_int[Delta_Theta_high_int<0]+=360
Delta_Theta_low_int[Delta_Theta_low_int<0]+=360

Delta_Theta_high_int-=90
Delta_Theta_high_low-=90
Delta_Theta_low_int-=90

df_OF = Dataset(in_dir+"Dataset.nc")
Time_OF = np.array(df_OF.variables["time_OF"])
Time_start = 200
Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_OF = Time_OF[Time_start_idx:]
dt_OF = Time_OF[1] - Time_OF[0]


Azimuth = np.radians(np.array(df_OF.variables["Azimuth"][Time_start_idx:]))

RtAeroFyh = np.array(df_OF.variables["RtAeroFyh"][Time_start_idx:])
RtAeroFzh = np.array(df_OF.variables["RtAeroFzh"][Time_start_idx:])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000


RtAeroMyh = np.array(df_OF.variables["RtAeroMyh"][Time_start_idx:])
RtAeroMzh = np.array(df_OF.variables["RtAeroMzh"][Time_start_idx:])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000


LSShftFys = np.array(df_OF.variables["LSShftFys"][Time_start_idx:])
LSShftFzs = np.array(df_OF.variables["LSShftFzs"][Time_start_idx:])
LSSTipMys = np.array(df_OF.variables["LSSTipMys"][Time_start_idx:])
LSSTipMzs = np.array(df_OF.variables["LSSTipMzs"][Time_start_idx:])


L1 = 1.912; L2 = 2.09; L = L1 + L2

Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = -(Aero_FBMy + Aero_FBFy); Aero_FBz = -(Aero_FBMz + Aero_FBFz)
Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Aero_theta = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))-90
Aero_theta = np.array(theta_360(Aero_theta))

DAero_theta_dt = np.subtract(Aero_theta[1:],Aero_theta[:-1])/dt_OF



MR = np.add(np.square(RtAeroMys/L2), np.square(RtAeroMzs/L2))
Theta_MR = np.degrees(np.arctan2(-RtAeroMys,RtAeroMzs))
Theta_MR = np.array(theta_360(Theta_MR))

Dtheta_MR_dt = np.subtract(Theta_MR[1:],Theta_MR[:-1])/dt_OF

FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)
FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB = np.degrees(np.arctan2(FBz,FBy))
Theta_FB = np.array(theta_360(Theta_FB))

Dtheta_FB_dt = np.subtract(Theta_FB[1:],Theta_FB[:-1])/dt_OF

fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF[1:],Dtheta_FB_dt)
plt.ylim([-360,360])
plt.show()
        
fig = plt.figure(figsize=(14,8))
plt.plot(Time,Iy,"-k")
for it in Time_steps:

    if A_low[it] == 0:
        plt.plot(Time[it],Iy[it],"ob")
    else:
        plt.plot(Time[it],Iy[it],"*r")

plt.ylabel("Iy")

fig = plt.figure(figsize=(14,8))
plt.plot(Time,Iz,"-k")
for it in Time_steps:

    if A_low[it] == 0:
        plt.plot(Time[it],Iz[it],"ob")
    else:
        plt.plot(Time[it],Iz[it],"*r")

plt.ylabel("Iz")

plt.show()