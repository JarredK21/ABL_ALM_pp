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
import os



def low_pass_filter(signal, cutoff,dt):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


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


def polar_trajectory(it):
    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)

    fig = plt.figure(figsize=(8,8))

    plt.axes(projection="polar")
    plt.polar(Theta_I_var[it],I_var[it]/np.max(I_var),"ob",markersize=5)
    plt.arrow(0, 0, Theta_I_var[it], I_var[it]/np.max(I_var), length_includes_head=True, color="b")


    plt.polar(Theta_FB_var[it],FBR_var[it]/np.max(FBR_var),"ok",markersize=5)
    plt.arrow(0, 0, Theta_FB_var[it], FBR_var[it]/np.max(FBR_var), length_includes_head=True, color="k")


    plt.polar(Aero_Theta_FB_var[it],Aero_FBR_var[it]/np.max(Aero_FBR_var),"or",markersize=5)
    plt.arrow(0, 0, Aero_Theta_FB_var[it], Aero_FBR_var[it]/np.max(Aero_FBR_var), length_includes_head=True, color="r")

    plt.polar(Theta_MR_var[it],MR_var[it]/np.max(MR_var),"om",markersize=5)
    plt.arrow(0, 0, Theta_MR_var[it], MR_var[it]/np.max(MR_var), length_includes_head=True, color="m")


    plt.ylim([0,1])
    plt.title("Normalized vectors [-]\nTime = {}s".format(round(Time_OF[it],4)), va='top')
    T = Time_OF[it]
    plt.savefig(out_dir+"polar_plot_{}.png".format(Time_idx))
    plt.close(fig)

    return T



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df_OF = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(df_OF.variables["time_OF"])
Time_sampling = np.array(df_OF.variables["time_sampling"])
dt = Time_OF[1] - Time_OF[0]

Time_start = 200; Time_end = Time_sampling[-1]
Time_start_idx = np.searchsorted(Time_OF,Time_start); Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

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

L1 = 1.912; L2 = 2.09; L = L1 + L2

Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)


Aero_FBy = -(Aero_FBMy + Aero_FBFy); Aero_FBz = -(Aero_FBMz + Aero_FBFz)
Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Aero_Theta_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))
Aero_Theta_FB = theta_360(Aero_Theta_FB)
Aero_Theta_FB = np.radians(np.array(Aero_Theta_FB))

Aero_FBR_LPF_1 = low_pass_filter(Aero_FBR,0.3,dt)
Aero_Theta_FB_LPF_1 = low_pass_filter(Aero_Theta_FB,0.3,dt)

Aero_FBR_LPF_2 = low_pass_filter(Aero_FBR,1.0,dt)
Aero_Theta_FB_LPF_2 = low_pass_filter(Aero_Theta_FB,1.0,dt)

Aero_FBR_HPF_1 = np.subtract(Aero_FBR,Aero_FBR_LPF_2)
Aero_Theta_FB_HPF_1 = np.subtract(Aero_Theta_FB,Aero_Theta_FB_LPF_2)


LSSTipMys = np.array(df_OF.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(df_OF.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])

LSShftFys = np.array(df_OF.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(df_OF.variables["LSShftFzs"][Time_start_idx:Time_end_idx])


L1 = 1.912; L2 = 2.09
FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB = np.degrees(np.arctan2(FBz,FBy))
Theta_FB = theta_360(Theta_FB)
Theta_FB = np.radians(np.array(Theta_FB))

FBR_LPF_1 = low_pass_filter(FBR,0.3,dt)
Theta_FB_LPF_1 = low_pass_filter(Theta_FB,0.3,dt)

FBR_LPF_2 = low_pass_filter(FBR,1.0,dt)
Theta_FB_LPF_2 = low_pass_filter(Theta_FB,1.0,dt)

FBR_HPF_1 = np.subtract(FBR,FBR_LPF_2)
Theta_FB_HPF_1 = np.subtract(Theta_FB,Theta_FB_LPF_2)

MR = np.add(np.square(RtAeroMys), np.square(RtAeroMzs))
Theta_MR = np.degrees(np.arctan2(-RtAeroMys,RtAeroMzs))
Theta_MR = theta_360(Theta_MR)
Theta_MR = np.radians(np.array(Theta_MR))

MR_LPF_1 = low_pass_filter(MR,0.3,dt)
Theta_MR_LPF_1 = low_pass_filter(Theta_MR,0.3,dt)

MR_LPF_2 = low_pass_filter(MR,1.0,dt)
Theta_MR_LPF_2 = low_pass_filter(Theta_MR,1.0,dt)

MR_HPF_1 = np.subtract(MR,MR_LPF_2)
Theta_MR_HPF_1 = np.subtract(Theta_MR,Theta_MR_LPF_2)


group = df_OF.groups["63.0"]
Iy = np.array(group.variables["Iy"])
Iz = -np.array(group.variables["Iz"])

f = interpolate.interp1d(Time_sampling,Iy)
Iy = f(Time_OF)

Iy_LPF_1 = low_pass_filter(Iy,0.3,dt)

Iy_LPF_2 = low_pass_filter(Iy,1.0,dt)

Iy_HPF_1 = np.subtract(Iy,Iy_LPF_2)

f = interpolate.interp1d(Time_sampling,Iz)
Iz = f(Time_OF)

Iz_LPF_1 = low_pass_filter(Iz,0.3,dt)

Iz_LPF_2 = low_pass_filter(Iz,1.0,dt)

Iz_HPF_1 = np.subtract(Iz,Iz_LPF_2)

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
Theta_I = np.degrees(np.arctan2(Iz,Iy))
Theta_I = theta_360(Theta_I)
Theta_I = np.radians(np.array(Theta_I))

I_LPF_1 = np.sqrt(np.add(np.square(Iy_LPF_1),np.square(Iz_LPF_1)))
Theta_I_LPF_1 = np.degrees(np.arctan2(Iz_LPF_1,Iy_LPF_1))
Theta_I_LPF_1 = theta_360(Theta_I_LPF_1)
Theta_I_LPF_1 = np.radians(np.array(Theta_I_LPF_1))

I_LPF_2 = np.sqrt(np.add(np.square(Iy_LPF_2),np.square(Iz_LPF_2)))
Theta_I_LPF_2 = np.degrees(np.arctan2(Iz_LPF_2,Iy_LPF_2))
Theta_I_LPF_2 = theta_360(Theta_I_LPF_2)
Theta_I_LPF_2 = np.radians(np.array(Theta_I_LPF_2))

I_HPF_1 = np.sqrt(np.add(np.square(Iy_HPF_1),np.square(Iz_HPF_1)))
Theta_I_HPF_1 = np.degrees(np.arctan2(Iz_HPF_1,Iy_HPF_1))
Theta_I_HPF_1 = theta_360(Theta_I_HPF_1)
Theta_I_HPF_1 = np.radians(np.array(Theta_I_HPF_1))


time_shift = Time_OF[0]+4.78; time_shift_idx = np.searchsorted(Time_OF,time_shift)
Time_OF = Time_OF[:-time_shift_idx]

Time_steps = np.arange(0,len(Time_OF))

I_vars = [I,I_LPF_1,I_LPF_1,I_HPF_1]
Theta_I_vars = [Theta_I,Theta_I_LPF_1,Theta_I_LPF_2,Theta_I_HPF_1]

MR_vars = [MR,MR_LPF_1,MR_LPF_2,MR_HPF_1]
Theta_MR_vars = [Theta_MR,Theta_MR_LPF_1,Theta_MR_LPF_2,Theta_MR_HPF_1]

FBR_vars = [FBR,FBR_LPF_1,FBR_LPF_2,FBR_HPF_1]
Theta_FB_vars = [Theta_FB,Theta_FB_LPF_1,Theta_FB_LPF_2,Theta_FB_HPF_1]

Aero_FBR_vars = [Aero_FBR,Aero_FBR_LPF_1,Aero_FBR_LPF_2,Aero_FBR_HPF_1]
Aero_Theta_FB_vars = [Aero_Theta_FB,Aero_Theta_FB_LPF_1,Aero_Theta_FB_LPF_2,Aero_Theta_FB_HPF_1]

folder = ["Total", "LPF_1", "LPF_2", "HPF_1"]

for i in np.arange(0,len(I_vars)):

    print(folder)

    out_dir = in_dir+"polar_plots/{}/".format(folder[i])

    isExist = os.path.exists(out_dir)
    if isExist == False:
        os.makedirs(out_dir)

    I_var = I_vars[i]; Theta_I_var = Theta_I_vars[i]
    MR_var = MR_vars[i]; Theta_MR_var = Theta_MR_vars[i]
    FBR_var = FBR_vars[i]; Theta_FB_var = Theta_FB_vars[i]
    Aero_FBR_var = Aero_FBR_vars[i]; Aero_Theta_FB_var = Aero_Theta_FB_vars[i]

    I_var = I_var[:-time_shift_idx]
    Theta_I_var = Theta_I_var[:-time_shift_idx]

    Aero_FBR_var = Aero_FBR_var[time_shift_idx:]
    Aero_Theta_FB_var = Aero_Theta_FB_var[time_shift_idx:]

    FBR_var = FBR_var[time_shift_idx:]
    Theta_FB_var = Theta_FB_var[time_shift_idx:]

    MR_var = MR_var[time_shift_idx:]
    Theta_MR_var = Theta_MR_var[time_shift_idx:]
    
    with Pool() as pool:
        for T in pool.imap(polar_trajectory,Time_steps):

            print(T)
