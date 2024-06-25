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


def polar_units(it):
    if it < 10:
        Time_idx = "00000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "0000{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "000{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "00{}".format(it)
    elif it >= 10000 and it < 100000:
        Time_idx = "0{}".format(it)
    elif it >= 100000 and it < 1000000:
        Time_idx = "{}".format(it)


    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(projection="polar")
  

    ax.scatter(Theta_FB_var[it],FBR_var[it],c="k", s=20)
    ax.scatter(Aero_Theta_FB_var[it],Aero_FBR_var[it],c="r", s=20)
    ax.scatter(Theta_MR_mod_var[it],MR_mod_var[it],c="m", s=20)
    ax.scatter(Theta_MR_var[it],MR_var[it],c="g",s=20)

    ax.legend(["Main Bearing force vector\n$-1/L_2[(M_{H_z} e_y - M_{H_y} e_z) - LF_H] + L/L_2 W_R$", "Aerodynamic Main bearing force vector\n$-1/L_2[(M_{H_z} e_y - M_{H_y} e_z) - LF_H]$", "Modified OOPBM vector-$(-M_{H_z} e_y,M_{H_y} e_z)$","OOPBM vector-$(M_{H_y} e_y,M_{H_z} e_z)$"],loc="upper right",bbox_to_anchor=(1.12, 1.15))

    ax.arrow(0, 0, Theta_FB_var[it], FBR_var[it], length_includes_head=True, color="k")
    ax.arrow(0, 0, Aero_Theta_FB_var[it], Aero_FBR_var[it], length_includes_head=True, color="r")
    ax.arrow(0, 0, Theta_MR_mod_var[it], MR_mod_var[it], length_includes_head=True, color="m")
    ax.arrow(0, 0, Theta_MR_var[it], MR_var[it], length_includes_head=True, color="g")

    
    ax.set_ylim([0,np.max([np.max(FBR_var),np.max(Aero_FBR_var),np.max(MR_mod_var),np.max(MR_var)])])
    ax.set_title("Vectors [kN] [-]\nTime = {}s".format(round(Time_sampling[it],4)), loc='left',fontsize=20)
    
    T = Time_sampling[it]
    plt.savefig(out_dir+"polar_plot_{}.png".format(Time_idx))
    plt.close(fig)

    return T


def polar_trajectory(it):
    if it < 10:
        Time_idx = "00000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "0000{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "000{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "00{}".format(it)
    elif it >= 10000 and it < 100000:
        Time_idx = "0{}".format(it)
    elif it >= 100000 and it < 1000000:
        Time_idx = "{}".format(it)


    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(projection="polar")
    ax.scatter(Theta_I_var[it],I_var[it]/np.max(I_var),c="b", s=20)
    

    ax.scatter(Theta_FB_var[it],FBR_var[it]/np.max(FBR_var),c="k", s=20)
    ax.scatter(Aero_Theta_FB_var[it],Aero_FBR_var[it]/np.max(Aero_FBR_var),c="r", s=20)
    ax.scatter(Theta_MR_mod_var[it],MR_mod_var[it]/np.max(MR_mod_var),c="m", s=20)
    ax.scatter(Theta_MR_var[it],MR_var[it]/np.max(MR_var),c="g",s=20)

    ax.legend(["Asymmetry vector-$(I_y e_y, I_z e_z)$", "Main Bearing force vector\n$-1/L_2[(M_{H_z} e_y - M_{H_y} e_z) - LF_H] + L/L_2 W_R$", "Aerodynamic Main bearing force vector\n$-1/L_2[(M_{H_z} e_y - M_{H_y} e_z) - LF_H]$", "Modified OOPBM vector-$(-M_{H_z} e_y,M_{H_y} e_z)$","OOPBM vector-$(M_{H_y} e_y,M_{H_z} e_z)$"],loc="upper right",bbox_to_anchor=(1.12, 1.15))

    ax.arrow(0, 0, Theta_I_var[it], I_var[it]/np.max(I_var), length_includes_head=True, color="b")
    ax.arrow(0, 0, Theta_FB_var[it], FBR_var[it]/np.max(FBR_var), length_includes_head=True, color="k")
    ax.arrow(0, 0, Aero_Theta_FB_var[it], Aero_FBR_var[it]/np.max(Aero_FBR_var), length_includes_head=True, color="r")
    ax.arrow(0, 0, Theta_MR_mod_var[it], MR_mod_var[it]/np.max(MR_mod_var), length_includes_head=True, color="m")
    ax.arrow(0, 0, Theta_MR_var[it], MR_var[it]/np.max(MR_var), length_includes_head=True, color="g")

    ax.plot(Theta_I_var[:it+1],I_var[:it+1]/np.max(I_var),"-b")
    ax.plot(Theta_FB_var[:it+1],FBR_var[:it+1]/np.max(FBR_var),"-k")
    ax.plot(Aero_Theta_FB_var[:it+1],Aero_FBR_var[:it+1]/np.max(Aero_FBR_var),"-r")
    ax.plot(Theta_MR_mod_var[:it+1],MR_mod_var[:it+1]/np.max(MR_mod_var),"-m")
    ax.plot(Theta_MR_var[:it+1],MR_var[:it+1]/np.max(MR_var),"-g")

    ax.set_ylim([0,1])
    ax.set_title("Normalized vectors [-]\nTime = {}s".format(round(Time_sampling[it],4)), loc='left',fontsize=20)
    
    T = Time_sampling[it]
    plt.savefig(out_dir+"polar_plot_{}.png".format(Time_idx))
    plt.close(fig)

    return T


def polar_trajectory_2(it):
    if it < 10:
        Time_idx = "00000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "0000{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "000{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "00{}".format(it)
    elif it >= 10000 and it < 100000:
        Time_idx = "0{}".format(it)
    elif it >= 100000 and it < 1000000:
        Time_idx = "{}".format(it)


    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(projection="polar")  

    ax.scatter(Theta_FB_var[it],FBR_var[it]/np.max(FBR_var),c="k", s=20)
    ax.scatter(Aero_Theta_FB_var[it],Aero_FBR_var[it]/np.max(Aero_FBR_var),c="r", s=20)
    ax.scatter(Theta_MR_mod_var[it],MR_mod_var[it]/np.max(MR_mod_var),c="m", s=20)
    ax.scatter(Theta_MR_var[it],MR_var[it]/np.max(MR_var),c="g",s=20)

    ax.legend(["Main Bearing force vector\n$-1/L_2[(M_{H_z} e_y - M_{H_y} e_z) - LF_H] + L/L_2 W_R$", "Aerodynamic Main bearing force vector\n$-1/L_2[(M_{H_z} e_y - M_{H_y} e_z) - LF_H]$", "Modified OOPBM vector-$(-M_{H_z} e_y,M_{H_y} e_z)$","OOPBM vector-$(M_{H_y} e_y,M_{H_z} e_z)$"],loc="upper right",bbox_to_anchor=(1.12, 1.15))

    ax.arrow(0, 0, Theta_FB_var[it], FBR_var[it]/np.max(FBR_var), length_includes_head=True, color="k")
    ax.arrow(0, 0, Aero_Theta_FB_var[it], Aero_FBR_var[it]/np.max(Aero_FBR_var), length_includes_head=True, color="r")
    ax.arrow(0, 0, Theta_MR_mod_var[it], MR_mod_var[it]/np.max(MR_mod_var), length_includes_head=True, color="m")
    ax.arrow(0, 0, Theta_MR_var[it], MR_var[it]/np.max(MR_var), length_includes_head=True, color="g")

    ax.plot(Theta_FB_var[:it+1],FBR_var[:it+1]/np.max(FBR_var),"-k")
    ax.plot(Aero_Theta_FB_var[:it+1],Aero_FBR_var[:it+1]/np.max(Aero_FBR_var),"-r")
    ax.plot(Theta_MR_mod_var[:it+1],MR_mod_var[:it+1]/np.max(MR_mod_var),"-m")
    ax.plot(Theta_MR_var[:it+1],MR_var[:it+1]/np.max(MR_var),"-g")

    ax.set_ylim([0,1])
    ax.set_title("Normalized vectors [-]\nTime = {}s".format(round(Time_sampling[it],4)), loc='left',fontsize=20)
    
    T = Time_sampling[it]
    plt.savefig(out_dir+"polar_plot_{}.png".format(Time_idx))
    plt.close(fig)

    return T



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df_OF = Dataset(in_dir+"Dataset.nc")

#asymmetry variables
Time_sampling = np.array(df_OF.variables["time_sampling"])
dt = Time_sampling[1] - Time_sampling[0]
Time_start = 200; Time_start_idx = np.searchsorted(Time_sampling,Time_start)
Time_end = 1201; Time_end_idx = np.searchsorted(Time_sampling,Time_end)

Time_sampling = Time_sampling[Time_start_idx:Time_end_idx]

group = df_OF.groups["63.0"]
Iy = np.array(group.variables["Iy"][Time_start_idx:Time_end_idx])
Iz = -np.array(group.variables["Iz"][Time_start_idx:Time_end_idx])

Iy_LPF = low_pass_filter(Iy,0.3,dt)

Iz_LPF = low_pass_filter(Iz,0.3,dt)

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
Theta_I = np.degrees(np.arctan2(Iz,Iy))
Theta_I = theta_360(Theta_I)
Theta_I = np.radians(np.array(Theta_I))

I_LPF = np.sqrt(np.add(np.square(Iy_LPF),np.square(Iz_LPF)))
Theta_I_LPF = np.degrees(np.arctan2(Iz_LPF,Iy_LPF))
Theta_I_LPF = theta_360(Theta_I_LPF)
Theta_I_LPF = np.radians(np.array(Theta_I_LPF))


#Turbine variables
Time_OF = np.array(df_OF.variables["time_OF"])

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

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

#Interpolate and filter Aerodynamic variables
f = interpolate.interp1d(Time_OF,RtAeroFys)
RtAeroFys = f(Time_sampling)
RtAeroFys_LPF_1 = low_pass_filter(RtAeroFys,0.3,dt)
RtAeroFys_LPF_2 = low_pass_filter(RtAeroFys,1.0,dt)
RtAeroFys_HPF_1 = np.subtract(RtAeroFys,RtAeroFys_LPF_2)

f = interpolate.interp1d(Time_OF,RtAeroFzs)
RtAeroFzs = f(Time_sampling)
RtAeroFzs_LPF_1 = low_pass_filter(RtAeroFzs,0.3,dt)
RtAeroFzs_LPF_2 = low_pass_filter(RtAeroFzs,1.0,dt)
RtAeroFzs_HPF_1 = np.subtract(RtAeroFzs,RtAeroFzs_LPF_2)

f = interpolate.interp1d(Time_OF,RtAeroMys)
RtAeroMys = f(Time_sampling)
RtAeroMys_LPF_1 = low_pass_filter(RtAeroMys,0.3,dt)
RtAeroMys_LPF_2 = low_pass_filter(RtAeroMys,1.0,dt)
RtAeroMys_HPF_1 = np.subtract(RtAeroMys,RtAeroMys_LPF_2)

f = interpolate.interp1d(Time_OF,RtAeroMzs)
RtAeroMzs = f(Time_sampling)
RtAeroMzs_LPF_1 = low_pass_filter(RtAeroMzs,0.3,dt)
RtAeroMzs_LPF_2 = low_pass_filter(RtAeroMzs,1.0,dt)
RtAeroMzs_HPF_1 = np.subtract(RtAeroMzs,RtAeroMzs_LPF_2)

L1 = 1.912; L2 = 2.09; L = L1 + L2

#Aerodynamic main bearing force vector Total
Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)


Aero_FBy = -(Aero_FBMy + Aero_FBFy); Aero_FBz = -(Aero_FBMz + Aero_FBFz)
Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Aero_Theta_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))
Aero_Theta_FB = theta_360(Aero_Theta_FB)
Aero_Theta_FB = np.radians(np.array(Aero_Theta_FB))

#Aerodynamic main bearing force vector LPF 1
Aero_FBMy = RtAeroMzs_LPF_1/L2; Aero_FBFy = -RtAeroFys_LPF_1*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys_LPF_1/L2; Aero_FBFz = -RtAeroFzs_LPF_1*((L1+L2)/L2)


Aero_FBy = -(Aero_FBMy + Aero_FBFy); Aero_FBz = -(Aero_FBMz + Aero_FBFz)
Aero_FBR_LPF_1 = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Aero_Theta_FB_LPF_1 = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))
Aero_Theta_FB_LPF_1 = theta_360(Aero_Theta_FB_LPF_1)
Aero_Theta_FB_LPF_1 = np.radians(np.array(Aero_Theta_FB_LPF_1))

#Aerodynamic main bearing force vector LPF 2
Aero_FBMy = RtAeroMzs_LPF_2/L2; Aero_FBFy = -RtAeroFys_LPF_2*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys_LPF_2/L2; Aero_FBFz = -RtAeroFzs_LPF_2*((L1+L2)/L2)


Aero_FBy = -(Aero_FBMy + Aero_FBFy); Aero_FBz = -(Aero_FBMz + Aero_FBFz)
Aero_FBR_LPF_2 = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Aero_Theta_FB_LPF_2 = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))
Aero_Theta_FB_LPF_2 = theta_360(Aero_Theta_FB_LPF_2)
Aero_Theta_FB_LPF_2 = np.radians(np.array(Aero_Theta_FB_LPF_2))

#Aerodynamic main bearing force vector HPF 1
Aero_FBMy = RtAeroMzs_HPF_1/L2; Aero_FBFy = -RtAeroFys_HPF_1*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys_HPF_1/L2; Aero_FBFz = -RtAeroFzs_HPF_1*((L1+L2)/L2)


Aero_FBy = -(Aero_FBMy + Aero_FBFy); Aero_FBz = -(Aero_FBMz + Aero_FBFz)
Aero_FBR_HPF_1 = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Aero_Theta_FB_HPF_1 = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))
Aero_Theta_FB_HPF_1 = theta_360(Aero_Theta_FB_HPF_1)
Aero_Theta_FB_HPF_1 = np.radians(np.array(Aero_Theta_FB_HPF_1))


#Interpolate amd filter weighted variables 
LSSTipMys = np.array(df_OF.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(df_OF.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])

LSShftFys = np.array(df_OF.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(df_OF.variables["LSShftFzs"][Time_start_idx:Time_end_idx])

f = interpolate.interp1d(Time_OF,LSSTipMys)
LSSTipMys = f(Time_sampling)
LSSTipMys_LPF_1 = low_pass_filter(LSSTipMys,0.3,dt)
LSSTipMys_LPF_2 = low_pass_filter(LSSTipMys,1.0,dt)
LSSTipMys_HPF_1 = np.subtract(LSSTipMys,LSSTipMys_LPF_2)

f = interpolate.interp1d(Time_OF,LSSTipMzs)
LSSTipMzs = f(Time_sampling)
LSSTipMzs_LPF_1 = low_pass_filter(LSSTipMzs,0.3,dt)
LSSTipMzs_LPF_2 = low_pass_filter(LSSTipMzs,1.0,dt)
LSSTipMzs_HPF_1 = np.subtract(LSSTipMzs,LSSTipMzs_LPF_2)

f = interpolate.interp1d(Time_OF,LSShftFys)
LSShftFys = f(Time_sampling)
LSShftFys_LPF_1 = low_pass_filter(LSShftFys,0.3,dt)
LSShftFys_LPF_2 = low_pass_filter(LSShftFys,1.0,dt)
LSShftFys_HPF_1 = np.subtract(LSShftFys,LSShftFys_LPF_2)

f = interpolate.interp1d(Time_OF,LSShftFzs)
LSShftFzs = f(Time_sampling)
LSShftFzs_LPF_1 = low_pass_filter(LSShftFzs,0.3,dt)
LSShftFzs_LPF_2 = low_pass_filter(LSShftFzs,1.0,dt)
LSShftFzs_HPF_1 = np.subtract(LSShftFzs,LSShftFzs_LPF_2)


L1 = 1.912; L2 = 2.09

#Main Bearing total
FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB = np.degrees(np.arctan2(FBz,FBy))
Theta_FB = theta_360(Theta_FB)
Theta_FB = np.radians(np.array(Theta_FB))

#Main Bearing LPF 1
FBMy = LSSTipMzs_LPF_1/L2; FBFy = -LSShftFys_LPF_1*((L1+L2)/L2)
FBMz = -LSSTipMys_LPF_1/L2; FBFz = -LSShftFzs_LPF_1*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

FBR_LPF_1 = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB_LPF_1 = np.degrees(np.arctan2(FBz,FBy))
Theta_FB_LPF_1 = theta_360(Theta_FB_LPF_1)
Theta_FB_LPF_1 = np.radians(np.array(Theta_FB_LPF_1))

#Main Bearing LPF 2
FBMy = LSSTipMzs_LPF_2/L2; FBFy = -LSShftFys_LPF_2*((L1+L2)/L2)
FBMz = -LSSTipMys_LPF_2/L2; FBFz = -LSShftFzs_LPF_2*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

FBR_LPF_2 = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB_LPF_2 = np.degrees(np.arctan2(FBz,FBy))
Theta_FB_LPF_2 = theta_360(Theta_FB_LPF_2)
Theta_FB_LPF_2 = np.radians(np.array(Theta_FB_LPF_2))

#Main Bearing HPF 1
FBMy = LSSTipMzs_HPF_1/L2; FBFy = -LSShftFys_HPF_1*((L1+L2)/L2)
FBMz = -LSSTipMys_HPF_1/L2; FBFz = -LSShftFzs_HPF_1*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

FBR_HPF_1 = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB_HPF_1 = np.degrees(np.arctan2(FBz,FBy))
Theta_FB_HPF_1 = theta_360(Theta_FB_HPF_1)
Theta_FB_HPF_1 = np.radians(np.array(Theta_FB_HPF_1))


#Modified OOPBM Total
MR_mod = np.sqrt(np.add(np.square(RtAeroMys), np.square(RtAeroMzs)))
Theta_MR_mod = np.degrees(np.arctan2(RtAeroMys,-RtAeroMzs))
Theta_MR_mod = theta_360(Theta_MR_mod)
Theta_MR_mod = np.radians(np.array(Theta_MR_mod))

#Modified OOPBM LPF 1
MR_mod_LPF_1 = np.sqrt(np.add(np.square(RtAeroMys_LPF_1), np.square(RtAeroMzs_LPF_1)))
Theta_MR_mod_LPF_1 = np.degrees(np.arctan2(RtAeroMys_LPF_1,-RtAeroMzs_LPF_1))
Theta_MR_mod_LPF_1 = theta_360(Theta_MR_mod_LPF_1)
Theta_MR_mod_LPF_1 = np.radians(np.array(Theta_MR_mod_LPF_1))

#Modified OOPBM LPF 2
MR_mod_LPF_2 = np.sqrt(np.add(np.square(RtAeroMys_LPF_2), np.square(RtAeroMzs_LPF_2)))
Theta_MR_mod_LPF_2 = np.degrees(np.arctan2(RtAeroMys_LPF_2,-RtAeroMzs_LPF_2))
Theta_MR_mod_LPF_2 = theta_360(Theta_MR_mod_LPF_2)
Theta_MR_mod_LPF_2 = np.radians(np.array(Theta_MR_mod_LPF_2))

#Modified OOPBM HPF 1
MR_mod_HPF_1 = np.sqrt(np.add(np.square(RtAeroMys_HPF_1), np.square(RtAeroMzs_HPF_1)))
Theta_MR_mod_HPF_1 = np.degrees(np.arctan2(RtAeroMys_HPF_1,-RtAeroMzs_HPF_1))
Theta_MR_mod_HPF_1 = theta_360(Theta_MR_mod_HPF_1)
Theta_MR_mod_HPF_1 = np.radians(np.array(Theta_MR_mod_HPF_1))


#OOPBM Total
MR = np.sqrt(np.add(np.square(RtAeroMys), np.square(RtAeroMzs)))
Theta_MR = np.degrees(np.arctan2(RtAeroMzs,RtAeroMys))
Theta_MR = theta_360(Theta_MR)
Theta_MR = np.radians(np.array(Theta_MR))

#OOPBM LPF 1
MR_LPF_1 = np.sqrt(np.add(np.square(RtAeroMys_LPF_1), np.square(RtAeroMzs_LPF_1)))
Theta_MR_LPF_1 = np.degrees(np.arctan2(RtAeroMzs_LPF_1,RtAeroMys_LPF_1))
Theta_MR_LPF_1 = theta_360(Theta_MR_LPF_1)
Theta_MR_LPF_1 = np.radians(np.array(Theta_MR_LPF_1))

#OOPBM LPF 2
MR_LPF_2 = np.sqrt(np.add(np.square(RtAeroMys_LPF_2), np.square(RtAeroMzs_LPF_2)))
Theta_MR_LPF_2 = np.degrees(np.arctan2(RtAeroMzs_LPF_2,RtAeroMys_LPF_2))
Theta_MR_LPF_2 = theta_360(Theta_MR_LPF_2)
Theta_MR_LPF_2 = np.radians(np.array(Theta_MR_LPF_2))

#OOPBM HPF 1
MR_HPF_1 = np.sqrt(np.add(np.square(RtAeroMys_HPF_1), np.square(RtAeroMzs_HPF_1)))
Theta_MR_HPF_1 = np.degrees(np.arctan2(RtAeroMzs_HPF_1,RtAeroMys_HPF_1))
Theta_MR_HPF_1 = theta_360(Theta_MR_HPF_1)
Theta_MR_HPF_1 = np.radians(np.array(Theta_MR_HPF_1))


time_shift = Time_sampling[0]+4.78; time_shift_idx = np.searchsorted(Time_sampling,time_shift)

I_vars = [I,I_LPF]
Theta_I_vars = [Theta_I,Theta_I_LPF]

MR_vars = [MR,MR_LPF_1]
Theta_MR_vars = [Theta_MR,Theta_MR_LPF_1]

MR_mod_vars = [MR_mod,MR_mod_LPF_1]
Theta_MR_mod_vars = [Theta_MR_mod,Theta_MR_mod_LPF_1]

FBR_vars = [FBR,FBR_LPF_1]
Theta_FB_vars = [Theta_FB,Theta_FB_LPF_1]

Aero_FBR_vars = [Aero_FBR,Aero_FBR_LPF_1]
Aero_Theta_FB_vars = [Aero_Theta_FB,Aero_Theta_FB_LPF_1]

folder = ["Total_traj_{}_{}".format(Time_start,Time_end), "LPF_1_traj_{}_{}".format(Time_start,Time_end)]

for i in np.arange(0,len(I_vars)):

    print(folder[i])

    out_dir = in_dir+"polar_plots/{}/".format(folder[i])

    isExist = os.path.exists(out_dir)
    if isExist == False:
        os.makedirs(out_dir)

    Time_sampling = Time_sampling[:-time_shift_idx]

    Time_steps = np.arange(0,len(Time_sampling))

    I_var = I_vars[i]; Theta_I_var = Theta_I_vars[i]
    MR_var = MR_vars[i]; Theta_MR_var = Theta_MR_vars[i]
    MR_mod_var = MR_mod_vars[i]; Theta_MR_mod_var = Theta_MR_mod_vars[i]
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

    MR_mod_var = MR_mod_var[time_shift_idx:]
    Theta_MR_mod_var = Theta_MR_mod_var[time_shift_idx:]
    
    with Pool() as pool:
        for T in pool.imap(polar_trajectory,Time_steps):

            print(T)


MR_vars = [MR_LPF_2,MR_HPF_1]
Theta_MR_vars = [Theta_MR_LPF_2,Theta_MR_HPF_1]

MR_mod_vars = [MR_mod_LPF_2,MR_mod_HPF_1]
Theta_MR_mod_vars = [Theta_MR_mod_LPF_2,Theta_MR_mod_HPF_1]

FBR_vars = [FBR_LPF_2,FBR_HPF_1]
Theta_FB_vars = [Theta_FB_LPF_2,Theta_FB_HPF_1]

Aero_FBR_vars = [Aero_FBR_LPF_2,Aero_FBR_HPF_1]
Aero_Theta_FB_vars = [Aero_Theta_FB_LPF_2,Aero_Theta_FB_HPF_1]

folder = ["LPF_2_{}_{}".format(Time_start,Time_end), "HPF_1_{}_{}".format(Time_start,Time_end)]

for i in np.arange(0,len(FBR_vars)):

    print(folder[i])

    out_dir = in_dir+"polar_plots/{}/".format(folder[i])

    isExist = os.path.exists(out_dir)
    if isExist == False:
        os.makedirs(out_dir)

    MR_var = MR_vars[i]; Theta_MR_var = Theta_MR_vars[i]
    MR_mod_var = MR_mod_vars[i]; Theta_MR_mod_var = Theta_MR_mod_vars[i]
    FBR_var = FBR_vars[i]; Theta_FB_var = Theta_FB_vars[i]
    Aero_FBR_var = Aero_FBR_vars[i]; Aero_Theta_FB_var = Aero_Theta_FB_vars[i]
    
    with Pool() as pool:
        for T in pool.imap(polar_trajectory_2,Time_steps):

            print(T)



MR_vars = [(1/L2)*MR]
Theta_MR_vars = [Theta_MR]

MR_mod_vars = [(1/L2)*MR_mod]
Theta_MR_mod_vars = [Theta_MR_mod]

FBR_vars = [FBR]
Theta_FB_vars = [Theta_FB]

Aero_FBR_vars = [Aero_FBR]
Aero_Theta_FB_vars = [Aero_Theta_FB]


folder = ["Total_units"]

for i in np.arange(0,len(FBR_vars)):

    print(folder[i])

    out_dir = in_dir+"polar_plots/{}/".format(folder[i])

    isExist = os.path.exists(out_dir)
    if isExist == False:
        os.makedirs(out_dir)

    Time_sampling = Time_sampling[:-time_shift_idx]

    Time_steps = np.arange(0,len(Time_sampling))

    MR_var = MR_vars[i]; Theta_MR_var = Theta_MR_vars[i]
    MR_mod_var = MR_mod_vars[i]; Theta_MR_mod_var = Theta_MR_mod_vars[i]
    FBR_var = FBR_vars[i]; Theta_FB_var = Theta_FB_vars[i]
    Aero_FBR_var = Aero_FBR_vars[i]; Aero_Theta_FB_var = Aero_Theta_FB_vars[i]

    Aero_FBR_var = Aero_FBR_var[time_shift_idx:]
    Aero_Theta_FB_var = Aero_Theta_FB_var[time_shift_idx:]

    FBR_var = FBR_var[time_shift_idx:]
    Theta_FB_var = Theta_FB_var[time_shift_idx:]

    MR_var = MR_var[time_shift_idx:]
    Theta_MR_var = Theta_MR_var[time_shift_idx:]

    MR_mod_var = MR_mod_var[time_shift_idx:]
    Theta_MR_mod_var = Theta_MR_mod_var[time_shift_idx:]
    
    with Pool() as pool:
        for T in pool.imap(polar_units,Time_steps):

            print(T)