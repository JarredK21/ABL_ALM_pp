from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import os


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


Start_time = time.time()

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])

Time_start = 200
Time_end = 1201

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

time_steps = np.arange(0,len(Time_OF))

Azimuth = np.array(a.variables["Azimuth"][Time_start_idx:Time_end_idx])
Azimuth = np.radians(Azimuth)

RtAeroFyh = np.array(a.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
RtAeroFzh = np.array(a.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)

RtAeroMyh = np.array(a.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
RtAeroMzh = np.array(a.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])

LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])

L1 = 1.912; L2 = 2.09
FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = FBMy + FBFy; FBz = FBMz + FBFz

FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB = np.degrees(np.arctan2(FBz,FBy))
Theta_FB = theta_360(Theta_FB)
Theta_FB = np.radians(np.array(Theta_FB))

Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz

Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Theta_Aero_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))
Theta_Aero_FB = theta_360(Theta_Aero_FB)
Theta_Aero_FB = np.radians(np.array(Theta_Aero_FB))

print("line 106", time.time()-Start_time)

def Update(ic):

    if ic < 10:
        Time_idx = "000{}".format(ic)
    elif ic >= 10 and ic < 100:
        Time_idx = "00{}".format(ic)
    elif ic >= 100 and ic < 1000:
        Time_idx = "0{}".format(ic)
    elif ic >= 1000 and ic < 10000:
        Time_idx = "{}".format(ic)

    it = time_steps[ic]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='polar')
    c = ax.scatter(x_var[it], y_var[it], c="k", s=20)
    ax.arrow(0, 0, x_var[it], y_var[it], length_includes_head=True)
    ax.set_ylim(0,np.max(y_var))
    ax.set_title("{} {}\nTime = {}s".format(Ylabel,units,Time_OF[it]), va='bottom')
    T = Time_OF[it]
    plt.savefig(out_dir+"polar_plot_{}.png".format(Time_idx))
    plt.close(fig)

    return T


units = "[kN]"
Ylabels = ["Main Bearing Force"]
x_vars = [Theta_FB]
y_vars = [FBR]
out_dirs = [in_dir+"Direction/Total/"]

rotor_weight = -1079.1
percentage = [0.3, 0.5, 0.7, 0.9]

for perc in percentage:
    Fzs = LSShftFzs-(perc*rotor_weight)
    FBFz_perc = -Fzs*((L1+L2)/L2)
    FBz_perc = FBMz + FBFz_perc
    FBR_perc = np.sqrt(np.add(np.square(FBy),np.square(FBz_perc)))
    Theta_FB_perc = np.degrees(np.arctan2(FBz_perc,FBy))
    Theta_FB_perc = theta_360(Theta_FB_perc)
    Theta_FB_perc = np.radians(np.array(Theta_FB_perc))
    y_vars.append(FBR_perc)
    x_vars.append(Theta_FB_perc)
    Ylabels.append("Main Bearing Force with \n{} reduction in weight [kN]".format(perc))
    out_dirs.append(in_dir+"Direction/perc_{}/".format(perc))


x_vars.append(Theta_Aero_FB)
y_vars.append(Aero_FBR/1000)
Ylabels.append("Aerodynamic Main Bearing Forces")
out_dirs.append(in_dir+"Direction/Aero/")


ic = np.arange(0,len(time_steps),1)

for j in np.arange(0,len(x_vars)):

    x_var = x_vars[j]; y_var = y_vars[j]; Ylabel = Ylabels[j]; out_dir = out_dirs[j]

    isExist = os.path.exists(out_dir)
    if isExist == False:
        os.makedirs(out_dir)

    with Pool() as pool:
        for T in pool.imap(Update,ic):

            print(T,time.time()-Start_time)