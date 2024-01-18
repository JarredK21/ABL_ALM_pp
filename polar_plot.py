from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import pyFAST.input_output as io
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


Start_time = time.time()

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

out_dir = in_dir + "Role_of_weight/polar_plots_weight/"

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])

Time_start = 200
Time_end = 1200

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

time_steps = np.arange(0,len(Time_OF),100)

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
    ax.set_title("{} {}\nTime = {}s".format(Ylabels[j],units[j],Time_OF[it]), va='bottom')
    T = Time_OF[it]
    plt.savefig(out_dir+"polar_plot_{}.png".format(Time_idx))
    plt.close(fig)

    return T


Variables = ["BearingF"]
units = ["[kN]"]
Ylabels = ["Main Bearing Force"]
x_vars = [Theta_FB]
y_vars = [FBR]

ic = np.arange(0,len(time_steps),1)
for j in np.arange(0,len(x_vars)):

    x_var = x_vars[j]; y_var = y_vars[j]

    with Pool() as pool:
        for T in pool.imap(Update,ic):

            print(T,time.time()-Start_time)