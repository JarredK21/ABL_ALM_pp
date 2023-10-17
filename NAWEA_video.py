from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob 
import os
from matplotlib import cm
from matplotlib.animation import PillowWriter
import operator
import math
import sys
import time
from multiprocessing import Pool
import cv2
import re
import pyFAST.input_output as io
from scipy.signal import butter,filtfilt
from scipy import interpolate


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def low_pass_filter(signal, cutoff):  

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


def blade_positions(it):

    R = 63
    Az = Azimuth[it]
    Y = [2560]; Y2 = [2560]; Y3 = [2560]
    Z = [90]; Z2 = [90]; Z3 = [90]

    Y.append(Y[0]+R*np.sin(Az))
    Z.append(Z[0]+R*np.cos(Az))

    Az2 = Az+(2*np.pi)/3
    if Az2 > 2*np.pi:
        Az2 -= (2*np.pi)
    
    Az3 = Az-(2*np.pi)/3
    if Az2 < 0:
        Az2 += (2*np.pi)

    Y2.append(Y2[0]+R*np.sin(Az2))
    Z2.append(Z2[0]+R*np.cos(Az2))

    Y3.append(Y3[0]+R*np.sin(Az3))
    Z3.append(Z3[0]+R*np.cos(Az3))

    return Y, Z, Y2, Z2, Y3, Z3


def Update(it):

    T = Time_OF[it]

    fig = plt.figure(figsize=(14,8),constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    f3_ax1 = fig.add_subplot(gs[1, :])
    f3_ax2 = fig.add_subplot(gs[0, :1])
    f3_ax3 = fig.add_subplot(gs[0, 1:])

    #bottom plot
    f3_ax1.plot(Time_OF[:it],LPF_Aero_FBR[:it]/1000,"r")
    f3_ax1_2 = f3_ax1.twinx()
    f3_ax1_2.plot(Time_OF[:it],LPF_IA[:it],"b")
    f3_ax1.set_title("Filtered at 0.1Hz: Correlations = {}".format(np.round(corr,2)),fontsize=16)
    f3_ax1.set_xlabel("Time [s]",fontsize=16)
    f3_ax1.set_ylabel("Bearing Force [kN]",fontsize=16)
    f3_ax1_2.set_ylabel("Asymmetry Parameter [$m^4/s$]",fontsize=16)
    f3_ax1.set_xlim([np.min(Time_OF),np.max(Time_OF)])
    f3_ax1.set_ylim([np.min(LPF_Aero_FBR),np.max(LPF_Aero_FBR)])
    f3_ax1_2.set_ylim([np.min(LPF_IA),np.max(LPF_IA)])


    #rotor disk plot
    U_r = u_r[it] #velocity time step it

    u_plane = U_r.reshape(y_r,x_r)
    X,Y = np.meshgrid(ys_r,zs_r)

    Z = u_plane

    cs = f3_ax2.contourf(X,Y,Z,levels=levels_r, cmap=cm.coolwarm,vmin=cmin_r,vmax=cmax_r)
    f3_ax2.set_xlabel("Y' axis (rotor frame of reference) [m]")
    f3_ax2.set_ylabel("Z' axis (rotor frame of reference) [m]")

    cb = f3_ax2.colorbar(cs)


    YB1,ZB1,YB2,ZB2,YB3,ZB3 = blade_positions(it)

    f3_ax2.plot(YB1,ZB1,color="k",linewidth = 0.5)
    f3_ax2.plot(YB2,ZB2,color="k",linewidth = 0.5)
    f3_ax2.plot(YB3,ZB3,color="k",linewidth = 0.5)  

    Title = "Rotor Plane. \nTotal Horizontal velocity [m/s]: Time = {}[s]".format(round(T,4))

    f3_ax2.set_title(Title)


    U_l = u_l[it]

    u_plane = U_l.reshape(x_l,y_l)
    X,Y = np.meshgrid(xs_l,ys_l)

    Z = u_plane

    cz = f3_ax3.contourf(X,Y,Z,levels=levels_l, cmap=cm.coolwarm,vmin=cmin_l,vmax=cmax_l)
    f3_ax3.set_xlabel("X axis [m]")
    f3_ax3.set_ylabel("Y axis [m]")

    cd = plt.colorbar(cz)

    Title = "Horizotnal Plane hub height. \nTotal Horizontal velocity [m/s]: Time = {}[s]".format(round(T,4))

    f3_ax3.set_title(Title)

    plt.savefig(out_dir+"NAWEA_plot_{}.png".format(round(T,4)))
    plt.cla()
    cb.remove()
    cd.remove()
    plt.close(fig)

    return T



def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(u,v,twist,x,normal,zs,h,height):
    if normal == "z":
        h_idx = np.searchsorted(h,height)
        mag_horz_vel = np.add( np.multiply(u,np.cos(twist[h_idx])) , np.multiply( v,np.sin(twist[h_idx])) )
    else:
        mag_horz_vel = []
        for i in np.arange(0,len(zs)):
            u_i = u[i*x:(i+1)*x]; v_i = v[i*x:(i+1)*x]
            height = zs[i]
            h_idx = np.searchsorted(h,height,side="left")
            if h_idx > 127:
                h_idx = 127
            mag_horz_vel_i = np.add( np.multiply(u_i,np.cos(twist[h_idx])) , np.multiply( v_i,np.sin(twist[h_idx])) )
            mag_horz_vel.extend(mag_horz_vel_i)
        mag_horz_vel = np.array(mag_horz_vel)
    return mag_horz_vel




start_time = time.time()

#in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"
in_dir = "./"
out_dir = in_dir+"NAWEA_plots/"
a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 200
Time_end = Time_sampling[-1]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

dt = Time_OF[1] - Time_OF[0]


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


L1 = 1.912; L2 = 5

Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz

Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))


group = a.groups["63.0"]
IA = np.array(group.variables["IA"])

f = interpolate.interp1d(Time_sampling,IA)
IA = f(Time_OF)


Time_start_idx = np.searchsorted(Time_sampling,Time_start)
Time_end_idx = np.searchsorted(Time_sampling,Time_end)

Time_sampling = Time_sampling[Time_start_idx:Time_end_idx]


Time_shift = 5.5
Time_shift_idx = np.searchsorted(Time_OF,Time_OF[0]+Time_shift)

Aero_FBR = Aero_FBR[Time_shift_idx:]
Azimuth = Azimuth[Time_shift_idx:]
IA = IA[:len(Time_OF)-Time_shift_idx]
Time_OF = Time_OF[Time_shift_idx:]

Time_shift_idx = np.searchsorted(Time_sampling,Time_sampling[0]+Time_shift)

LPF_IA = low_pass_filter(IA,0.1)
LPF_Aero_FBR = low_pass_filter(Aero_FBR,0.1)

corr = correlation_coef(LPF_IA,LPF_Aero_FBR)

Time_steps = np.arange(0,len(Time_OF),100)

print("line 263", time.time()-start_time)


#defining twist angles with height from precursor
precursor = Dataset(in_dir+"abl_statistics60000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],32300)
t_end = np.searchsorted(precursor.variables["time"],33500)
u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
del precursor


#rotor disk data
a = Dataset(in_dir+"sampling_r_-63.0.nc")
p = a.groups["p_r"]

x_r = p.ijk_dims[0] #no. data points
y_r = p.ijk_dims[1] #no. data points

normal = int(np.degrees(np.arccos(p.axis3[0])))

#define plotting axes
coordinates = np.array(p.variables["coordinates"])

xo = coordinates[0:x_r,0]
yo = coordinates[0:x_r,1]

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-normal)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
xs = xs + rotor_coordiates[0]
ys_r = ys + rotor_coordiates[1]
zs_r = np.linspace(p.origin[2],p.origin[2]+p.axis2[2],y_r)

del a

u = np.array(p.variables["velocityx"])
v = np.array(p.variables["velocityy"])
u_r = Horizontal_velocity(u,v,twist,x_r,normal,zs_r,h,height=90); del u; del v; del p

cmin_r = 0
cmax_r = math.ceil(np.max(u_r))

nlevs = (cmax_r-cmin_r)
levels_r = np.linspace(cmin_r,cmax_r,nlevs,dtype=int)
print("line 317",cmin_r,cmax_r)


#longitudinal plane data
a = Dataset(in_dir+"sampling_l_85.nc")
p = a.groups["p_l"]

x_l = p.ijk_dims[0] #no. data points
y_l = p.ijk_dims[1] #no. data points

normal = "z"

#define plotting axes
coordinates = np.array(p.variables["coordinates"])

xo = coordinates[0:x_l,0]
yo = coordinates[0:x_l,1]

xs_l = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x_l)
ys_l = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y_l)
zs_l = 0

del a

u = np.array(p.variables["velocityx"])
v = np.array(p.variables["velocityy"])
u_l = Horizontal_velocity(u,v,twist,x_l,normal,zs_l,h,height=90); del u; del v; del p

cmin_l = 0
cmax_l = math.ceil(np.max(u_l))

nlevs = (cmax_l-cmin_l)
levels_l = np.linspace(cmin_l,cmax_l,nlevs,dtype=int)
print("line 349",cmin_l,cmax_l)


with Pool() as pool:
    for T in pool.imap(Update,Time_steps):

        print(T,time.time()-start_time)



# #whether or not folder exists execute code
# #sort files
# def atof(text):
#     try:
#         retval = float(text)
#     except ValueError:
#         retval = text
#     return retval

# def natural_keys(text):
    
#     return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

# #sort files
# files = glob.glob(out_dir+"*.png")
# files.sort(key=natural_keys)

# #write to video
# img_array = []
# for file in files:
#     img = cv2.imread(file)
#     height, width, layers = img.shape
#     size = (width,height)
#     img_array.append(img)
#     print("line 475)",time.time()-start_time)

# #cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter(out_dir+'NAWEA_video.avi',0, 12, size)
# for im in range(len(img_array)):
#     out.write(img_array[im])
#     print("Line 482)",time.time()-start_time)
# out.release(); del img_array
# print("Line 485)",time.time()-start_time)