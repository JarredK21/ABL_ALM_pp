import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy.stats import pearsonr
import glob 
from scipy import interpolate
from netCDF4 import Dataset
import pandas as pd
from multiprocessing import Pool
import time
import math


start_time = time.time()


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    f_ux = interpolate.interp1d(h,ux_mean_profile)
    mag_horz_vel = []
    mag_fluc_horz_vel = []
    for i in np.arange(0,len(zs)):
        u_i = u[it,i*x:(i+1)*x]; v_i = v[it,i*x:(i+1)*x]
        if zs[i] < h[0]:
            twist_h = f(h[0])
            ux_mean = f_ux(h[0])

        elif zs[i] > h[-1]:
            twist_h = f(h[-1])
            ux_mean = f_ux(h[-1])
        else:
            twist_h = f(zs[i])
            ux_mean = f_ux(zs[i])

        mag_horz_vel_i = u_i*np.cos(twist_h) + v_i*np.sin(twist_h)
        mag_fluc_horz_vel_i = np.subtract(mag_horz_vel_i,ux_mean)
        mag_horz_vel.extend(mag_horz_vel_i)
        mag_fluc_horz_vel.extend(mag_fluc_horz_vel_i)
    mag_horz_vel = np.array(mag_horz_vel)
    mag_fluc_horz_vel = np.array(mag_fluc_horz_vel)
    return mag_horz_vel,mag_fluc_horz_vel


def BPF_calc(it):
    U = u[it]*mask
    min_idx = np.argmin(U); max_idx = np.argmax(U)
    delta_ux = (U[max_idx] - U[min_idx]) * abs(np.degrees(np.arctan2(zs[min_idx],ys[min_idx])) - np.degrees(np.arctan2(zs[max_idx],ys[max_idx])))/180
    
    return delta_ux



#defining twist angles with height from precursor
precursor = Dataset("./abl_statistics76000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38000)
t_end = np.searchsorted(precursor.variables["time"],39201)
u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v



print("line 531", time.time()-start_time)

#directories
in_dir = "./"
out_dir = in_dir

#create netcdf file
ncfile = Dataset(out_dir+"3P_Dataset.nc",mode="w",format='NETCDF4')
ncfile.title = "OpenFAST data sampling output"

#create global dimensions
OF_dim = ncfile.createDimension("OF",None)
sampling_dim = ncfile.createDimension("sampling",None)

#create variables
time_OF = ncfile.createVariable("Time_OF", np.float64, ('OF',),zlib=True)
time_sampling = ncfile.createVariable("Time_sampling", np.float64, ('sampling',),zlib=True)

#sampling data -63.0 plane
print("Rotor avg calcs -63.0 plane",time.time()-start_time)
a = Dataset(in_dir+"sampling_r_-63.0.nc")

#sampling time
Time_sample = np.array(a.variables["time"])
Time_steps = np.arange(0,len(Time_sample))
Time_sample = Time_sample - Time_sample[0]
time_sampling[:] = Time_sample

print("line 542", time.time()-start_time)


p_rotor = a.groups["p_r"]; del a

x = p_rotor.ijk_dims[0] #no. data points
y = p_rotor.ijk_dims[1] #no. data points


normal = 29

#define plotting axes
coordinates = np.array(p_rotor.variables["coordinates"])


xo = coordinates[0:x,0]
yo = coordinates[0:x,1]

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-normal)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
xs = xs + rotor_coordiates[0]
ys = ys + rotor_coordiates[1]
zs = np.linspace(p_rotor.origin[2],p_rotor.origin[2]+p_rotor.axis2[2],y)

print("line 572",time.time()-start_time)

#velocity field
u = np.array(p_rotor.variables["velocityx"])
v = np.array(p_rotor.variables["velocityy"])


u[u<0]=0; v[v<0]=0 #remove negative velocities

#fluctuating streamwise velocity
with Pool() as pool:
    u_hvel = []; u_pri = []
    for u_hvel_it,u_fluc_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_pri.append(u_fluc_hvel_it)
        u_hvel.append(u_hvel_it)
        print(len(u_hvel),time.time()-start_time)
u_pri = np.array(u_pri)
u = np.array(u_hvel); del u_hvel; del v

print("line 592",time.time()-start_time)

YS = ys
ZS = zs


xo = coordinates[:,0]
yo = coordinates[:,1]
zo = coordinates[:,2]

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-29)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
zs = zo - rotor_coordiates[2]

Y = np.linspace(round(np.min(ys),0), round(np.max(ys),0),x )
Z = np.linspace(round(np.min(zs),0), round(np.max(zs),0),y )

del coordinates

dy = (max(Y) - min(Y))/x
dz = (max(Z) - min(Z))/y
dA = dy * dz

del p_rotor

ijk = 0
mask = np.zeros(len(ys))
for j,k in zip(ys,zs):
    r = np.sqrt(j**2 + k**2)
    if r <= 57 and r > 44:
        mask[ijk] = 1
    ijk+=1

group = ncfile.createGroup("3P_analysis")
delta_Ux = group.createVariable("delta_Ux", np.float64, ('sampling'),zlib=True)

delta_Ux_array = []
with Pool() as pool:
    ix = 1
    for delta_Ux_it in pool.imap(BPF_calc, Time_steps):
        delta_Ux_array.append(delta_Ux_it)
        print(ix,time.time()-start_time)
        ix+=1
delta_Ux[:] = np.array(delta_Ux_array); del delta_Ux_array

print(group)
print(ncfile)
ncfile.close()

print("line 959",time.time()-start_time)