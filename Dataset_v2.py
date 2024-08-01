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


def Rotor_Avg_calc(it):

    U = u[it]

    Ux_rotor = []
    Iy = 0
    Iz = 0
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if 44.1 <= r <= 50.4:
            Ux_rotor.append(U[ijk])
            Iy+=U[ijk]*k*dA
            Iz+=U[ijk]*j*dA

        ijk+=1
    return np.average(Ux_rotor),Iy,Iz


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



def dUx_calc(it):
    U = u[it]
    u_plane = U.reshape(y,x)

    du_dy = []
    for k in np.arange(0,len(u_plane)):
        du_dy_k = np.subtract(u_plane[k][1:],u_plane[k][:-1])/dy
        du_dy_k = np.insert(du_dy_k,0,du_dy_k[0])
        du_dy.append(du_dy_k)
    du_dy = np.array(du_dy).flatten()

    du_dz = []
    for j in np.arange(0,x):
        du_dz_j = np.subtract(u_plane[1:,j],u_plane[:-1,j])/dz
        du_dz_j = np.insert(du_dz_j,0,du_dz_j[0])
        du_dz.append(du_dz_j)
    du_dz = np.array(du_dz).reshape(y,x).flatten()

    du_dr = np.sqrt(np.add(np.square(du_dy),np.square(du_dz)))

    ijk = 0
    du_dy_avg = []
    du_dz_avg = []
    du_dr_avg = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if 44.1 <= r <= 50.4:
            du_dy_avg.append(du_dy[ijk])
            du_dz_avg.append(du_dz[ijk])
            du_dr_avg.append(du_dr[ijk])
    
        ijk+=1

    return np.average(du_dy_avg),np.average(du_dz_avg),np.average(du_dr_avg)


#directories
in_dir = "./"
out_dir = in_dir


#create netcdf file
ncfile = Dataset(out_dir+"Dataset_2.nc",mode="w",format='NETCDF4')
ncfile.title = "OpenFAST data sampling output"

#create global dimensions
OF_dim = ncfile.createDimension("OF",None)
sampling_dim = ncfile.createDimension("sampling",None)

#create variables
time_OF = ncfile.createVariable("Time_OF", np.float64, ('OF',),zlib=True)
time_sampling = ncfile.createVariable("Time_sampling", np.float64, ('sampling',),zlib=True)

print("Outputting openfast variables",time.time()-start_time)
group = ncfile.createGroup("OpenFAST_Variables")

Azimuth = group.createVariable("Azimuth", np.float64, ('OF',),zlib=True)
RtAeroFxh = group.createVariable("RtAeroFxh", np.float64, ('OF',),zlib=True)
RtAeroFyh = group.createVariable("RtAeroFyh", np.float64, ('OF',),zlib=True)
RtAeroFzh = group.createVariable("RtAeroFzh", np.float64, ('OF',),zlib=True)
RtAeroMxh = group.createVariable("RtAeroMxh", np.float64, ('OF',),zlib=True)
RtAeroMyh = group.createVariable("RtAeroMyh", np.float64, ('OF',),zlib=True)
RtAeroMzh = group.createVariable("RtAeroMzh", np.float64, ('OF',),zlib=True)

LSSGagMys = group.createVariable("LSSGagMys", np.float64, ('OF',),zlib=True)
LSSGagMzs = group.createVariable("LSSGagMzs", np.float64, ('OF',),zlib=True)
LSShftMxa = group.createVariable("LSShftMxa", np.float64, ('OF',),zlib=True)
LSSTipMys = group.createVariable("LSSTipMys", np.float64, ('OF',),zlib=True)
LSSTipMzs = group.createVariable("LSSTipMzs", np.float64, ('OF',),zlib=True)
LSShftFxa = group.createVariable("LSShftFxa", np.float64, ('OF',),zlib=True)
LSShftFys = group.createVariable("LSShftFys", np.float64, ('OF',),zlib=True)
LSShftFzs = group.createVariable("LSShftFzs", np.float64, ('OF',),zlib=True)


#openfast data
df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

time_OF[:] = np.array(df["Time_[s]"])

print("line 466",time.time()-start_time)

Variables = ["Azimuth","RtAeroFxh","RtAeroFyh","RtAeroFzh","RtAeroMxh","RtAeroMyh","RtAeroMzh",
             "LSSGagMys","LSSGagMzs", "LSShftMxa","LSSTipMys","LSSTipMzs","LSShftFxa","LSShftFys","LSShftFzs"]
units = ["[deg]","[N]","[N]","[N]","[N-m]","[N-m]","[N-m]","[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN-m]",
         "[kN]","[kN]","[kN]"]
for i in np.arange(0,len(Variables)):
    Variable = Variables[i]

    txt = "{0}_{1}".format(Variable,units[i])
    signal = np.array(df[txt])
    if Variable == "RtAeroFxh":
        RtAeroFxh[:] = signal; del signal
    elif Variable == "RtAeroFyh":
        RtAeroFyh[:] = signal; del signal
    elif Variable == "RtAeroFzh":
        RtAeroFzh[:] = signal; del signal
    elif Variable == "RtAeroMxh":
        RtAeroMxh[:] = signal; del signal
    elif Variable == "RtAeroMyh":
        RtAeroMyh[:] = signal; del signal
    elif Variable == "RtAeroMzh":
        RtAeroMzh[:] = signal; del signal
    elif Variable == "LSSGagMys":
        LSSGagMys[:] = signal; del signal
    elif Variable == "LSSGagMzs":
        LSSGagMzs[:] = signal; del signal
    elif Variable == "LSShftMxa":
        LSShftMxa[:] = signal; del signal
    elif Variable == "LSSTipMys":
        LSSTipMys[:] = signal; del signal
    elif Variable == "LSSTipMzs":
        LSSTipMzs[:] = signal; del signal
    elif Variable == "LSShftFxa":
        LSShftFxa[:] = signal; del signal
    elif Variable == "LSShftFys":
        LSShftFys[:] = signal; del signal
    elif Variable == "LSShftFzs":
        LSShftFzs[:] = signal; del signal
    elif Variable == "Azimuth":
        Azimuth[:] = signal; del signal

del df

print(ncfile.groups)


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

#sampling data
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

print("line 623",time.time()-start_time)


print("Rotor avg calcs",time.time()-start_time)
group = ncfile.createGroup("Rotor_Avg_Variables")

Ux = group.createVariable("Ux", np.float64, ('sampling'),zlib=True)
Iy = group.createVariable("Iy", np.float64, ('sampling'),zlib=True)
Iz = group.createVariable("Iz", np.float64, ('sampling'),zlib=True)

Ux_array = []; Iy_array = []; Iz_array = []
with Pool() as pool:
    ix = 1
    for Ux_it,Iy_it,Iz_it in pool.imap(Rotor_Avg_calc, Time_steps):
        Ux_array.append(Ux_it)
        Iy_array.append(Iy_it)
        Iz_array.append(Iz_it)
        print(ix,time.time()-start_time)
        ix+=1
Ux[:] = np.array(Ux_array); del Ux_array
Iy[:] = np.array(Iy_array); del Iy_array
Iz[:] = np.array(Iz_array); del Iz_array


#rotor gradients calc
print("Rotor gradients output",time.time()-start_time)
group = ncfile.createGroup("Rotor_Gradients")

dyUx = group.createVariable("dyUx", np.float64, ('sampling'),zlib=True)
dzUx = group.createVariable("dzUx", np.float64, ('sampling'),zlib=True)
drUx = group.createVariable("drUx", np.float64, ('sampling'),zlib=True)


dyUx_array = []
dzUx_array = []
drUx_array = []
print("dUx calcs")
with Pool() as pool:
    ix = 1
    for dyUx_it,dzUx_it,drUx_it in pool.imap(dUx_calc, Time_steps):
        dyUx_array.append(dyUx_it)
        dzUx_array.append(dzUx_it)
        drUx_array.append(drUx_it)
        print(ix,time.time()-start_time)
        ix+=1
dyUx[:] = np.array(dyUx_array); del dyUx_array
dzUx[:] = np.array(dzUx_array); del dzUx_array
drUx[:] = np.array(drUx_array); del drUx_array


print(ncfile.groups)

print(ncfile)
ncfile.close()

print("line 959",time.time()-start_time)