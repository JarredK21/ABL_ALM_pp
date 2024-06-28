from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import math
import time
from multiprocessing import Pool
from scipy import interpolate
from matplotlib.patches import Circle


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    f_ux = interpolate.interp1d(h,ux_mean_profile)
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
        mag_fluc_horz_vel.extend(mag_fluc_horz_vel_i)
    mag_fluc_horz_vel = np.array(mag_fluc_horz_vel)
    return mag_fluc_horz_vel

start_time = time.time()

#defining twist angles with height from precursor
precursor = Dataset("./abl_statistics76000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38200)
t_end = np.searchsorted(precursor.variables["time"],39201)
u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v

print("line 60", time.time()-start_time)


#directories
in_dir = "./"

a = Dataset("./sampling_r_-63.0.nc")

#time options
Time = np.array(a.variables["time"])
dt = Time[1] - Time[0]
tstart = 38200
tstart_idx = np.searchsorted(Time,tstart)
tend = 39201
tend_idx = np.searchsorted(Time,tend)
Time_steps = np.arange(0, tend_idx-tstart_idx)
Time = Time[tstart_idx:tend_idx]


#rotor data
p = a.groups["p_r"]; del a

x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points


normal = 29

#define plotting axes
coordinates = np.array(p.variables["coordinates"])


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
zs = np.linspace(p.origin[2],p.origin[2]+p.axis2[2],y)

dy = (max(ys) - min(ys))/x
dz = (max(zs) - min(zs))/y
dA = dy * dz

#velocity field
u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
del p

u[u<0]=0; v[v<0]=0 #remove negative velocities

with Pool() as pool:
    u_pri = []
    for u_pri_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_pri.append(u_pri_it)
        print(len(u_pri),time.time()-start_time)
u_pri = np.array(u_pri); del u; del v

print("line 126",time.time()-start_time)

#find vmin and vmax for isocontour plots            
#min and max over data
cmin = math.floor(np.min(u_pri))
cmax = math.ceil(np.max(u_pri))


nlevs = int((cmax-cmin)/2)
if nlevs>abs(cmin) or nlevs>cmax:
    nlevs = min([abs(cmin),cmax])+1

levs_min = np.linspace(cmin,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax,nlevs,dtype=int)
levels = np.concatenate((levs_min,levs_max[1:]))

print("line 141",levels)


def Update_locs(it):

    #algorithm for ejections

    U_pri = u_pri[it] #velocity time step it

    u_plane = U_pri.reshape(y,x)

    H = np.zeros(len(ys))
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)-1):

            if u_plane[k+1,j] > threshold:
                H[j] = zs[k]
                break

    return H




ncfile = Dataset(in_dir+"Threshold_heights_Dataset.nc",mode="w",format="NETCDF4")
ncfile.title = "Heights at threshold data sampling output"

#create global dimensions
sampling_dim = ncfile.createDimension("sampling",None)
y_dim = ncfile.createDimension("num_points",None)

Time_sampling = ncfile.createVariable("Time", np.float64, ('sampling',),zlib=True)
Time_sampling[:] = Time
y_locs = ncfile.createVariable("ys", np.float64, ('num_points',),zlib=True)
y_locs[:] = ys

#thresholds to output data
thresholds = [-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.4]


for threshold in thresholds:

    print("line 293",threshold)

    group = ncfile.createGroup("{}".format(abs(threshold)))

    H_ejection = group.createVariable("Height_ejection", np.float64, ('sampling','num_points'),zlib=True)

    H_array = []
    ix = 1
    with Pool() as pool:
        for H_it in pool.imap(Update_locs,Time_steps):

            H_array.append(H_it)

            print(ix,time.time()-start_time)

            ix+=1

    H_ejection[:] = np.array(H_array); del H_array

    print(ncfile.groups)


print(ncfile)
ncfile.close()